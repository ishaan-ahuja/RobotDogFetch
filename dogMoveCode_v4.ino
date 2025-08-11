#include "ServoDriver.h"
#include "Utils.h"
#include "Ultrasonic.h"
#include "Gait.h"
#include "Matrix.h"

#include <Wire.h>

#include "EvoMPU6050Simple.h"

EvoMPU6050Simple mpu;

bool got_imu = false;
char balance_mode = 0;

// steps to take, saved times 4, cuz 1 for each leg
// max steps is 32 (or 31? not sure)
signed char steps_left = -1;

#define FIRMWARE_VER 21

#define DEBUG_FLAG  1
#define PAUSE_FLAG  2
#define POWER_FLAG  4
#define BARK_FLAG   8
#define IMU_FLAG    16

#define BARK_PLAY_E 8

struct Flags {
  unsigned char flags;
  bool hasFlag(unsigned char flag) {
    return (flags & flag);
  }
  void addFlag(unsigned char flag) {
    flags |= flag;
  }
  void removeFlag(unsigned char flag) {
    flags &= ~flag;
  }
};
Flags flags = { PAUSE_FLAG };

ServoDriver servoDriver;

void moveServo(char servoId, float pos) {
  servoDriver.gotoPos(servoId, pos + 90.0);
}

void printVal(const char* name, float val) {
  Serial.print(name);
  Serial.print(": ");
  Serial.println(val);
}

void printPosFloat(float* pos, const char* prefix = "") {
  // this is broken lol
  /*
  Serial.print(prefix);
  Serial.print(pos[0]);
  Serial.print(',');
  Serial.print(pos[1]);
  Serial.print(',');
  Serial.println(pos[2]);
  */
}

void printPosChar(signed char* pos, const char* prefix = "") {
  Serial.print(prefix);
  Serial.print((float)pos[0]);
  Serial.print(',');
  Serial.print((float)pos[1]);
  Serial.print(',');
  Serial.print((float)pos[2]);
}

// measurements in mm or whatever
#define ROTATE_RADIUS 31
#define THIGH_LEN 100.0
#define CALF_LEN 125.3

#define BODY_LEN 222
#define BODY_WID 86
#define HALF_BODY_LEN 111
#define HALF_BODY_WID 43

#define ROTATE 0
#define HIP_JOINT 1
#define KNEE_JOINT 2

#define FRONT_LEFT 0
#define BACK_LEFT  1
#define FRONT_RIGHT 2
#define BACK_RIGHT 3
#define FRONT_SIDE 4
#define BACK_SIDE 5
#define LEFT_SIDE 6
#define RIGHT_SIDE 7
#define ALL_LEGS 8

#define REVERSED_FLAG 1
#define XYZ_FLAG 2

struct LegInfo {
  char servoIDs[3]; // servo numbers for leg parts
  signed char startPos[3]; // initial pos, keep in -127 to 127

  signed char offsetPos[3]; // offsets to zero it out, installation stuff
  Flags flags;
};

#define TOTAL_LEGS 4
LegInfo legInfo[] = {
  { {0,1,2}, {0,0,0}, {0,0,-45}, 0 },  // front left
  { {3,4,5}, {0,0,0}, {0,0,-45}, 0 },  // back left
  { {6,7,8}, {0,0,0}, {0,0,45}, REVERSED_FLAG }, // front right
  { {9,10,11}, {0,0,0}, {0,0,45}, REVERSED_FLAG } // back right
};

signed char offsets[4][3] = {0};
signed char gaitOffsets[4][3] = {0};

// robot position vars
bool balancing = false;
signed char robotPos[3] = {0,0,0}; // x,y,z pos
int robotRot[3] = {0}; // x,y,z rotation times 10
float bodyRotation[3][3] = {0};

int imuAngles[3] = {0}; // imu data * 10

void updateBodyRotation() {
  // update rotation matrix based on robotRot
  get3drot((float)robotRot[0]/10.0, (float)robotRot[1]/10.0, (float)robotRot[2]/10.0, bodyRotation);
}

#define MOUTH_MAX_OPEN 45
#define MOUTH_CLOSED 0
#define MOUTH_OPENED 20
signed char mouthPos = MOUTH_CLOSED;
signed char mouthTarget = MOUTH_CLOSED;

GaitLegParam trotGait = {
  // x,y,z: amplitude, center, phase, clipmin, clipmax
  { {30,0,0,-100,100},
    {20,190,90,-100,20},
    {0,0,0,-100,100}
  },
  290, // touchdown start
  70,  // touchdown end
  300, // down time ms
  130  // up time ms
};

GaitRobot trotRobots[4] = {
  { trotGait, 0, NULL, 0, {25,0,0} },
  { trotGait, 180, NULL, 0, {45,0,0} },
  { trotGait, 180, NULL, 0, {25,0,0} },
  { trotGait, 0, NULL, 0, {45,0,0} },
};

GaitLegParam walkGait = {
  { {30,0,0,-100,100},
    {20,190,90,-90,20},
    {0,0,0,-100,100}
  },
  290,
  70,
  600,
  300
};

signed char walkCGOffsetsX[30] = {5,5,0,0,0,0,0,-5,-5,-5,0,0,0,0,5,5,5,0,0,0,0,-5,-5,-5,0,0,0,0,0,0};
signed char walkCGOffsetsZ[30] = {20,30,40,40,40,40,40,40,40,40,40,30,20,0,-20,-30,-40,-40,-40,-40,-40,-40,-40,-40,-40,-40,-40,-30,-20,0};
signed char* walkCGOffsets[3] = {walkCGOffsetsX, NULL, walkCGOffsetsZ};

GaitRobot walkRobots[4] = {
  { walkGait, 0, walkCGOffsets, 30, {25,0,0} },
  { walkGait, 90, NULL, 0, {45,0,0} },
  { walkGait, 180, NULL, 0, {25,0,0} },
  { walkGait, 270, NULL, 0, {45,0,0} },
};

GaitRobot* activeGait = trotRobots;
#define GAIT_STEP_DEG 12
signed char turnRight = 0;
signed char shuffleRight = 0;
bool gaitDir = true;

void resetGaits() {
  for(char i=0; i<4; i++){
    trotRobots[i].gait = trotGait;
    walkRobots[i].gait = walkGait;
    turnRight = 0;
    shuffleRight = 0;
    gaitDir = true;
  }
}

GaitLegParam* getGaitParam(GaitRobot* gait) {
  if (gait == trotRobots)
    return &trotGait;
  return &walkGait;
}

class Leg {
  public:
    char id;
    float target[3];
    float start[3];
    float current[3];

    int index = 0;
    unsigned long startTime = 0;
    unsigned long endTime = 0;

    void setup(char legId) {
      pinMode(2, OUTPUT); // servo enable pin
      digitalWrite(2, HIGH); // off by default
      pinMode(A0, INPUT); // ultrasonic echo
      pinMode(A1, OUTPUT); // ultrasonic trig

      id = legId;
      for(char i=0; i<3; i++) {
        current[i] = legInfo[legId].startPos[i];
        start[i] = current[i];
        target[i] = current[i];
      }
    }

    void printPos(bool addOffset) {
      Serial.print((int)id);
      for(char i=0; i<3; i++) {
        Serial.print('\t');
        Serial.print(current[i] + (addOffset ? offsets[id][i] : 0));
      }
      Serial.println();
    }

    //function below is all chatgpt math, hella funky, idk why it works

    void goTo(float* pos, int duration, bool reversed, unsigned long curTime) {
      for(char i=0; i<3; i++) {
        if(isnan(pos[i]) || isinf(pos[i]) || !isValidAngle(pos[i]))
          return;
      }
      startTime = curTime;
      endTime = curTime + duration;
      for(char i=0; i<3; i++) {
        start[i] = current[i];
        if(i == 0)
          target[i] = (id == 1 || id == 3) ? -pos[i] : pos[i];
        else
          target[i] = reversed ? -pos[i] : pos[i];

        int tpos = target[i] + legInfo[id].offsetPos[i];
        if(tpos < -85)
          target[i] = -85 - legInfo[id].offsetPos[i];
        else if(tpos > 85)
          target[i] = 85 - legInfo[id].offsetPos[i];
      }
    }
    void goTo(int* pos, int duration, bool reversed, unsigned long curTime) {
      float fpos[3] = { pos[0], pos[1], pos[2] };
      goTo(fpos, duration, reversed, curTime);
    }

    void goTo(float* pos, int duration, bool reversed, unsigned long curTime, bool isXYZ) {
      if(isXYZ) {
        // do some funky math to rotate body etc.
        float legOrigin[3] = { (id==0 || id==2 ? HALF_BODY_LEN : -HALF_BODY_LEN), (id<2 ? HALF_BODY_WID : -HALF_BODY_WID), 0};
        add(pos, offsets[id]);
        float footPos[3] = {
          -pos[0] + legOrigin[0] - robotPos[0],
          -pos[2] + legOrigin[1] - robotPos[1],
          -pos[1] - robotPos[2]
        };
        float virtFootPos[3];
        multtranspose(bodyRotation, footPos, virtFootPos);
        pos[0] = -(virtFootPos[0] - legOrigin[0]);
        pos[1] = -virtFootPos[2];
        pos[2] = -(virtFootPos[1] - legOrigin[1]);

        float angPos[3];
        if(calcAnglesXYZ(pos[0], pos[1], pos[2], angPos[0], angPos[1], angPos[2]))
          goTo(angPos, duration, reversed, curTime);
        else if(flags.hasFlag(DEBUG_FLAG))
          printVal("E", 0);
      }
      else {
        goTo(pos, duration, reversed, curTime);
      }
    }
    void goTo(int* pos, int duration, bool reversed, unsigned long curTime, bool isXYZ) {
      float fpos[3] = { pos[0], pos[1], pos[2] };
      goTo(fpos, duration, reversed, curTime, isXYZ);
    }
    void goTo(signed char* pos, int duration, bool reversed, unsigned long curTime, bool isXYZ) {
      float fpos[3] = { pos[0], pos[1], pos[2] };
      goTo(fpos, duration, reversed, curTime, isXYZ);
    }

    void updateTarget(float* newTarget, unsigned long curTime, int duration = 500) {
      if(endTime == 0 || curTime > endTime)
        goTo(newTarget, duration, legInfo[id].flags.hasFlag(REVERSED_FLAG), curTime, true);
      else
        goTo(newTarget, endTime - startTime, legInfo[id].flags.hasFlag(REVERSED_FLAG), startTime, true);
    }

    void directGoTo(float* pos) {
      for(char j=0; j<3; j++) {
        if(pos[j] < -500 || pos[j] < -150 || pos[j] > 150) return; // no way
        moveServo(legInfo[id].servoIDs[j], legInfo[id].offsetPos[j] + pos[j]);
        current[j] = pos[j];
        target[j] = pos[j];
      }
    }
};
  void directGoto( int* pos )
    {
      float fpos[3] = { pos[0], pos[1], pos[2] };
      directGoto( fpos );
    }
    void loop( unsigned long curtime )
    {
      if (end_time == 0 )
        return;
      for ( char i = 0; i < 3; i++ )
      {
        //desired position at this time in the cycle
        cur_pos[i] = ( float(curtime - start_time ) /
                       (float)(end_time - start_time) * (float)(target_pos[i] - start_pos[i]) ) +
                     (float)start_pos[i];
      }

      for ( char i = 0; i < 3; i++ )
      {
        int pos = cur_pos[i] + leg_info[leg_id].ninety_offset_pos[i]; //add the installation error adjustment
        servogoto( leg_info[leg_id].ids[i], pos );
      }
      if ( curtime > end_time )
      {
        end_time = 0; //done, mark motion as complete
      }
    }
    bool calcAngles( float x, float y, float& h, float& k ) //from xy position
    {
      double max_length = LTHIGH + LCALF;

      double l = sqrt(x * x + y * y); //length of leg in desired  position
      if ( l > max_length )
        l = max_length;
      //printf( "length: %0.1lf\r\n", l );
      //now upper leg and lower leg and l form a triangle
      //this is angle from l, not vertical
      double l_hip = getangleabc( LTHIGH, l, LCALF);
      //printf( "l_hip: %0.1lf\r\n", l_hip );
      double l_angle = getangleabc( l, y, x );
      if ( x < 0 )
        l_angle = -l_angle;
      //printf( "l_angle: %0.1lf\r\n", l_angle );
      double hip_angle = l_hip + l_angle;

      //knee angle
      double knee_angle = 180.0 - getangleabc( LCALF, LTHIGH, l );

      h = hip_angle;
      k = knee_angle;
      float p = h + leg_info[leg_id].ninety_offset_pos[1];

      if ( p < -85 or p > 85 )
      {
        return false;
      }

      p = (leg_info[leg_id].flags.isset(REVERSED) ? -k : k) + leg_info[leg_id].ninety_offset_pos[2];
      return ( p >= -85 and p <= 85 );

      //clip( hip_angle, -60.0, 80.0 );
      //clip( knee_angle, -90.0, 135.0 );
    }
    bool calcAnglesXYZ( float x, float y, float z, float& r, float& h, float& k ) //from xyz position
    {
      float max_length = LTHIGH + LCALF;
      if ( x < -max_length || x > max_length || y < 0 || y > max_length ||
           z < -150 || z > 150 )
      {
        if ( flags.isset( FLAG_DEBUG) )
          println( "E", 1 );
        return false;
      }
      double oplusz = ROTRADIUS + z;
      double dsq = y * y + oplusz * oplusz;
      double d = sqrt( dsq );
      double cosa = oplusz / d;
      double a = acos( cosa );
      double cosaplusr = ROTRADIUS / d;
      double aplusr = acos( cosaplusr );

      r = aplusr - a;//in radians
      if ( isnan(r ) )
      {
        if ( flags.isset( FLAG_DEBUG) )
          println( "E", 2 );
        return false;
      }

      r = r * RAD_TO_DEG;
      r = -r;


      double lsq = dsq - ROTRADIUS * ROTRADIUS;
      double l = sqrt( lsq );
      //if ( debug )
      //  println( "r", r );
      //now the 2d part
      return calcAngles( x, l, h, k );
    }




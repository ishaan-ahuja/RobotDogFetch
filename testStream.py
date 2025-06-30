import torch
import cv2
import numpy as np
import time
import requests



# time.sleep(2)  

# retry_count = 0
# max_retries = 5

# prev_frame_time = 0
# new_frame_time = 0

# url=("http://10.0.0.200:8000/frame.jpg")


# while retry_count < max_retries:

#     response= requests.get(url)

#     # ret, frame = cap.read()
#     response= requests.get(url)
#     nparr=np.frombuffer(response.content, np.uint8)
#     frame=cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)


#     # if not ret:
#     #     print(f"Error: Could not read frame. ({retry_count + 1}/{max_retries})")
#     #     retry_count += 1
#     #     time.sleep(1) 
#     #     continue
    
#     new_frame_time = time.time() 

#     fps = 1/(new_frame_time-prev_frame_time) 
#     prev_frame_time = new_frame_time 
#     fps = int(fps) 
#     fps = str(fps) 
#     print(fps)

#     retry_count = 0  
#     time.sleep(0.03)

#     # cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA) 

#     # cv2.imshow('frame', frame)
#     # if cv2.waitKey(10) & 0xFF == ord('q'):
#     #     break



prev_frame_time = 0
new_frame_time = 0
# cv2.destroyAllWindows()
vid = cv2.VideoCapture("http://10.0.0.200:8000/stream.mjpg") 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    #qcv2.imshow('frame', frame) 

    new_frame_time = time.time() 

    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    fps = int(fps) 
    fps = str(fps) 
    print(fps)

    retry_count = 0  

    #cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA) 
    #cv2.imshow('frame', frame)

    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows()
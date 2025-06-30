import torch
import cv2
import numpy as np
import time
import requests

model_path = 'runs/train/exp7/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.eval()

time.sleep(2)

prev_frame_time = 0
new_frame_time = 0

url = "http://10.0.0.200:8000/frame.jpg"

commandCounter = 0

def sendCommand(cmd):
    global commandCounter

    commandUrl = f"http://10.0.0.200:9072/command?cmd=direct&v1={cmd}"
    requests.get(commandUrl)
    commandCounter += 1
    print(f"sending command: {cmd}")

def getInfo(cmd):
    commandUrl = f"http://10.0.0.200:9072/command?cmd=direct&v1={cmd}"
    response = requests.get(commandUrl)
    return response.text

right = 0
left = 0

stepDis = 24

posNum = 0

direction = -1

while True:
    response = requests.get(url)
    nparr = np.frombuffer(response.content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)

    image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(image_array)

    boxes = results.xyxy[0].numpy()
    labels = results.names
    scores = results.xyxy[0][:, 4].numpy()

    cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    right = 0
    left = 0

    if direction != -1:
        sendCommand("gs")
        sendCommand(f"gl, {direction}")

    for box in boxes:
        x1, y1, x2, y2, score, class_id = box
        
        if score > 0.4:
            label = labels[int(class_id)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f'{label} {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            distance = 7800 / ((x2 - x1) + (y2 - y1) / 2)
            cv2.putText(frame, f'{distance:.2f}', (int(x1), int(y2) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            center_x = (x1 + x2) / 2
            frame_center_x = frame.shape[1] / 2

            if center_x < frame_center_x - 50:
                sendCommand("rg")
                time.sleep(0.2)
                sendCommand("gs")
                time.sleep(0.2)
                sendCommand("gl, 3")
                direction = 2
                print("GOING LEFT")
            elif center_x > frame_center_x + 50:
                sendCommand("rg")
                time.sleep(0.2)
                sendCommand("gs")
                time.sleep(0.2)
                sendCommand("gr, 3")
                direction = -2
                print("GOING RIGHT")
            else:
                sendCommand("rg")
                time.sleep(0.2)
                sendCommand("gs")
                direction = 0

            steps = int(distance / stepDis)

            sendCommand("r,0,0,0")
            sendCommand(f"trot, {steps}")
            print(f"GOING STRAIGHT {steps}")
            #time.sleep(2)

    cv2.imshow('Detected Objects', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if commandCounter != 0:
        curInfo = getInfo("i")
        time.sleep(0.7)
        lateInfo = getInfo("i")
        if curInfo[-150:] == lateInfo[-150:]:
            break

requests.get("http://10.0.0.200:9072/command?cmd=sequence&v1=down")

time.sleep(2)

response = requests.get(url)
nparr = np.frombuffer(response.content, np.uint8)
frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

new_frame_time = time.time()
fps = 1 / (new_frame_time - prev_frame_time)
prev_frame_time = new_frame_time
fps = int(fps)
fps = str(fps)

image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = model(image_array)

boxes = results.xyxy[0].numpy()
labels = results.names
scores = results.xyxy[0][:, 4].numpy()

cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

for box in boxes:
    x1, y1, x2, y2, score, class_id = box
    
    if score > 0.4:
        label = labels[int(class_id)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f'{label} {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        distance = 7800 / ((x2 - x1) + (y2 - y1) / 2)
        cv2.putText(frame, f'{distance:.2f}', (int(x1), int(y2) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        center_x = (x1 + x2) / 2
        frame_center_x = frame.shape[1] / 2

        if center_x < frame_center_x - 30:
            sendCommand("rg")
            time.sleep(0.2)
            sendCommand("gs")
            time.sleep(0.2)
            sendCommand("gl, 6")
            print("GOING LEFT BIG")
        elif center_x > frame_center_x + 30:
            sendCommand("rg")
            time.sleep(0.2)
            sendCommand("gs")
            time.sleep(0.2)
            sendCommand("gr, 6")
            print("GOING RIGHT BIG")
        else:
            sendCommand("rg")
            time.sleep(0.2)
            sendCommand("gs")

        steps = int(distance / 5)

        sendCommand("r,0,0,0")
        sendCommand(f"trot, {steps}")
        print(f"GOING STRAIGHT {steps}")
        time.sleep(steps/4)
    
sendCommand("rg")
sendCommand("gs")
sendCommand("trot, 5")
time.sleep(5)

sendCommand("tall")
time.sleep(1)
sendCommand("stand")
time.sleep(1)
requests.get("http://10.0.0.200:9072/command?cmd=sequence&v1=kickReady")
time.sleep(2)
sendCommand("s:3")
requests.get("http://10.0.0.200:9072/command?cmd=sequence&v1=kick")
sendCommand("s:1")

cv2.destroyAllWindows()

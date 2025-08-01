from flask import Flask, render_template, request, jsonify
import cv2, numpy as np, base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
COLOR_RANGES={'blue':([100,150,50],[140,255,255]),'green':([40,70,50],[80,255,255]),'red':([0,100,50],[10,255,255])}

@app.route('/')
def index(): return render_template('index.html')

def encode_img(img):
    _,buf=cv2.imencode('.jpg',img)
    return base64.b64encode(buf).decode('utf-8')

@app.route('/process',methods=['POST'])
def process():
    d=request.get_json(); color=d['color']
    img_b64=d['image'].split(',')[1]
    img_bytes=base64.b64decode(img_b64)
    frame=cv2.cvtColor(np.array(Image.open(BytesIO(img_bytes))),cv2.COLOR_RGB2BGR)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower,upper=COLOR_RANGES[color]
    mask=cv2.inRange(hsv,np.array(lower),np.array(upper))
    color_mask=cv2.bitwise_and(frame,frame,mask=mask)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(frame,(15,15),0)
    canny=cv2.Canny(gray,100,200)
    gray_bgr=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    canny_bgr=cv2.cvtColor(canny,cv2.COLOR_GRAY2BGR)
    return jsonify({
        'mask':'data:image/jpeg;base64,'+encode_img(color_mask),
        'gray':'data:image/jpeg;base64,'+encode_img(gray_bgr),
        'blur':'data:image/jpeg;base64,'+encode_img(blurred),
        'canny':'data:image/jpeg;base64,'+encode_img(canny_bgr)
    })

app.run(host='0.0.0.0',port=81)

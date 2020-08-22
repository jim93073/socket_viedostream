# YOLO應用-使用Flask網頁，透過socket通訊，並進行影像辨識
###### tags: `YOLO` `Flask` `socket`
![](https://i.imgur.com/yRwArHX.png )

Github:
```bash=
git clone https://github.com/jim93073/socket_viedostream.git
```

>Web: (需命名為index.html，放在templates資料夾底下)
1.接收Server的影像，並傳送使用者的按鈕。
2.按下Auto可開始影像辨識，Manual可暫停

>Server: (放在與templates同一層)
1.接收Web的資料，並儲存，等待client接收
2.傳送影像至Web
3.進行影像辨識(Samply.py)

>Client
1.監聽Server資料

Requirement:
```bash=
pip install websocket-client flask-sockets flask gevent flask_threaded_sockets opencv-python threaded
```
- Server (結合YOLO，github名稱為Sample.py)
需先配置環境:https://hackmd.io/kUecqBdITuilOVrpHi6x1A?view
```python=
from ctypes import *
import math
import random
import cv2
from flask import Flask,render_template, Response 
from flask_sockets import Sockets
import time
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
from flask_threaded_sockets import Sockets, ThreadedWebsocketServer
import threading


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/jim/Desktop/darknet_pjreddie/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


#ADD by jim 
ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    #im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image

#add by jim
#vid = cv2.VideoCapture(0)

app = Flask(__name__)
sockets = Sockets(app)

# 0:Status      
# 1:motor1
# 2:motor2
# 3:motor3

change = [False for i in range(5)] 
message = ""
Status = 1 # 0:auto 1:manual
motor1 = 0 # 0:正轉 1 反轉
motor2 = 0
motor3 = 0

vid = cv2.VideoCapture(0)



#Server
def server():
    server = ThreadedWebsocketServer("127.0.0.1", 5000, app)
    print("web server start ... ")
    server.serve_forever()
#end server

#object detect
def obj_detect():
    global Status
    while True:
        if Status == 0:
            return_value,arr=vid.read()
            try:
                im=nparray_to_image(arr)
            except:
                pass
            r = detect(net, meta, im)
            print (r)
#end object detect

# http 路由，访问url是： http://localhost:5000/
@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/status/auto')
def auto():    
    print("Rescived auto")
    change[0] = True
    global Status
    Status = 0
    return render_template('index.html', status="Auto")

@app.route('/status/manual')
def manual():    
    print("Rescived manual")
    change[0] = True
    global Status 
    Status = 1
    return render_template('index.html', status="Manual")

@app.route('/motor1/pos')
def motor1_pos():    
    print("Rescived motor1_pos")
    change[1] = True
    global motor1 
    motor1 = 0
    return render_template('index.html')

@app.route('/motor1/neg')
def motor1_neg():    
    print("Rescived motor1_pos")
    change[1] = True
    global motor1 
    motor1 = 1
    return render_template('index.html')

@app.route('/motor2/pos')
def motor2_pos():    
    print("Rescived motor2_pos")
    change[2] = True
    global motor2 
    motor2 = 0
    #return render_template('index.html')

@app.route('/motor2/neg')
def motor2_neg():    
    print("Rescived motor2_pos")
    change[2] = True
    global motor2 
    motor2 = 1
    return render_template('index.html')

@app.route('/motor3/pos')
def motor3_pos():    
    print("Rescived motor3_pos")
    change[3] = True
    global motor3 
    motor3 = 0
    return render_template('index.html')

@app.route('/motor3/neg')
def motor3_neg():    
    print("Rescived motor3_pos")
    change[3] = True
    global motor3 
    motor3 = 1
    return render_template('index.html')

@app.route('/motor4/pos')
def motor4_pos():    
    print("Rescived motor4_pos")
    change[4] = True
    global motor4 
    motor4 = 0
    return render_template('index.html')

@app.route('/motor4/neg')
def motor4_neg():    
    print("Rescived motor4_pos")
    change[4] = True
    global motor4
    motor4 = 1
    return render_template('index.html')



@sockets.route('/echo')
def echo_socket(ws):
    while not ws.closed:
        time.sleep(1)       
#         print(message)
        for i in range(5):
            if change[i] == True:
                change[i] = False #initial  
                print("Change: "+str(i))
                
#                 message = ws.receive()                
                if i == 0:       
                    ws.send(str(i)+str(Status))
                    print("Status:"+str(Status))
                if i == 1:      
                    ws.send(str(i)+str(motor1))
                if i == 2:   
                    ws.send(str(i)+str(motor2))
                if i == 3:               
                    ws.send(str(i)+str(motor3))
                if i == 4:               
                    ws.send(str(i)+str(motor4))
                
    print("OUT")

                




def gen():
    #video = cv2.VideoCapture(0)
    global vid
    video = vid
    while True:
        success, image = video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]

    
    net = load_net("cfg/yolov3.cfg".encode("utf8"), "yolov3.weights".encode("utf8"), 0)
    meta = load_meta("cfg/coco.data".encode("utf8"))
    #r = detect(net, meta, "data/eagle.jpg".encode("utf8"))


    #vid = cv2.VideoCapture(0)
    
    t1 = threading.Thread(target=server, args=())
    t1.start()
    obj_detect()
#    t2 = threading.Thread(target=obj_detect, args=())    
#    t2.start()
```

- client (github名稱為client)
```python=
from websocket import create_connection

def client_handle():
    ws = create_connection('ws://127.0.0.1:5000/echo')
    while ws.connected:        
#             ws.send('RECEIVED') 
        result = ws.recv()  
        print(result)            
            
        # ws.close()
    print("OUT")
if __name__ == "__main__":
    client_handle()
```



- Web (templates/index.html)
```html=
<html>

<head>
  <title>DD夜總會</title>
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css"
    integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">
  <style>
    img {
      display: block;
      margin-left: auto;
      margin-right: auto;
      width: 35%
    }
  </style>
</head>

<body>
  <h1 class="alert alert-info text-center">DD夜總會</h1>

  <div class="container ">
    <div class="row ">
      <div class="col-4 m-auto">
        <img src="chang.jpg" class="rounded-circle border border-primary ">
      </div>
      <div class="col-4 m-auto">
        <img src="chang.jpg" class="rounded-circle border border-primary ">
      </div>
      <div class="col-4 m-auto">
        <img src="chang.jpg" class="rounded-circle border border-primary ">
      </div>
      <div class="col-4 m-auto">
        <img src="chang.jpg" class="rounded-circle border border-primary ">
      </div>
      <div class="col-4 m-auto">
        <img src="chang.jpg" class="rounded-circle border border-primary ">
      </div>
    </div>
  </div>

  <div class="container my-5">
    <div class="row">
      <div class="col-6">
        <!-- <img src="{{ url_for('video_feed') }}"> -->
        <img src="chang.jpg" class="img-fluid img-thumbnail">
      </div>
      <div class="col-6">
        <div class="col-8 my-3 mx-auto">
          Status
          <button id="auto" class="btn btn-outline-primary">Auto</button>
          <button id="manual" class="btn btn-outline-primary">Manual</button>
        </div>
        <div class="col-8 my-3 mx-auto">
          Motor1
          <button id="motor1_pos" class="btn btn-outline-secondary">Positive</button>
          <button id="motor1_neg" class="btn btn-outline-secondary">Negative</button>
        </div>
        <div class="col-8 my-3 mx-auto">
          Motor2
          <button id="motor2_pos" class="btn btn-outline-secondary">Positive</button>
          <button id="motor2_neg" class="btn btn-outline-secondary">Negative</button>

        </div>
        <div class="col-8 my-3 mx-auto">
          Motor3
          <button id="motor3_pos" class="btn btn-outline-secondary">Positive</button>
          <button id="motor3_neg" class="btn btn-outline-secondary">Negative</button>
        </div>
        <div class="col-8 my-3 mx-auto">
          Motor4
          <button id="motor4_pos" class="btn btn-outline-secondary">Positive</button>
          <button id="motor4_neg" class="btn btn-outline-secondary">Negative</button>
        </div>
      </div>

    </div>
  </div>




  <!-- <button onclick="location.href='http://127.0.0.1:5000/status/auto'">Auto</button> -->
  <!-- <button id="manual" onclick="location.href='http://127.0.0.1:5000/status/manual'">Manual</button> -->


  <script>
    $("#manual").click(function () {
      $.ajax({
        type: "GET",
        url: "http://127.0.0.1:5000/status/manual",
        success: function (data) {
          console.log(data);
        }
      });
    });
    $("#auto").click(function () {
      $.ajax({
        type: "GET",
        url: "http://127.0.0.1:5000/status/auto",
        success: function (data) {
          console.log(data);
        }
      });
    });
    $("#motor1_pos").click(function () {
      $.ajax({
        type: "GET",
        url: "http://127.0.0.1:5000/motor1/pos",
      });
    });
    $("#motor1_neg").click(function () {
      $.ajax({
        type: "GET",
        url: "http://127.0.0.1:5000/motor1/neg",
      });
    });
    $("#motor2_pos").click(function () {
      $.ajax({
        type: "GET",
        url: "http://127.0.0.1:5000/motor2/pos",
      });
    });
    $("#motor2_neg").click(function () {
      $.ajax({
        type: "GET",
        url: "http://127.0.0.1:5000/motor2/neg",
      });
    });
    $("#motor3_pos").click(function () {
      $.ajax({
        type: "GET",
        url: "http://127.0.0.1:5000/motor3/pos",
      });
    });
    $("#motor3_neg").click(function () {
      $.ajax({
        type: "GET",
        url: "http://127.0.0.1:5000/motor3/neg",
      });
    });
    $("#motor4_pos").click(function () {
      $.ajax({
        type: "GET",
        url: "http://127.0.0.1:5000/motor4/pos",
      });
    });
    $("#motor4_neg").click(function () {
      $.ajax({
        type: "GET",
        url: "http://127.0.0.1:5000/motor4/neg",
      });
    });




  </Script>

  <!-- <div>Status: {{status}} </div> -->


</body>

</html>
```


- Server(無結合YOLO) (github名稱為server)
```python=
from flask import Flask,render_template, Response 
from flask_sockets import Sockets
import time
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
from flask_threaded_sockets import Sockets, ThreadedWebsocketServer
import cv2

app = Flask(__name__)
sockets = Sockets(app)

# 0:Status      
# 1:motor1
# 2:motor2
# 3:motor3

change = [False for i in range(5)] 
message = ""
Status = 0 # 0:auto 1:manual
motor1 = 0 # 0:正轉 1 反轉
motor2 = 0
motor3 = 0

# http 路由，访问url是： http://localhost:5000/
@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/status/auto')
def auto():    
    print("Rescived auto")
    change[0] = True
    global Status
    Status = 0
    return render_template('index.html', status="Auto")

@app.route('/status/manual')
def manual():    
    print("Rescived manual")
    change[0] = True
    global Status 
    Status = 1
    return render_template('index.html', status="Manual")

@app.route('/motor1/pos')
def motor1_pos():    
    print("Rescived motor1_pos")
    change[1] = True
    global motor1 
    motor1 = 0
    return render_template('index.html')

@app.route('/motor1/neg')
def motor1_neg():    
    print("Rescived motor1_pos")
    change[1] = True
    global motor1 
    motor1 = 1
    return render_template('index.html')

@app.route('/motor2/pos')
def motor2_pos():    
    print("Rescived motor2_pos")
    change[2] = True
    global motor2 
    motor2 = 0
    return render_template('index.html')

@app.route('/motor2/neg')
def motor2_neg():    
    print("Rescived motor2_pos")
    change[2] = True
    global motor2 
    motor2 = 1
    return render_template('index.html')

@app.route('/motor3/pos')
def motor3_pos():    
    print("Rescived motor3_pos")
    change[3] = True
    global motor3 
    motor3 = 0
    return render_template('index.html')

@app.route('/motor3/neg')
def motor3_neg():    
    print("Rescived motor3_pos")
    change[3] = True
    global motor3 
    motor3 = 1
    return render_template('index.html')

@app.route('/motor4/pos')
def motor4_pos():    
    print("Rescived motor4_pos")
    change[4] = True
    global motor4 
    motor4 = 0
    return render_template('index.html')

@app.route('/motor4/neg')
def motor4_neg():    
    print("Rescived motor4_pos")
    change[4] = True
    global motor4
    motor4 = 1
    return render_template('index.html')



@sockets.route('/echo')
def echo_socket(ws):
    while not ws.closed:
        time.sleep(1)       
#         print(message)
        for i in range(5):
            if change[i] == True:
                change[i] = False #initial  
                print("Change: "+str(i))
                
#                 message = ws.receive()                
                if i == 0:       
                    ws.send(str(i)+str(Status))
                    print("Status:"+str(Status))
                if i == 1:      
                    ws.send(str(i)+str(motor1))
                if i == 2:   
                    ws.send(str(i)+str(motor2))
                if i == 3:               
                    ws.send(str(i)+str(motor3))
                if i == 4:               
                    ws.send(str(i)+str(motor4))
                
    print("OUT")

                




def gen():
    video = cv2.VideoCapture(0)
    
    
    while True:
        success, image = video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

    
    
if __name__ == "__main__":    
#     server = pywsgi.WSGIServer(('127.0.0.1', 5000), app, handler_class=WebSocketHandler)
    server = ThreadedWebsocketServer("127.0.0.1", 5000, app)
    print("web server start ... ")
    server.serve_forever()
```
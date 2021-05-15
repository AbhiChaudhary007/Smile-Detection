# Importing the libraries
import cv2
from flask import Flask,Response
import os

app = Flask(__name__)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

# Loading the cascades
def get_path(filename):
    return os.path.join(face_file, "./", filename)

# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        #for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        if len(smiles) == 0:
            print(len(smiles))
            cv2.putText(frame, 'Smile Please', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    return frame

# Doing some Face Recognition with the webcam
def gen_frames():
    video_capture = cv2.VideoCapture(0)
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             canvas = detect(gray, frame)
             ret, buffer = cv2.imencode('.jpg', canvas)
             canvas = buffer.tobytes()
             yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + canvas + b'\r\n')
       
    video_capture.release()
    cv2.destroyAllWindows()

@app.route('/')
def video():
    return Response(gen_frames(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    face_file = os.path.dirname(os.path.abspath(__file__))
    filename = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(get_path(filename))
    filename = 'haarcascade_smile.xml'
    smile_cascade = cv2.CascadeClassifier(get_path(filename))
    app.run()
import base64
import json
from flask import request
from flask import (Flask, jsonify, render_template, Response)
from flask_cors import CORS
from livereload import Server
import cv2
import torch
from model_experiment1 import load_images, train
import face_recognition
import torchvision.transforms.functional as TF


app = Flask(__name__)
CORS(app)
camera = cv2.VideoCapture(0)

# x, y = load_images("data/images", "data/labels.csv")
# model = train(x, y, epochs=0, load=False, save=False)
# x = x.reshape(x.shape[0] * x.shape[1], 3, 224, 224)
# sample = model(x)
# name = ['Dang', 'Nguyen', 'Phuong Nguyen', 'Unknown']
# dictionary = {"name": "None"}


def generate_frames():
    while True:
        success, frame = camera.read()

        # locations = face_recognition.face_locations(frame)
        # if len(locations) != 0:
        #     for location in locations:
        #         x1 = location[3]
        #         y1 = location[0]
        #         w = location[1] - x1
        #         h = location[2] - y1
        #         predc = TF.to_tensor(cv2.resize(frame[y1:y1 + h, x1:x1 + w], (224, 224))).unsqueeze(0)
        #
        #         output = model(predc)
        #         dist = torch.norm(output - sample, dim=1, p=None)
        #         knn = dist.topk(1, largest=False)
        #         idx = torch.div(knn.indices, 8, rounding_mode='trunc')
        #
        #         top, right, bottom, left = location
        #         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
        #         cv2.rectangle(frame, (left, bottom - 17), (right, bottom), (0, 0, 255), cv2.FILLED)
        #         cv2.putText(frame, name[idx], (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        #
        #         dictionary["name"] = name[idx]
        #         # Serializing json
        #         json_object = json.dumps(dictionary, indent=4)
        #
        #         # Writing to sample.json
        #         with open("data.json", "w") as outfile:
        #             outfile.write(json_object)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/result')
def result():
    f = open("data.json")
    data = json.load(f)
    return render_template('result.html', jsonfile=json.dumps(data))


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
import csv
import json
import os

import cv2
import face_recognition
import numpy as np
from flask import (Flask, render_template, Response)
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def initialize():
    path = 'Images-2'
    persons = os.listdir(path)

    encodes = []
    face_names = []

    i = 1
    for person in persons:
        files = os.listdir(path + '/' + person)
        for _ in files:
            image = face_recognition.load_image_file(path + '/' + person + '/' + str(i) + '.png')
            encod = face_recognition.face_encodings(image)[0]
            encodes.append(encod)
        face_names.append(person)
    #
    return face_names, encodes


def read_file():
    # csv file name
    filename = "demofile.txt"

    # initializing the titles and rows list
    rows = []

    # reading csv file
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

    return rows


def check(name, rows):
    for row in rows:
        if row[0] == name:
            return bool(name[2]), row
        else:
            return False, row


face_names, encod_face = initialize()

locations = []
encodings = []

cap = cv2.VideoCapture(0)
rows = read_file()

studentImages = {"La Tran Hai Dang": "1.png", "Pham Gia Nguyen": "2.png", "Unknown": "0.png"}


def generate_frames():
    i = 0
    while True:
        _, frame1 = cap.read()
        _, frame2 = cap.read()
        name = 'Unknown'
        locations = face_recognition.face_locations(frame2)
        encodings = face_recognition.face_encodings(frame2, locations)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            for location in locations:
                distance = face_recognition.face_distance(encod_face, encodings[i])
                best_matches = np.argmin(distance)

                if distance[best_matches] < 0.6:
                    name = face_names[best_matches]

                top, right, bottom, left = location

                print(name)
                result, data = check(name, rows)
                cv2.rectangle(frame2, (left, top), (right, bottom), (0, 255, 0), 1)
                cv2.rectangle(frame2, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame2, "Face", (left + 7, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if name != 'Unknown':
                    status = "0 Dose"
                    if data[3] != "NULL" and data[4] != "NULL":
                        status = "2 Doses"
                    elif data[3] == "NULL" or data[4] == "NULL":
                        status = "1 Doses"
                    dictionary = {
                        "Full Name": name,
                        "Student Id": data[1],
                        "Expired Date": data[6],
                        "Campus": data[7],
                        "First Dose": data[3],
                        "Second Dose": data[4],
                        "Vaccine status": status,
                        "Infected": data[5],
                        "Health Declaration Date": data[8],
                        "Health Declaration Status": data[9],
                        "Access Permission": "Allowed" if result else "Denied",
                        "Image": studentImages[name]
                    }
                else:
                    dictionary = {
                        "Full Name": "Unknown",
                        "Student Id": "Unknown",
                        "Expired Date": "Unknown",
                        "Campus": "Unknown",
                        "First Dose": "Unknown",
                        "Second Dose": "Unknown",
                        "Vaccine status": "Unknown",
                        "Infected": "Unknown",
                        "Health Declaration Date": "Unknown",
                        "Health Declaration Status ": "Unknown",
                        "Access Permission": "NO",
                        "Image": studentImages[name]
                    }
                    print(studentImages[name])

                with open("templates/data.json", "w") as outfile:
                    json.dump(dictionary, outfile)
                i += 1
            i = 0

        cv2.imshow("Admin Page", frame2)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            pass


        ret, buffer = cv2.imencode('.jpg', frame1)
        frame1 = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/result')
def result():
    f = open("templates/data.json")
    data = json.load(f)
    print(data['Image'])
    return render_template('result.html', jsonfile=json.dumps(data), studentImage=f"{data['Image']}")


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
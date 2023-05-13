from flask import Flask, render_template, request, flash, redirect, Response
import cv2
import numpy as np
import face_recognition
import os
import csv

app = Flask(__name__)

path = 'training-images'
images = []
voterIDs = []
voterList = os.listdir(path)

for vid in voterList:
    curImg = cv2.imread(f'{path}/{vid}')
    images.append(curImg)
    voterIDs.append(os.path.splitext(vid)[0])


class Voter:
    def __init__(self, name):
        self.name = name
        self.voted = False

    def markAsVoted(self):
        if self.voted:
            return "Already Voted"
        else:
            with open('Voters-List.csv', 'a+', newline='') as f:
                writer = csv.writer(f)
                f.seek(0)
                voterDataList = [line for line in csv.reader(f)]
                voter_names = [row[1] for row in voterDataList]

                if self.name in voter_names:
                    self.voted = True
                    return "Already Voted"
                else:
                    writer.writerow([len(voterDataList) + 1, self.name, "Voted"])
                    self.voted = True
                    return "Voted"


def findFaceEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faceEncode = face_recognition.face_encodings(img)[0]
        encodeList.append(faceEncode)
    return encodeList


knownVoterFaceEncodings = findFaceEncodings(images)

cap = cv2.VideoCapture(0)

voters = []
for name in voterIDs:
    voters.append(Voter(name.upper()))


def gen_frames():
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        faceEncodingsCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for faceEncode, faceLoc in zip(faceEncodingsCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(knownVoterFaceEncodings, faceEncode)
            faceDis = face_recognition.face_distance(knownVoterFaceEncodings, faceEncode)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                voter = voters[matchIndex]
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                color = (0, 255, 0)

                if voter.voted:
                    color = (0, 0, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                cv2.putText(img, voter.name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                voter.markAsVoted()

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        flash('No file uploaded!')
        return render_template('index.html')

    file = request.files['file']

    if file.filename == '':
        flash('No file selected!')
        return render_template('index.html')

    if file:
        file_path = os.path.join(path, file.filename)
        file.save(file_path)

        unknown_image = face_recognition.load_image_file(file_path)
        unknown_encoding = face_recognition.face_encodings(unknown_image)

        if len(unknown_encoding) > 0:
            unknown_encoding = unknown_encoding[0]

            for voter in voterList:
                match = face_recognition.compare_faces([voter['encoding']], unknown_encoding)
                if match[0]:
                    flash('Voter Identified: {}'.format(voter['name']))
                    return render_template('index.html')

            flash('Unknown Voter!')
            return render_template('index.html')

        else:
            flash('No face detected in the uploaded image!')
            return render_template('index.html')

    flash('Error processing the file!')
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Check if the login credentials are correct (e.g., admin:admin)
        if username == 'admin' and password == 'admin':
            return redirect('/voted_persons')
        else:
            flash('Invalid login credentials!', 'error')

    return render_template('login.html')


@app.route('/voted_persons')
def voted_persons_list():
    return render_template('voted_persons.html', voted_persons=voterList)


def draw_square(frame, coordinates):
    top, right, bottom, left = coordinates
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)


def process_frame(frame):
    rgb_frame = frame[:, :, ::-1]  # Convert BGR frame to RGB

    # Find faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the detected face matches with any known voter
        matches = face_recognition.compare_faces([voter[0] for voter in voterList], face_encoding)

        if True in matches:
            matched_voter = voterList[matches.index(True)]

            # Check if the voter has already voted
            if matched_voter[1] in voterList:
                flash('You have already voted!', 'error')
                continue

            # Add the voted person to the list
            voterList.append(matched_voter[1])

            # Save the voter's details in a CSV file
            with open('voted_persons.csv', 'a') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow([matched_voter[1]])

            flash('Voting Successful!', 'success')

        draw_square(frame, (top, right, bottom, left))

    return frame


if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)

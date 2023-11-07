from flask import Flask, render_template, request, flash, redirect, Response
import cv2
import numpy as np
import face_recognition
import os
import csv

# Initialize the Flask app
app = Flask(__name__)

# Set the directory path where the training images are stored
path = 'training-images'

# Set the directory for serving static files (e.g., CSS, JS, images)
app.static_folder = 'static'

# Initialize empty lists to store the training images and their corresponding voter IDs
images = []
voterIDs = []

# Get the list of files in the specified directory
voterList = os.listdir(path)

# Read in the training images and store them in the images list, and store the corresponding voter IDs in the
# voterIDs list
for vid in voterList:
    curImg = cv2.imread(f'{path}/{vid}')
    images.append(curImg)
    voterIDs.append(os.path.splitext(vid)[0])

# Define a Voter class to store voter information
class Voter:
    def __init__(self, name):
        self.name = name
        self.voted = False

    def markAsVoted(self):
        if self.voted:
            return "Already Voted"
        else:
            # Open the CSV file for append mode which means both reading and writing
            with open('Voters-List.csv', 'a+', newline='') as f:
                # Initialize a CSV writer object
                writer = csv.writer(f)

                # Go to the beginning of the file
                f.seek(0)

                # Read in the contents of the file and store them in a list
                voterDataList = [line for line in csv.reader(f)]

                # Extract the names of the voters from the list of voter data
                voter_names = [row[1] for row in voterDataList]

                # Check if the current voter has already voted
                if self.name in voter_names:
                    self.voted = True
                    return "Already Voted"
                else:
                    # If the current voter has not already voted, write their name and "Voted" to the CSV file
                    writer.writerow([len(voterDataList) + 1, self.name, "Voted"])
                    self.voted = True
                    return "Voted"


def findFaceEncodings(images):
    encodeList = []

    for img in images:
        # Convert the image from BGR to RGB color space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Encode the face in the image using face_recognition library
        faceEncode = face_recognition.face_encodings(img)[0]
        # Append the face encoding
        encodeList.append(faceEncode)
    return encodeList


# Get the face encodings for the known voters' images
knownVoterFaceEncodings = findFaceEncodings(images)

# Open a video capture device
cap = cv2.VideoCapture(0)

# Create an empty list to store voter objects
voters = []

# Loop through each voter ID and create a Voter object with the uppercase name
for name in voterIDs:
    voters.append(Voter(name.upper()))

# Define a function to generate video frames
def gen_frames():
    # Use the global variable current_user
    global current_user

    # Loop indefinitely
    while True:
        # Read a frame from the video capture device
        success, img = cap.read()

        # Resize the frame to a quarter of its size to improve it's speed
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)

        # Convert the color space of the frame from BGR to RGB
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Detect the face locations in the current frame
        facesCurFrame = face_recognition.face_locations(imgS)

        # Encode the faces in the current frame
        faceEncodingsCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        # Iterate through each face encoding and location in the current frame
        for faceEncode, faceLoc in zip(faceEncodingsCurFrame, facesCurFrame):
            # Compare the face encoding to the known voter face encodings
            matches = face_recognition.compare_faces(knownVoterFaceEncodings, faceEncode)

            # Calculate the face distances between the current face encoding and the known voter face encodings
            faceDis = face_recognition.face_distance(knownVoterFaceEncodings, faceEncode)

            # Get the index of the known voter face encoding with the smallest face distance
            matchIndex = np.argmin(faceDis)

            # If the current face encoding matches a known voter face encoding
            if matches[matchIndex]:
                # Get the corresponding voter object
                voter = voters[matchIndex]

                # Set the current recognized user to the name of the voter
                current_user = voter.name

                # Get the face location coordinates in the original size frame
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                # Set the color of the rectangle based on whether the voter has voted
                color = (0, 255, 0)
                if voter.voted:
                    color = (0, 0, 255)

                # Draw a rectangle around the face in the original size frame
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Draw a filled rectangle below the face in the original size frame
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)

                # Write the name of the voter below the face in the original size frame
                cv2.putText(img, voter.name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                # Mark the voter as voted
                voter.markAsVoted()

        # Encode the processed frame as a JPEG image
        ret, buffer = cv2.imencode('.jpg', img)

        # Convert the image buffer to bytes
        frame = buffer.tobytes()

        # Yield the frame as a multipart MIME message
        yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r'


# Render the index.html template
@app.route('/')
def index():
    return render_template('index.html')


# Stream video frames as a response
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Global variable to track authentication
authenticate = False

# Handle the login page
@app.route('/login', methods=['POST'])
def login():
    # Declare authenticate as global
    global authenticate

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the login credentials are correct (e.g., admin:admin)
        if username == 'admin' and password == 'admin':
            authenticate = True
            # Redirect the user to the voted_persons page.
            return redirect('/voted_persons')  
        else:
             # If the username or password are incorrect, flash a message to the user.
            flash('Invalid username or password', 'danger') 
            # Redirect the user back to the login page.
            return redirect('/login')  
        
    # If the request method is not POST, render the login template.
    return render_template('login.html')

# Render the list of voted persons
@app.route('/voted_persons')
def voted_persons_list():
    # Check if the user is authenticated, if not, redirect them to the access_denied page.
    if not authenticate:  
        return redirect('/access_denied')
    # Get the names of all voted persons.
    voted_persons = [voter.name for voter in voters if voter.voted]  
    # Render the template with the list of voted persons and the current user.
    return render_template('voted_persons.html', voted_persons=voted_persons, current_user=current_user)  

@app.route('/access_denied')
def access_denied():
    # Render the access_denied template if the user is not authenticated.
    return render_template('access_denied.html')  

# Function to draw a square around a face in a frame
def draw_square(frame, coordinates):
    top, right, bottom, left = coordinates
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)


# Function to process a frame
def process_frame(frame):
    # Convert BGR frame to RGB
    rgb_frame = frame[:, :, ::-1]

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
                flash('You have already voted!', 'warning')
                continue

            # Check if the voter is a registered voter
            if matched_voter[1] not in voterIDs:
                flash('You are not a registered voter!', 'danger')
                continue

            # Add the voted person to the list
            voterList.append(matched_voter[1])

            # Save the voter's details in a CSV file
            with open('voted_persons.csv', 'a') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow([matched_voter[1]])

            flash('You May Go For Voting!', 'success')

        draw_square(frame, (top, right, bottom, left))

    return frame


# Run Flask application on server
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)

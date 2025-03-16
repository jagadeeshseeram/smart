import datetime
import os
import shutil
import smtplib
import time
from email.mime.multipart import MIMEMultipart

import cv2
import numpy as np
import pandas as pd
import pymysql
import requests
from flask import (Flask, flash, redirect, render_template, request, session,
                   url_for)
from flask_mail import *
from PIL import Image

facedata = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(facedata)

mydb=pymysql.connect(host='localhost', user='root', password='', port=3306, database='smart_voting_system')

app=Flask(__name__)
app.config['SECRET_KEY']='ajsihh98rw3fyes8o3e9ey3w5dc'

@app.before_request
def initialize():
    if 'IsAdmin' not in session:
        session['IsAdmin'] = False
    if 'User' not in session:
        session['User'] = None


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/admin', methods=['POST','GET'])
def admin():
    if request.method=='POST':
        email = request.form['email']
        password = request.form['password']
        if (email=='admin@voting.com') and (password=='admin'):
            session['IsAdmin']=True
            session['User']='admin'
            flash('Admin login successful','success')
    return render_template('admin.html', admin=session['IsAdmin'])

@app.route('/add_nominee', methods=['POST','GET'])
def add_nominee():
    if request.method=='POST':
        member=request.form['member_name']
        party=request.form['party_name']
        logo=request.form['test']
        nominee=pd.read_sql_query('SELECT * FROM nominee', mydb)
        all_members=nominee.member_name.values
        all_parties=nominee.party_name.values
        all_symbols=nominee.symbol_name.values
        if member in all_members:
            flash(r'The member already exists', 'info')
        elif party in all_parties:
            flash(r"The party already exists", 'info')
        elif logo in all_symbols:
            flash(r"The logo is already taken", 'info')
        else:
            sql="INSERT INTO nominee (member_name, party_name, symbol_name) VALUES (%s, %s, %s)"
            cur=mydb.cursor()
            cur.execute(sql, (member, party, logo))
            mydb.commit()
            cur.close()
            flash(r"Successfully registered a new nominee", 'primary')
    return render_template('nominee.html', admin=session['IsAdmin'])

@app.route('/registration', methods=['GET', 'POST'])
def registration():
    if 'User' not in session:
        flash('Please login first!', 'warning')
        return redirect(url_for('login'))  # Ensure 'login' route exists!

    if request.method == 'POST':
        Fullname = request.form['Fullname']
        aadhar_id = request.form['aadhar_id']
        College_id = request.form['College_id']
        email = request.form.get('email')  # Ensure email is provided

        if not email:  # If email is missing, show an error
            flash('Email is required!', 'danger')
            return redirect(url_for('registration'))

        voters = pd.read_sql_query('SELECT * FROM voters', mydb)
        if aadhar_id in voters['aadhar_id'].values:
            flash('Already registered as a voter.', 'info')
            return redirect(url_for('home'))

        sql = "INSERT INTO voters (Fullname, aadhar_id, College_id, email, verified) VALUES (%s, %s, %s, %s, %s)"
        cur = mydb.cursor()
        cur.execute(sql, (Fullname, aadhar_id, College_id, email, 'no'))
        mydb.commit()
        cur.close()

        session['aadhar'] = aadhar_id
        session['email'] = email  # Store email in session
        session['status'] = 'no'
        flash("Voter registration successful!", "success")
        return redirect(url_for('capture_images'))  # Skip OTP step
    return render_template('voter_reg.html')




@app.route('/train', methods=['POST', 'GET'])
def train():
    if request.method == 'POST':
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, Id = getImagesAndLabels(r"all_images")
        recognizer.train(faces, np.array(Id))
        recognizer.save("Trained.yml")
        flash("Model Trained Successfully", 'success')
        return redirect(url_for('home'))
    return render_template('train.html')



import pickle


@app.route('/capture_images', methods=['POST', 'GET'])
def capture_images():
    if request.method == 'POST':
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cam.isOpened():
            flash("Error: Unable to access the camera.", "danger")
            return redirect(url_for('home'))

        sampleNum = 0
        path_to_store = os.path.join(os.getcwd(), "all_images", session['aadhar'])

        # Create directory if not exists
        os.makedirs(path_to_store, exist_ok=True)

        faces_data = []  # List to store face data
        try:
            while sampleNum < 50:  # Capture 50 images per person
                ret, img = cam.read()

                if not ret or img is None:
                    flash("Camera error: Unable to capture image.", "danger")
                    cam.release()
                    return redirect(url_for('home'))

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face_img = gray[y:y + h, x:x + w]
                    face_resized = cv2.resize(face_img, (50, 50))  # Resize for consistency

                    # Save face to disk
                    cv2.imwrite(os.path.join(path_to_store, f"{sampleNum + 1}.jpg"), face_resized)

                    # Store face data (flattened) in the list
                    faces_data.append(face_resized.flatten())

                    sampleNum += 1

            # Save face data to file
            save_face_data(session['aadhar'], faces_data)

        finally:
            cam.release()
            cv2.destroyAllWindows()

        flash("Face registered successfully!", "success")
        return redirect(url_for('home'))

    return render_template('capture.html')


# Helper function to save face data
def save_face_data(aadhaar, face_list):
    file_path = "faces_data.pkl"
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        else:
            data = {}

        data[aadhaar] = face_list  # Store face data with Aadhaar

        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    except Exception as e:
        flash(f"Error saving face data: {str(e)}", "danger")



import pickle

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def getImagesAndLabels(path):
    folderPaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    global le
    for folder in folderPaths:
        imagePaths = [os.path.join(folder, f) for f in os.listdir(folder)]
        aadhar_id = folder.split("\\")[1]
        for imagePath in imagePaths:
            # loading the image and converting it to gray scale
            pilImage = Image.open(imagePath).convert('L')
            # Now we are converting the PIL image into numpy array
            imageNp = np.array(pilImage, 'uint8')
            # extract the face from the training image sample
            faces.append(imageNp)
            Ids.append(aadhar_id)
            # Ids.append(int(aadhar_id))
    Ids_new=le.fit_transform(Ids).tolist()
    output = open('encoder.pkl', 'wb')
    pickle.dump(le, output)
    output.close()
    return faces, Ids_new

@app.route('/update')
def update():
    return render_template('update.html')
@app.route('/updateback', methods=['POST','GET'])
def updateback():
    if request.method=='POST':
        Fullname = request.form['Fullname']
        aadhar_id = request.form['aadhar_id']
        College_id = request.form['College_id']
        email = request.form.get('email')  
        voters=pd.read_sql_query('SELECT * FROM voters', mydb)
        all_aadhar_ids=voters.aadhar_id.values
        if age >= 18:
            if (aadhar_id in all_aadhar_ids):
                sql = "INSERT INTO voters (Fullname, aadhar_id, College_id, email, verified) VALUES (%s, %s, %s, %s, %s)"
                cur = mydb.cursor()
                cur.execute(sql, (Fullname, aadhar_id, College_id, email, 'no'))
                mydb.commit()
                cur.close()
                session['aadhar']=aadhar_id
                session['status']='no'
                session['email']=email
                flash(r'Database Updated Successfully','Primary')
                return redirect(url_for('verify'))
            else:
                flash(f"Aadhar: {aadhar_id} doesn't exists in the database for updation", 'warning')
    return render_template('update.html')

def find_best_match(new_face):
    """Compares the new face with stored faces and returns the best match"""
    best_match = None
    best_score = 0  # Higher = better match

    all_images_path = "all_images/"
    for aadhar_id in os.listdir(all_images_path):  # Loop through all Aadhaar folders
        person_path = os.path.join(all_images_path, aadhar_id)

        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                stored_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                stored_img = cv2.resize(stored_img, (50, 50)).flatten()

                similarity = cosine_similarity([new_face], [stored_img])[0][0]  # Cosine similarity
                if similarity > best_score:
                    best_score = similarity
                    best_match = aadhar_id

    return best_match, best_score


import os

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@app.route('/voting', methods=['POST', 'GET'])
def voting():
    if request.method == 'POST':
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cam.isOpened():
            flash("Error: Unable to access the camera.", "danger")
            return redirect(url_for('home'))

        facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        detected_aadhar = None
        max_confidence = -1  # Keep track of the best match

        try:
            while True:
                ret, img = cam.read()
                if not ret:
                    flash("Camera error: Unable to capture frame.", "danger")
                    break

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    cropped_face = gray[y:y + h, x:x + w]
                    resized_face = cv2.resize(cropped_face, (50, 50)).flatten()

                    # Compare with stored faces
                    best_match_aadhar, confidence = find_best_match(resized_face)

                    if confidence > max_confidence:  # Update the best match
                        max_confidence = confidence
                        detected_aadhar = best_match_aadhar

                if detected_aadhar:
                    session['select_aadhar'] = detected_aadhar
                    flash(f"Face matched with Aadhaar: {detected_aadhar}", "success")
                    break

                cv2.imshow('Face Recognition', img)
                cv2.setWindowProperty('Face Recognition', cv2.WND_PROP_TOPMOST, 1)
                if cv2.waitKey(1) == ord('q'):
                    break

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

        finally:
            cam.release()  
            cv2.destroyAllWindows()

        if detected_aadhar:
            return redirect(url_for('select_candidate'))
        else:
            flash("No valid voter detected!", "danger")
            return redirect(url_for('home'))

    return render_template('voting.html')

import cv2
import mediapipe as mp
import numpy as np


def detect_gesture():
    """Detects hand gestures and maps them to candidates based on the number of raised fingers."""
    cap = cv2.VideoCapture(0)
    hands = mp.solutions.hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    candidate_map = {1: "Party_1", 2: "Party_2", 3: "Party_3", 4: "Party_4", 5: "Party_5"}
    vote = None
    
    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Identify raised fingers
                fingers_up = []
                finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
                finger_bottoms = [3, 6, 10, 14, 18]  # Bottom of fingers

                for tip, bottom in zip(finger_tips, finger_bottoms):
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[bottom].y:
                        fingers_up.append(1)
                    else:
                        fingers_up.append(0)

                num_raised = sum(fingers_up)  # Count the raised fingers
                
                if num_raised in candidate_map:
                    vote = candidate_map[num_raised]
                    break

        cv2.imshow("Gesture Voting", frame)

        if vote:
            print(f"Vote registered for {vote}")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return vote

@app.route('/select_candidate', methods=['POST', 'GET'])
def select_candidate():
    aadhar = session.get('select_aadhar')
    if not aadhar:
        flash("No Aadhaar detected. Please verify your face first.", "danger")
        return redirect(url_for('home'))

    df_nom = pd.read_sql_query('SELECT * FROM nominee', mydb)
    all_nom = df_nom['symbol_name'].values.tolist()

    votes_db = pd.read_sql_query("SELECT aadhar FROM vote", mydb)
    voted_aadhaars = votes_db['aadhar'].values.tolist()

    if aadhar in voted_aadhaars:
        flash("You have already voted.", "warning")
        return redirect(url_for('home'))

    vote = detect_gesture()  # Capture vote via gesture
    print("Detected Vote:", vote)  # Debugging print

    if not vote:
        flash("No gesture detected!", "danger")
        return redirect(url_for('select_candidate'))

    try:
        sql = "INSERT INTO vote (vote, aadhar) VALUES (%s, %s)"
        cur = mydb.cursor()
        cur.execute(sql, (vote, aadhar))
        mydb.commit()
        cur.close()
        flash(f"Voted Successfully via Gesture for {vote}!", 'success')

        # Debugging: Check if vote is stored
        result = pd.read_sql_query("SELECT * FROM vote", mydb)
        print(result)  # Print votes stored in database
    except Exception as e:
        flash(f"Error storing vote: {str(e)}", "danger")

    return redirect(url_for('home'))




@app.route('/voting_res')
def voting_res():
    try:
        votes = pd.read_sql_query('SELECT * FROM vote', mydb)
        if votes.empty:
            flash("No votes recorded yet!", "info")
            return render_template('voting_res.html', freq=[], noms=[])

        # Debugging: Check retrieved votes
        print("Votes Data:", votes)

        counts = votes['vote'].value_counts().reset_index()
        counts.columns = ['symbol_name', 'count']

        all_imgs = ['1.png', '2.png', '3.jpg', '4.png', '5.png', '6.png']

        # Ensure index error is avoided
        all_freqs = [counts[counts['symbol_name'] == i]['count'].values[0] if i in counts['symbol_name'].values else 0 for i in all_imgs]

        df_nom = pd.read_sql_query('SELECT * FROM nominee', mydb)
        all_nom = df_nom['symbol_name'].values

        return render_template('voting_res.html', freq=all_freqs, noms=all_nom)
    except Exception as e:
        flash(f"Error retrieving vote results: {str(e)}", "danger")
        return redirect(url_for('home'))



if __name__=='__main__':
    app.run(debug=True)


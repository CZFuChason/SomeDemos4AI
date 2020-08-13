import cv2
import numpy as np
import sys

Face_casc = '../../haarcascade_files/haarcascade_frontalface_default.xml'
Eye_casc = '../../haarcascade_files/haarcascade_eye.xml'
face_cascade_classifier = cv2.CascadeClassifier(Face_casc)
eye_cascade_classifier = cv2.CascadeClassifier(Eye_casc)

video_capture = cv2.VideoCapture(0)

while True:


    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade_classifier.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade_classifier.detectMultiScale(
            roi_gray,
            scaleFactor=1.3,
            minNeighbors=1,
            minSize=(2, 2),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
        roi_color = frame[y:y+h, x:x+w]
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),5)

    cv2.imshow('Face', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

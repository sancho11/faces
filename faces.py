#!/usr/bin/env python3
import sys
import cv2
import sys

vid_cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_id = 1
count = 0
while(vid_cam.isOpened()):
    ret, image_frame = vid_cam.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        font= cv2.FONT_HERSHEY_SIMPLEX
        name="Persona"
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(image_frame,name, (x,y), font,1,(255,0,0), 2, cv2.LINE_AA)
        count += 1
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('frame', image_frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif count>100:
        break
vid_cam.release()
cv2.destroyAllWindows()

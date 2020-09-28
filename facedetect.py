import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/home/moody/installation/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('/home/moody/installation/opencv/data/haarcascades/haarcascade_eye.xml')
#omega_cascade = cv2.CascadeClassifier('/home/moody/WORK/openCV/TRAIN/data/cascade.xml')

cap = cv2.VideoCapture(0)
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (100,100)
fontScale              = 1
fontColor              = (0,255,255)
lineType               = 2

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #omegas = omega_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # cv2.putText(img,'DETECTED', (x,y-20), font, fontScale,fontColor,lineType)
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in eyes:
        #   cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # for (x,y,w,h) in omegas:
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    #     cv2.putText(img,'Omega DETECTED', (x,y-20), font, fontScale,fontColor,lineType)
    #     print(".")
    #roi_gray = gray[y:y+h, x:x+w]
    #roi_color = img[y:y+h, x:x+w]

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
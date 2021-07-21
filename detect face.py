import cv2
import numpy as np
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

#image = cv2.imread('./img.jpeg')

def detect_face(grayImage, image):
    faces = face_cascade.detectMultiScale(grayImage, 1.03, 5)
        # (이미지에서 얼굴크기가 서로 다른 것을 보상해주는 값, 얼굴 사이의 최소 간격, 얼굴의 최소 크기)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 20) #이미지, 시작점, 정료점, 색상, 선두께
    return image
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.flip(img,1)#좌우반전, 0을 넣는다면 상하반전
    if(success):
        canvas = detect_face(grayImage, img)
        cv2.imshow("video", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
'''
plt.figure(figsize=(12, 12))
plt.imshow(image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
'''
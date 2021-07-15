'''
import cv2
cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 1080)

while True: 
    ret, frame = cap.read() 
    cv2.imshow('test', frame) 
    k = cv2.waitKey(1) 
    if k == 27: 
        break 
cap.release()
cv2.destroyAllWindows()
'''

import cv2

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("resources/test_video.mp4")

cap.set(3,640)
cap.set(4,480)

print('width :%d, height : %d' % (cap.get(3), cap.get(4)))

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)#좌우반전, 0을 넣는다면 상하반전
    if(success):
        cv2.imshow("video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

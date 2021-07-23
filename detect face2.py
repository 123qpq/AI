import cv2, dlib
import numpy as np
from matplotlib import pyplot as plt


scaler = 0.3
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    if(success):
        #img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
        ori = img.copy()

        faces = detector(img)
        if len(faces) == 0:
            continue
        face = faces[0]

        cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 5) #이미지, 시작점, 정료점, 색상, 선두께
        dlib_shape = predictor(img, face)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        for s in shape_2d:
            cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness = 2)

        cv2.imshow("video", img)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
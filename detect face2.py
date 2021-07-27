from numpy.core.fromnumeric import shape
import cv2, dlib
import numpy as np
from matplotlib import pyplot as plt


eyes = list(range(36, 48))
reye = list(range(36, 42))
reye = list(range(42, 48))
nose = list(range(27, 36))
mouth = list(range(48, 68))


scaler = 0.3
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    if(success):
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
        ori = img.copy()

        faces = detector(img)
        if len(faces) == 0:
            continue
        face = faces[0]

        cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 5) #이미지, 시작점, 정료점, 색상, 선두께
        dlib_shape = predictor(img, face)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)
        '''
        showpart = shape_2d[0][36:48]
        print(shape_2d)
        for s in showpart:
            cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness = 2)
        '''
        for s in shape_2d:
            cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness = 2)
        cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color=(255, 0, 0), thickness=10, lineType=cv2.LINE_AA)
        cv2.imshow("video", img)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
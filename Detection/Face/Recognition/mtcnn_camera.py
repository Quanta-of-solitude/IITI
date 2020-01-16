from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from model import getFace

detector = MTCNN()
rgb = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX




def start_app(cnn):

    ix=0
    while True:
        ix +=1
        _,fr = rgb.read()
        faces = detector.detect_faces(fr)
        for face in faces:

            roi = cv2.resize(fr,(40,40))
            x, y, width, height = face['box']
            x2, y2 = x + width, y + height
            pred = cnn.whoisface(roi)
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            print(roi.shape)
            cv2.rectangle(fr,(x,y),(x2,y2),(255,0,0),2)

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Filter', fr)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model = getFace("recognizer_skb.h5")
    start_app(model)

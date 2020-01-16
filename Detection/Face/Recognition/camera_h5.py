import cv2
from model import getFace
import numpy as np

rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
sidec = cv2.CascadeClassifier('haarcascade_profileface.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

def __get_data__():
    """
    __get_data__: Gets data from the VideoCapture object and classifies them
    to a face or no face.

    returns: tuple (faces in image, frame read, grayscale frame)
    """
    _, fr = rgb.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)
    sides = sidec.detectMultiScale(gray,1.3,5)
    return sides,faces, fr,gray

def start_app(cnn):
    skip_frame = 10
    data = []
    flag = False
    ix = 0
    while True:
        ix += 1

        sides,faces, fr,gray_fr = __get_data__()
        
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (40, 40))
            pred = cnn.whoisface(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        for (x, y, w, h) in sides:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (40, 40))
            pred = cnn.whoisface(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Filter', fr)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = getFace("recognizer_skb.h5")
    start_app(model)

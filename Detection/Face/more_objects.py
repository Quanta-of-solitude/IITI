import cv2
#from model import FacialExpressionModel
import numpy as np

rgb = cv2.VideoCapture(0)
#handc = cv2.CascadeClassifier("hand.xml")
handc = cv2.CascadeClassifier("models/haarcascade_frontalface_alt.xml") #front face
facec = cv2.CascadeClassifier("models/haarcascade_profileface.xml") #side face
font = cv2.FONT_HERSHEY_SIMPLEX

def __get_data__():
    """
    __get_data__: Gets data from the VideoCapture object and classifies them
    to a face or no face.

    returns: tuple (faces in image, frame read, grayscale frame)
    """
    _, fr = rgb.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    fist = handc.detectMultiScale(gray, 1.3, 5)
    face = facec.detectMultiScale(gray,1.3, 5)


    return face,fist, fr, gray

def start_app():
    skip_frame = 10
    data = []
    flag = False
    ix = 0
    while True:
        ix += 1

        face,fist, fr, gray_fr = __get_data__()
        for (x, y, w, h) in fist:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            #pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, "Face Front Found", (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
            #print("Detection")

        for (x, y, w, h) in face:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            #pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, "Side Face", (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Filter', fr)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #model = FacialExpressionModel("face_model.json", "face_model.h5")
    #start_app(model)
    start_app()

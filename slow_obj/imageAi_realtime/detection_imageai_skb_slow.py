'''
IITI 2020 skb
'''

from imageai.Detection import VideoObjectDetection, ObjectDetection
import os
import cv2
import numpy as np

execution_path = os.getcwd()

camera = cv2.VideoCapture(0)

#detector = VideoObjectDetection()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
#detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
detector.loadModel()


#need the yolo h5

ix = 0
#while True:
#range is faster than whiles
for i in range(0,100000000000):
    ix +=1
    _,fr = camera.read()
    #gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    #gray = cv2.resize(gray,(40,40))

    cv2.imwrite("raw_image.jpg",fr)

    #video_path = detector.detectObjectsFromVideo(camera_input=fr)
    detections = detector.detectObjectsFromImage(input_image="raw_image.jpg", output_image_path="imagenew.jpg", minimum_percentage_probability=70)
    if cv2.waitKey(1) == 27:
        break
    image = cv2.imread("imagenew.jpg")
    cv2.imshow("Video",image)

    #cv2.destroyAllWindows()
cv2.destroyAllWindows()

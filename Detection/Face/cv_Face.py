import cv2

#load pretrained model
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
pixels = cv2.imread('faces_all.png')
#perform faceDetection
bboxes = classifier.detectMultiScale(pixels)


for box in bboxes:
    print(box) #print bounding coordinates
    #results like [123 11 190 190]
    x, y, width, height = box
    x2, y2 = x + width, y + height #other corner

    #draw the rectangle
    cv2.rectangle(pixels, (x,y),(x2,y2),(0,0,255),1)

#show image now
cv2.imshow('Faces', pixels)
cv2.waitKey(0)
# close the window
cv2.destroyAllWindows()

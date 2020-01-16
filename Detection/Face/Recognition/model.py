from keras.models import load_model
from keras.preprocessing import image
import numpy as np


class getFace(object):

    def __init__(self, model_file):

        self.model = load_model(model_file)

    def whoisface(self,img):

        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        list_faces = ["Pankhi", "Sanjeev"]
        self.preds = self.model.predict_classes(img)
        print(self.preds[0])
        label = self.preds[0][0]
        return list_faces[label]


if __name__ == '__main__':
    pass

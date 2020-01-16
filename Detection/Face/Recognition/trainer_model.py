import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#initializing the model
classifier = Sequential()

#conv 1
classifier.add(Conv2D(32,(3,3),input_shape=(40,40,3),activation="relu"))
classifier.add(MaxPooling2D(2,2))

#conv2
#classifier.add(Conv2D(64,(3,3),activation="relu"))
#classifier.add(MaxPooling2D(2,2))

classifier.add(Flatten())

#full connection1
classifier.add(Dense(output_dim=128,activation="relu"))

#full connecction 2
#classifier.add(Dense(output_dim=200,activation="relu"))


#output layer
classifier.add(Dense(output_dim=1,activation="sigmoid"))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])



train_datagen = ImageDataGenerator(
        rescale=1./255,

        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'data/train',
        target_size=(40, 40),
        #color_mode = "grayscale",
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'data/test',
        target_size=(40, 40),
        #color_mode = "grayscale",
        batch_size=32,
        class_mode='binary')

#fitting the model
classifier.fit_generator(
        training_set,
        steps_per_epoch=20,
        epochs=10,
        validation_data=test_set)

#classifier.summary()

model_json = classifier.to_json()
with open("recognizer_skb.json", "w") as json_file:
    json_file.write(model_json)

classifier.save("recognizer_skb.h5")

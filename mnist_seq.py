import tensorflow as tf
import numpy as np
from scipy.misc import imresize
from PIL import Image
import os

def read_dataset():

    if(os.path.exists("mnist_images_seq.npy") & os.path.exists("mnist_grounds_seq.npy")):
        data_images=np.load("mnist_images_seq.npy")
        data_grounds=np.load("mnist_grounds_seq.npy")
        return data_images, data_grounds
    else:
        #Number of images in dataset
        nImgs=60000

        #Open dataset as a byte file
        f=open("./dataset/train-images-idx3-ubyte", "rb")
        f_ground=open("./dataset/train-labels-idx1-ubyte", "rb")

        #Skip unimportant bytes
        f.read(16)
        f_ground.read(8)
        print(f_ground)
        #Create numpy array to store dataset
        data_images=np.zeros((nImgs,784), dtype=float)
        data_grounds=np.zeros(nImgs)

        #Loop through all images
        for x in range(nImgs):
            #Save the values of columns*rows pixels at a time
            for n in range(784):
                byte=f.read(1)
                pixel_value=int.from_bytes(byte,"big")
                data_images[x][n]=pixel_value
            byte=f_ground.read(1)
            ground_value=int.from_bytes(byte,"big")
            data_grounds[x]=ground_value
        vis(data_images, data_grounds)
        f_ground.close()
        f.close()
        np.save("mnist_images_seq.npy", data_images)
        np.save("mnist_grounds_seq.npy", data_grounds)
        return(data_images, data_grounds)

def vis(image, ground):
        index=9
        image=image[index].reshape(28,28)
        print(image)
        img = Image.fromarray(image)
        img.show()
        print(ground[index])

def train(data, grounds):

     model = tf.keras.models.Sequential()
     model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
     model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
     model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
     model.compile(optimizer="adam",
                     loss="sparse_categorical_crossentropy",
                     metrics=["accuracy"])
     model.fit(data, grounds, epochs=3)
     model.save("seq.h5")

data, grounds = read_dataset()
train(data, grounds)
import tensorflow as tf
import numpy as np
from scipy.misc import imresize
from PIL import Image
import os

def read_dataset():

    if(os.path.exists("mnist_images_conv.npy") & os.path.exists("mnist_grounds_conv.npy")):
        data_images=np.load("mnist_images_conv.npy")
        data_grounds=np.load("mnist_grounds_conv.npy")
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
        data_images=np.zeros((nImgs,28,28,1), dtype=float)
        data_grounds=np.zeros(nImgs)
        img_1d=np.zeros(784)

        #Loop through all images
        for x in range(nImgs):
            #Save the values of columns*rows pixels at a time
            for n in range(784):
                byte=f.read(1)
                pixel_value=int.from_bytes(byte,"big")
                img_1d[n]=pixel_value
            data_images[x]=x
            data_images[x,:,:,0]=img_1d.reshape(28,28)
            byte=f_ground.read(1)
            ground_value=int.from_bytes(byte,"big")
            data_grounds[x]=ground_value
        vis(data_images, data_grounds)
        f_ground.close()
        f.close()
        np.save("mnist_images_conv.npy", data_images)
        np.save("mnist_grounds_conv.npy", data_grounds)
        return(data_images, data_grounds)

def vis(image, ground):
        index=453
        print(image)
        img = Image.fromarray(image[index,:,:,0])
        img.show()
        print(ground[index])

def train(data, grounds):
    data /= 255
#    grounds = tf.keras.utils.to_categorical(grounds)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (5, 5), input_shape=(data.shape[1], data.shape[2], 1), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.fit(data, grounds, epochs=7, batch_size=10)
    model.save("model_conv.h5")


def vis_image(data, index):
    img = Image.fromarray(data[0,:,:,0])
    img.show()

def predict(data, grounds, index):
    data=data[index,:,:,0]
    data=data.reshape((1,28,28,1))
    vis_image(data, index)
    model = tf.keras.models.load_model("model_conv.h5")
    prediction = model.predict(data)
    print(grounds[index])
    print("prediccion ---->", np.argmax(prediction))


data, grounds = read_dataset()
#predict(data, grounds, 562)
train(data,grounds)
from flask import Flask, render_template, request
import keras
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D,Activation, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from skimage.io import imshow
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
#from matplotlib import pyplot as plt
from imutils import face_utils
import numpy as np
# import argparse
import imutils
import dlib
import cv2
# from google.colab.patches import cv2_imshow
from keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

def initialize_weights(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
                   kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the uploaded images
    image1 = request.files["image1"]
    image2 = request.files["image2"]

    # Perform facial recognition prediction....................................................
    model = get_siamese_model((80, 80, 3))
    optimizer = Adam(learning_rate = 0.00006)
    model.compile(loss="binary_crossentropy",optimizer=optimizer)
    # model.load_weights('/content/drive/My Drive/Colab Notebooks/capstone project/Nose_model/siamese_network.h5')
    model.load_weights('siamese_network.h5')
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # predictor = dlib.shape_predictor(args["shape_predictor"])
    
    #First image processing.....
    height, width, channels = image1.shape
    if height < 500 and width < 500:
        return render_template("index.html", prediction='Size of image 1 should be below 500,500')
        
    # load the input image, resize it, and convert it to grayscale
    # image = imutils.resize(image, width=500)
    height, width, channels = image1.shape
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
      # determine the facial landmarks for the face region, then
      # convert the facial landmark (x, y)-coordinates to a NumPy
      # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

    x_min,y_min = np.amin(shape[27:36], axis = 0)
    x_max,y_max = np.amax(shape[27:36], axis = 0)

    offsetX = (y_max-y_min - (x_max-x_min))/2
    offsetX = int(offsetX)

    img = cv2.cvtColor(image1[y_min:y_max, x_min-offsetX:x_max+offsetX], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = crop_max_square(img)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = imutils.resize(img, width=80, height=80)
    # cv2_imshow(img)

    img1 = img
    
    #Second image processing.....
    height, width, channels = image2.shape
    if height < 500 and width < 500:
        return render_template("index.html", prediction='Size of image2 should be below 500,500')
    # load the input image, resize it, and convert it to grayscale
    # image = imutils.resize(image, width=500)
    height, width, channels = image2.shape
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
      # determine the facial landmarks for the face region, then
      # convert the facial landmark (x, y)-coordinates to a NumPy
      # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

    x_min,y_min = np.amin(shape[27:36], axis = 0)
    x_max,y_max = np.amax(shape[27:36], axis = 0)

    offsetX = (y_max-y_min - (x_max-x_min))/2
    offsetX = int(offsetX)

    img = cv2.cvtColor(image2[y_min:y_max, x_min-offsetX:x_max+offsetX], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = crop_max_square(img)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = imutils.resize(img, width=80, height=80)
    # cv2_imshow(img)

    img2 = img
    
    #Final Prediction.....
    img1 = np.expand_dims(img1, axis=0)
    img1.shape
    img2 = np.expand_dims(img2, axis=0)
    img2.shape
    predTest = (model.predict([img1, img2]) > 0.5).astype("int32")
    if predTest==0:
        return render_template("index.html", prediction='Images are not of same person')
    if predTest==1:
        return render_template("index.html", prediction='Images are of same person')
    
    

if __name__ == "__main__":
    app.run(debug=True)

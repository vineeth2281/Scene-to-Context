#Explation 

#Small python code for deployment 

#It gets image as input from the user 

#preprocess the image (preprocess_image) 

#converts it into feature vector by using function (encode_image) 
#------------------------------------------------------------------------------------------#


#importing required library 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import re 
import nltk 
from nltk.corpus import stopwords 
import string 
import json 
from time import time 
import pickle 
import tensorflow as tf
import keras 
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model , load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Input,Dense,Dropout,Embedding,LSTM
from keras.layers.merge import add 
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions


#Loading the saved model...

model = load_model("./model_9.h5")
model_temp = ResNet50(weights="imagenet", input_shape=(224,224,3))
model_resnet = Model(model_temp.input, model_temp.layers[-2].output)


# preprocessing the the given image...

def preprocess_image(img):
    img = image.load_img(img, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  )
    return img



# Converting the preprocessed image into feature vector... 

def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1])
    return feature_vector



with open("./word_to_idx.pkl", 'rb') as w2i:
    word_to_idx = pickle.load(w2i)
    
with open("./idx_to_word.pkl", 'rb') as i2w:
    idx_to_word = pickle.load(i2w)

idx_to_word


#prediction caption using feature vector and finally returning predicted caption

def predict_caption(photo):
    
    in_text = "startseq"
    max_len=35 
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() #WOrd with max prob always - Greedy Sampling
        word = idx_to_word[ypred]
        in_text += (' ' + word)
        
        if word == "endseq":
            break
    
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption


# Final prediction function .... 

def caption_this_image(image): 
    enc = encode_image(image)
    caption = predict_caption(enc)
    return caption 
    







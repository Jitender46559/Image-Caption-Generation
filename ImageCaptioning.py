import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, \
    Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical



train_features = load(open("C:/Users/Jitender kumar/Desktop/Final Year Project/encoded_train_images.pkl", "rb"))

test_features = load(open("C:/Users/Jitender kumar/Desktop/Final Year Project/encoded_test_images.pkl", "rb"))

train_descriptions = load(open("C:/Users/Jitender kumar/Desktop/Final Year Project/train_descriptions.pkl", "rb"))

all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
print(len(all_train_captions))

word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1



max_length = 34


embedding_dim = 300
vocab_size = len(ixtoword)



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_last.h5")

with open("C:/Users/Jitender kumar/Desktop/Final Year Project/encoded_test_images.pkl", "rb") as encoded_pickle:
    encoding_test = load(encoded_pickle)

#images = 'C:/Users/Jitender kumar/Desktop/Final Year Project/Flicker8k_Dataset/'

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


modelImage = InceptionV3(weights='imagenet')
model_new = Model(modelImage.input, modelImage.layers[-2].output)


def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec



def generateCaption(image):
    imgtemp = encode(image)
    image2 = imgtemp.reshape((1, 2048))
    cap = greedySearch(image2)
    return cap
'''print("Greedy:", greedySearch(image2))
x = plt.imread(image)
plt.imshow(x)
plt.show() '''

if __name__=='__main__':
    pic_loc = 'C:\\Users\\Jitender kumar\\Desktop\\Final Year Project\\test2.jpg'
    print(generateCaption(pic_loc))
'''
z=210
pic = list(encoding_test.keys())[z]
image = encoding_test[pic].reshape((1,2048))
x=plt.imread(images+pic)
plt.imshow(x)
plt.show()
print("Greedy:",greedySearch(image))

'''

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 19:38:25 2019

@author: Rhiannon
"""

#%% - Setting working directory
#Set wd
import os
os.chdir("C:\\Users\\finba\\Documents\\GitHub\\IC-Hack19")

#%% - Loading in libraries
#Import libraries needed for analysis
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2grey
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, accuracy_score
from numpy import array

#%% - Loading in labels to reference picture dataset
#Load dataframe with labels and image names
labels = pd.read_csv("Data/labels1.csv", index_col=0)

#Show first five rows of dataframe
display(labels.head(5))

#%% - Selecting specific dog and displaying image
#Subset data (e.g. Bee species where a genus value of "1.0" = Bombus bees)
#And take 6th item of index (index starts at 0)
Dingo_1st = labels[labels.breed == "dingo"].index[0]

#Create function "get_image" required for analysis
def get_image(row_id, root="Data/train/train1/"):
    """
    Converts an image number into the file path where the image is located, 
    opens the image, and returns the image as a numpy array.
    """
    filename = "{}.jpg".format(row_id)
    file_path = os.path.join(root, filename)
    img = Image.open(file_path)
    return np.array(img)
def get_image_file(row_id, root="Data/train/train1/"):
    """
    Converts an image number into the file path where the image is located, 
    opens the image, and returns the image as a numpy array.
    """
    filename = "{}.jpg".format(row_id)
    file_path = os.path.join(root, filename)
    img = Image.open(file_path)
    return img

#Use "get_image" function to get image/create filepath to image and use
#"plt.imshow" to display image

plt.imshow(get_image(Dingo_1st))
plt.show()

#%% - Displaying dog images in greyscale
#Load in dog and show it's dimensions, including 3 channels (RGB)
Dingo = get_image(Dingo_1st)
print ('Dog shape is:', Dingo.shape)

#Convert to grey, reducing to 1 channel and display new dimensions
grey_Dingo = rgb2grey(Dingo)
print ('Greyscale Dog shape is:', grey_Dingo.shape)
plt.imshow(grey_Dingo)

#%% - Generating data/vector about shape of image
#Carry out HOG analysis on greyscale image (provides 2 outputs - features and new image)
hog_features, hog_image = hog(grey_Dingo, visualize=True, block_norm='L2-Hys', pixels_per_cell=(16,16))

#Display HOG analysis (should show shape)
plt.imshow(hog_image, cmap=mpl.cm.gray)

#%% - Generate array with pixel intensity and shape/HOG data
#Create function that will calculate intensity values of combined channels and HOG analysis
#Flatten RGB into single column matrix
#Convert image to grey
#HOG analysis
#Combine HOG and pixel features into one array
def create_features(img):
    colour_features = img.flatten()
    grey_image = rgb2grey(img)
    hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    flat_features = np.hstack((colour_features, hog_features))
    return flat_features

Dingo_Features = create_features(Dingo)

print(Dingo_Features.shape)

#%% - Generate single matrix of features for a number of images, rows = image, columns = features
#Create function to generate matrix
def create_feature_matrix(label_dataframe):
    features_list = []
    
    for img_id in label_dataframe.index:
        # load image
        img = get_image(img_id)
        #Pad image to 500*500
        desired_size = 500
        im = get_image_file(img_id)
        old_size = im.size  # old_size[0] is in (width, height) format
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        delta_w = desired_size - new_size[0]
        delta_h = desired_size - new_size[1]
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        new_img = ImageOps.expand(im, padding)
        
        # get features for image
        image_features = create_features(new_img)
        features_list.append(image_features)
        print(len(image_features))
    # convert list of arrays into a matrix
    feature_matrix = np.array(features_list)
    return feature_matrix

# run create_feature_matrix on our dataframe of images
feature_matrix = create_feature_matrix(labels)
print(feature_matrix.shape)
#%% - Pad images to make same size to allow concatination of list
desired_size = 500
im_pth = "Data/train/train1/000bec180eb18c7604dcecc8fe0dba07.jpg"
im = Image.open(im_pth)
old_size = im.size  # old_size[0] is in (width, height) format
ratio = float(desired_size)/max(old_size)
new_size = tuple([int(x*ratio) for x in old_size])

delta_w = desired_size - new_size[0]
delta_h = desired_size - new_size[1]
padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
new_im = ImageOps.expand(im, padding)
new_im.show()
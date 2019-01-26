# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 19:42:54 2019

@author: Rhiannon
"""

#%% - Loading in libraries
#Import libraries needed for analysis
import os
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
labels = pd.read_csv("labels.csv", index_col=0)

#Show first five rows of dataframe
display(labels.head(5))

#%%get the image path

def get_image_file(row_id, root="train/"):
    """
    Converts an image number into the file path where the image is located, 
    opens the image, and returns the image as a numpy array.
    """
    filename = "{}.jpg".format(row_id)
    file_path = os.path.join(root, filename)
    #img = Image.open(file_path)
    return file_path
#%% change the image size 

"""
RUN ONCE!!
twice wont do anything extra but it will take ages
"""

def change_size(label_dataframe):
    for img_id in label_dataframe.index:
        #produce image path
        im_pth = get_image_file(img_id)
        #read image in 
        im = Image.open(im_pth)
        
        #make all into squares
        desired_size = max(im.size)
        old_size = im.size
        
        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size-old_size[0])//2, (desired_size-old_size[1])//2))
        #new_im.show
        
        #resize them
        new_img = new_im.resize((500,500))
        new_img.save(im_pth, "JPEG", optimize = True)
        
change_size(labels)
        
#%% Make features for one

def create_features(img):
    colour_features = img.flatten()
    grey_image = rgb2grey(img)
    hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    flat_features = np.hstack((colour_features, hog_features))
    return flat_features

#%% create feature matrix
    
def create_feature_matrix(label_dataframe):
    features_list = []
    n = 0
    for img_id in label_dataframe.index:
        # load image
        img = np.array(
                Image.open(
                        get_image_file(img_id)
                        )
                )
        # get features for image
        image_features = create_features(img)
        features_list.append(image_features)
        #just to check stuff is the right len: 
        #print(len(image_features))
        #only run the print for first time and stop it
        n = n + 1
        if n % 1000 == 0:
            print(n)
    # convert list of arrays into a matrix
    feature_matrix = np.array(features_list)
    return feature_matrix

#%% Run feature matrix 
feature_matrix = create_feature_matrix(labels)
print(feature_matrix.shape)
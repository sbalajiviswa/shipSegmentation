# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 18:18:27 2023

@author: balaji
"""

import streamlit as st
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import skimage
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import img_to_array
from keras.models import Model
import matplotlib.pyplot as plt

def threshold_segmentation(image):
  gray_image = skimage.color.rgb2gray(image)
  blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)
  t = skimage.filters.threshold_otsu(blurred_image)
  binary_mask = blurred_image > t
  return binary_mask

vgg_model = VGG16()
convolution_blocks = [2, 5, 9, 13, 17]
outputs = [vgg_model.layers[i].output for i in convolution_blocks]
vgg_model = Model(inputs=vgg_model.inputs, outputs=outputs)

def VGG_Predict(image):
  partimages = []
  for i in range(0,768,256):
    for j in range(0,768,256):
      # concatImages = image[j:j+256,i:i+256]
      
      vgg_image = cv2.resize(image[j:j+256,i:i+256],(224,224))
      vgg_image = img_to_array(vgg_image)
      vgg_image = np.expand_dims(vgg_image, axis=0)
      vgg_image = preprocess_input(vgg_image)
      feature_map = vgg_model.predict(vgg_image,verbose = 0)[0][0,:,:]

      feature_map = cv2.resize(feature_map[:,:,0],(256,256))
      partimages.append(feature_map)
  mask = cv2.hconcat([cv2.vconcat([partimages[j] for j in range(i,i+3)]) for i in range(0,len(partimages),3)])
  return mask


st.title("ship segmentation")
st.write("Upload a sattelite image of a ship to perform segmentation")

fullres_model = load_model('F:/Final_Year_Project/WrkDrctry/fullres_model.h5')
imgFromweb = st.file_uploader("Chose a file")
if imgFromweb:
    # nparr = np.fromstring(imgFromweb, np.uint8)
    # image = np.array(cv2.imdecode(nparr,cv2.IMREAD_COLOR))
    # imageEXP = np.expand_dims(image, 0)/255.0
    # pred = fullres_model.predict(imageEXP)[0]
    image = Image.open(imgFromweb)
    image = np.array(image)
    image = cv2.resize(image,(768,768))
    st.image(image, caption='Origianl Image')
    
    imageEXP = np.expand_dims(image, 0)/255.0
    NormalunetPredict = fullres_model.predict(imageEXP)[0]
    NormaltsPredict = threshold_segmentation(image)
    NormalVGGPredict = VGG_Predict(image)
    fig,ax = plt.subplots(1,3,figsize = (14,8))
    ax[0].imshow(NormaltsPredict)
    ax[0].set_title("Using Threshold segmentation")
    ax[1].imshow(NormalVGGPredict)
    ax[1].set_title("Using VGG segmentation")
    ax[2].imshow(NormalunetPredict)
    ax[2].set_title("Using unet segmentation")
    st.pyplot(fig)
    # st.image(NormalVGGPredict , caption = 'segmentaion using VGG16',clamp=True)
    # st.image(NormalunetPredict , caption = 'segmentaion using unet',clamp=True)
    # imgNumpy = np.array(img)
    
    # st.write(str(imgNumpy.shape))
    
    # st.image(image)
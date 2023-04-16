# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:20:07 2023

@author: balaji
"""
print("Hello world")

def rle2bbox(rle, shape):
    
    a = np.fromiter(rle.split(), dtype=np.uint)
    a = a.reshape((-1, 2))
    a[:,0] -= 1
    
    y0 = a[:,0] % shape[0]
    y1 = y0 + a[:,1]
    if np.any(y1 > shape[0]):
        y0 = 0
        y1 = shape[0]
    else:
        y0 = np.min(y0)
        y1 = np.max(y1)
    
    x0 = a[:,0] // shape[0]
    x1 = (a[:,0] + a[:,1]) // shape[0]
    x0 = np.min(x0)
    x1 = np.max(x1)
    
    if x1 > shape[1]:
        raise ValueError("invalid RLE or image dimensions: x1=%d > shape[1]=%d" % (
            x1, shape[1]
        ))

    xc = (x0+x1)/(2*768)
    yc = (y0+y1)/(2*768)
    w = np.abs(x1-x0)/768
    h = np.abs(y1-y0)/768
    return [xc, yc, h, w]


import cv2
import numpy as np
import os
import glob
import shutil
import matplotlib.pyplot as plt
import pandas as pd 
import skimage.color
import skimage.filters
import seaborn as sns
from tqdm.notebook import tqdm_notebook as tqdm
import shutil
from sldl.image import ImageSR
from PIL import Image

#Import the required Libraries
from tkinter import *
from tkinter import ttk

#Create an instance of Tkinter frame
win= Tk()

#Set the geometry of Tkinter frame
win.geometry("750x250")

#Create an Entry widget to accept User Input
entry= Entry(win, width= 40)
entry.focus_set()
entry.pack()

def display_text():
    
    print(type(entry.get()))
    image = np.array(cv2.imread(entry.get())[:,:,::-1])
    
    
    RealESRGANsr = ImageSR('RealESRGAN')
    BSRGANsr = ImageSR('BSRGAN')
    swinIRsr = ImageSR('SwinIR-M')

    print('Working on ESRGAN')
    ESRGANimage = RealESRGANsr(image)
    print('Working on BSRGAN')
    BSRGANimage = BSRGANsr(image)
    print('Working on swinIR')
    swinIRimage = swinIRsr(image)
    
    ESRGANimage.save('results/ESRGANimage.jpg')
    BSRGANimage.save('results/BSRGANimage.jpg')
    swinIRimage.save('results/swinIRimage.jpg')
    
    "F:/Final_Year_Project/WrkDrctr/airbus_ship_detection/train_v2/d66cd8539.jpg"
    
    ESRGANimage = cv2.cvtColor(cv2.imread('results/ESRGANimage.jpg'),cv2.COLOR_BGR2RGB)
    BSRGANimage = cv2.cvtColor(cv2.imread('results/BSRGANimage.jpg'),cv2.COLOR_BGR2RGB)
    swinIRimage = cv2.cvtColor(cv2.imread('results/swinIRimage.jpg'),cv2.COLOR_BGR2RGB)
    
    fig, axs = plt.subplots(1, 4, figsize = (14, 8))
    axs[0].imshow(image)
    axs[0].set_title("Normal Image")
    
    axs[1].imshow(ESRGANimage)
    axs[1].set_title("ESRGAN Enhanced Image")
    
    axs[2].imshow(BSRGANimage)
    axs[2].set_title("BSRGAN Enhanced Image")
    
    axs[3].imshow(swinIRimage)
    axs[3].set_title("swinIR Enhanced Image")
    
    print("Completed Enchancing")

#Create a Button to validate Entry Widget
ttk.Button(win, text= "detect",width= 20, command= display_text).pack(pady=20)

win.mainloop()

# ships = pd.read_csv("airbus_ship_detection/train_ship_segmentations_v2.csv")
# ships["Ship"] = ships["EncodedPixels"].map(lambda x:1 if isinstance(x,str) else 0)
# ship_unique = ships[["ImageId","Ship"]].groupby("ImageId").agg({"Ship":"sum"}).reset_index()

# ships["Boundingbox"] = ships["EncodedPixels"].apply(lambda x:rle2bbox(x,(768,768)) if isinstance(x,str) else np.NaN)
# # ships.drop("EncodedPixels", axis =1, inplace =True)
# ships["BoundingboxArea"]=ships["Boundingbox"].map(lambda x:x[2]*768*x[3]*768 if x==x else 0)
# ships = ships[ships["BoundingboxArea"]>np.percentile(ships["BoundingboxArea"],1)]
# balanced_df = ship_unique.groupby("Ship").apply(lambda x:x.sample(1000) if len(x)>=1000 else x.sample(len(x)))
# balanced_df.reset_index(drop=True,inplace=True)
# balanced_bbox = ships.merge(balanced_df[["ImageId"]], how ="inner", on = "ImageId")
# print(balanced_bbox.head(20))

# path ="airbus_ship_detection/train_v2"
# plt.figure(figsize =(20,20))
# for i in range(15):
#     imageid = balanced_df[balanced_df.Ship ==i].iloc[0][0]
#     print(imageid)
#     image = np.array(cv2.imread(path+"/"+imageid)[:,:,::-1])
#     if i>0:
#         bbox = balanced_bbox[balanced_bbox.ImageId==imageid]["Boundingbox"]
        
#         for items in bbox:
#             Xmin  = int((items[0]-items[3]/2)*768)
#             Ymin  = int((items[1]-items[2]/2)*768)
#             Xmax  = int((items[0]+items[3]/2)*768)
#             Ymax  = int((items[1]+items[2]/2)*768)
#             cv2.rectangle(image,
#                           (Xmin,Ymin),
#                           (Xmax,Ymax),
#                           (255,0,0),
#                           thickness = 2)
#     plt.subplot(4,4,i+1)
#     plt.imshow(image)
#     plt.title("Number of ships = {}".format(i))


#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')
get_ipython().system(' pip install face_recognition')


# In[3]:


import dlib
import numpy as np
from PIL import Image
import os
from ISR.models import RDN, RRDN
import matplotlib.pyplot as plt
import cv2
import face_recognition


# In[2]:


# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detection(img,threshold = 1.08):
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, threshold, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return faces, Image.fromarray(img)


def extract(img,faces):
    faces_extracted = []
    for i in range(len(faces)):
        x1, y1, width, height = faces[i]
        x2, y2 = x1 + width, y1 + height

        # extract the face
        faces_extracted.append(img[y1:y2, x1:x2])
    return faces_extracted


def disp_side_by_side(imgs, titles = range(11)):
    plt.clf(); plt.cla(); plt.close();
    f, axs = plt.subplots(1, len(imgs), figsize=(16,16))
    for i in range(len(imgs)):
        axs[i].imshow(imgs[i], cmap = plt.cm.gray)
        axs[i].set_title(titles[i])
        axs[i].axis('off')
    plt.tight_layout(pad=-2)
    plt.show()




def is_player (image, im_ref_embeded, threshold = 0.5):
    target_embeded = face_recognition.face_encodings(image)[0]
    results = face_recognition.compare_faces([target_embeded],im_ref_embeded,tolerance=threshold)
    distance = face_recognition.face_distance([target_embeded],im_ref_embeded)
    return results, distance


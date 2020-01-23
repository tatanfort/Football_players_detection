#!/usr/bin/env python
# coding: utf-8

# In[3]:

!git clone https://github.com/idealo/image-super-resolution
!cd image-super-resolution

import numpy as np
from PIL import Image
import os
from ISR.models import RDN, RRDN
import matplotlib.pyplot as plt


# In[2]:


owd = os.getcwdb()
os.chdir(r'image-super-resolution')
get_ipython().system('python setup.py install')
os.chdir(owd)


# In[7]:


img = Image.open('team_reduced.jpg')
lr_img = np.array(img)


# In[8]:


model = RDN(weights='noise-cancel')
def upscale(img):

    sr_img = model.predict(np.array(img))
    return Image.fromarray(sr_img)


# In[ ]:





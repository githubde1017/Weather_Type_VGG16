#!/usr/bin/env python
# coding: utf-8

# In[7]:


from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import  img_to_array, load_img
from PIL import Image
import sys
import numpy as np
from glob import glob
from os.path import splitext
import matplotlib.pyplot as plt


# In[8]:


from tensorflow.keras.preprocessing.image import array_to_img
# 從參數讀取圖檔路徑
files = glob( "Weather/test/*.[jJ][pP][gG]" )

# 載入訓練好的模型
model = load_model('Weather_model_H_move_ocean_Estwind.h5')

cls_list = ['Est_season_wind','H_move_ocean']

# 辨識每一張圖
for f in files:
    img = image.load_img(f, target_size=(128, 128))
    if img is None:
        continue
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    pred = model.predict(x)[0]
    predictions=model.predict(x)
    top_inds = pred.argsort()[::-1][:5]
    print(f)
    for i in top_inds:
            print('    {:.3f}  {}'.format(pred[i], cls_list[i]))  
#    print(image.img_to_array(img))
#    print(np.expand_dims)
#    plt.imshow(img)
#    plt.show()
    
    if predictions[0,0] >= 0.5: 
        print('I am {:.2%} sure this is Est_season_wind'.format(predictions[0,0]))
    else: 
        print('I am {:.2%} sure this is H_move_ocean'.format(1-predictions[0,1]))
    print(pred)
    plt.imshow(array_to_img(x[0]))
    plt.show()


# In[ ]:





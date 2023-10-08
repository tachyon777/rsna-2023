#!/usr/bin/env python
# coding: utf-8

# # TotalSegmentatorを用いたbody(体輪郭)ラベルの作成

# In[1]:


import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

# リポジトリtopに移動
while os.path.basename(os.getcwd()) != 'rsna-2023':
    os.chdir('../')
    if os.getcwd() == '/':
        raise Exception('Could not find project root directory.')
sys.path.append("./models")
sys.path.append("./models/totalsegmentator")
from totalsegmentator.python_api import totalsegmentator


# In[2]:


df_train = pd.read_csv('data/rsna-2023-abdominal-trauma-detection/train.csv')
df_train_image_level = pd.read_csv('data/rsna-2023-abdominal-trauma-detection/image_level_labels.csv')
df_train_serirs_meta = pd.read_csv('data/rsna-2023-abdominal-trauma-detection/train_series_meta.csv')

dataset_dir = "data/dataset003"


# In[3]:


base_dir = "data/rsna-2023-abdominal-trauma-detection/segmentations/"
for sid in os.listdir(base_dir)[:2]:
    sid = int(sid.replace(".nii", ""))
    pid = df_train_serirs_meta[df_train_serirs_meta["series_id"] == sid]["patient_id"].values[0]
    ct_images_dir = f"data/rsna-2023-abdominal-trauma-detection/train_images/{pid}/{sid}"
    output_dir = os.path.join(dataset_dir, "segmentations", str(pid), str(sid))
    os.makedirs(output_dir, exist_ok=True)
    ret = totalsegmentator(ct_images_dir, output_dir, task="body", verbose=True)


# In[ ]:





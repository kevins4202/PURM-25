#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import dataframe_image as dfi
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# In[8]:


# Load your CSV file
presence_df = pd.read_csv("tables/presence.csv")
stance_df = pd.read_csv("tables/stance.csv")
socialneeds_df = pd.read_csv("tables/social_needs.csv")


# In[9]:


#presence_df


# In[16]:


stance_df_concat = stance_df[stance_df.columns[1:]].copy()
stance_df_concat.columns = ["stance_"+col for col in stance_df_concat.columns]
presence_df_concat = presence_df.copy()
presence_df_concat.columns = ["model_prompt"] + ["presence_"+col for col in presence_df_concat.columns[1:]]






# In[23]:

# In[24]:


def highlight_max(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]

# In[17]:
all_labels_df = pd.concat([presence_df_concat, stance_df_concat], axis=1)
all_labels_df


df_styled = all_labels_df.style.apply(highlight_max, axis=0).set_properties(**{
    'text-align': 'center',
    'border': '1px solid black',
    'padding': '5px',
    'font-size': '12pt'
}).set_table_styles([
    {'selector': 'thead th', 'props': [('background-color', '#f2f2f2')]},
    {'selector': 'tbody td', 'props': [('border', '1px solid #ccc')]}
])
df_styled

# In[34]:


#df_styled


# In[36]:


dfi.export(df_styled, "all_labels_metrics.png", max_cols=-1)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[6]:


import _pickle as pickle
with open(r"cl_NB.pickle", "rb") as input_file:
    cl_NB = pickle.load(input_file)
cl_NB.classify('Warriors Forward Andre Iguodala Earns NBA All-Defensive First Team Honors.')


# In[ ]:





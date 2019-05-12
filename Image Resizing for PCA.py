#!/usr/bin/env python
# coding: utf-8

# In[107]:
'''Shreyash Shrivastava
    1001397477'''


from PIL import Image
import os, sys

path = "Grape/" # Folder path to be resized
dirs = os.listdir(path)

def resize():
    count = 0 
    for item in dirs:
        if os.path.isfile(path+item):
#             if imagePath == directory + '.DS_Store':
#                 continue
        
            im = Image.open(path+item)
            
            
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            f, e = os.path.splitext(path+item)
            imResize = im.resize((25,25), Image.ANTIALIAS)
            imResize.save( 'resized_fruits/' + f + 'resized.jpg', 'JPEG', quality=100)
            count+=1

resize()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





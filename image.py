# %%

import os
from PIL import Image

# %%

# you need to change this to your data directory
train_dir = './image'
def chage_image():
    filenames = os.listdir(train_dir)
    filenames = [os.path.join(train_dir, item) for item in filenames]
    for filename in filenames:
        # print (filename)
        try:
            im = Image.open(filename)
            im = im.resize((208, 208))
            im = im.convert('RGB')
            im.save(filename)
        except OSError:
            print (filename)
            os.remove(filename)


chage_image()
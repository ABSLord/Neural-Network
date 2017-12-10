import os
import sys
import numpy as np
from scipy.misc import imresize
from PIL import Image


def resize(basedir, h, w):
    for d, dirs, files in os.walk(basedir):
        for f in files:
            img = os.path.join(d, f)
            img = np.asarray(Image.open(img))
            img = imresize(img, (w, h))
            img = Image.fromarray(img.astype("uint8"))
            img.save(os.path.join(d, f))


if __name__ == '__main__':
    basedir, h, w = sys.argv[1:]
    resize(basedir, int(h), int(w))

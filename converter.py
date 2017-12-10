from PIL import Image
import warnings
import numpy as np
from sklearn import preprocessing

warnings.filterwarnings("ignore")


# 0 or 1
BIN_MODE = 0

# -1 or 1
BIPOLAR_MODE = 1


def pred(r, g, b, c, e):
    return (r.all() >= c[0] - e) and (r.all() <= c[0] + e) and \
           (g.all() >= c[1] - e) and (g.all() <= c[1] + e) and \
           (b.all() >= c[2] - e) and (b <= c[2] + e)


class Converter:

    def img_to_nparray(self, img, mode=BIN_MODE, bcolor=(0, 0, 0), eps=10, vectorize=False):
        arr = np.array(Image.open(img))
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        result = np.zeros((arr.shape[0], arr.shape[1]))
        result[pred(r, g, b, bcolor, eps)] = 1
        if mode == BIN_MODE:
            result[~pred(r, g, b, bcolor, eps)] = 0
        else:
            result[~pred(r, g, b, bcolor, eps)] = -1
        if vectorize:
            result = result.reshape((result.shape[0] * result.shape[1], 1))
        return result

    def img_to_bwimage(self, arr, name, mode):
        img = np.zeros((arr.shape[0], arr.shape[1], 3))
        img[arr == 1] = 0, 0, 0
        if mode == BIN_MODE:
            img[arr == 0] = 255, 255, 255
        else:
            img[arr == -1] = 255, 255, 255
        img = Image.fromarray(img.astype("uint8"))
        img.save("./" + name)

    def img_to_range(self, img, dim, left, right):
        data = np.asarray(Image.open(img)).reshape((dim, ))
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(left, right))
        result = min_max_scaler.fit_transform(data)
        return result

    def data_to_range(self, data, left, right):
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(left, right))
        result = min_max_scaler.fit_transform(data)
        return result

    def to_file(self, arr, name):
        np.savetxt("./" + name, arr, fmt="%i", delimiter=",")

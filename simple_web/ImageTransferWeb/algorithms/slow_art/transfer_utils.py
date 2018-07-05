import skimage
import skimage.io
import skimage.transform
import numpy as np
import os


# return image shape [256,256,3]
# [h ,w, channel]
def load_image_crop(path):
    """
    :param path: img path
    :return:  image shape [256,256,3] scaled [0,1]
    """
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # crop image from center
    short_edge = min(img.shape[:2])
    yy = np.int32((img.shape[0] - short_edge) / 2)  # shape[0]:height
    xx = np.int32((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 256, 256
    resized_img = skimage.transform.resize(crop_img, (256, 256))
    return resized_img


def load_image_resize(path, height=None, width=None):
    """
    :param path: img path
    :param height: the wanted h
    :param width: the wanted w
    :return: resized img as para:h and para:w if none return img itself but all scaled [0,1]
    """
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = np.int32(img.shape[1] * ny / img.shape[0])
    elif width is not None:
        nx = width
        ny = np.int32(img.shape[0] * nx / img.shape[1])
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def save_image(path, img):
    # img can be scale [0-255] or [0-1]
    skimage.io.imsave(path, img)


def list_files(path):
    files = []
    # os.walk get the tree like file system's path dir files recurrently
    for(dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        break
    return files


if __name__ == '__main__':
    l = list_files('F:/ps文件/素材')
    print(len(l))
    print(l)
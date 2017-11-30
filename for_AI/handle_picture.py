#coding=utf-8
from __future__ import print_function
from scipy.misc import imread, imresize
import os, pickle, time, fnmatch
import collections
from PIL import Image, ImageOps
import numpy as np
import scipy.misc
from os.path import basename, splitext
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tag import _pos_tag, PerceptronTagger
def get_resized_image(img_path, height, width, save=True):
    image = Image.open(img_path)
    # it's because PIL is column major so you have to change place of width and height
    # this is stupid, i know
    image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)
    if save:
        image_dirs = img_path.split('/')
        image_dirs[-1] = 'resized_' + image_dirs[-1]
        out_path = '/'.join(image_dirs)
        if not os.path.exists(out_path):
            image.save(out_path)
    image = np.asarray(image, np.float32)
    return np.expand_dims(image, 0)

def save_image(path, image):
    # Output should add back the mean pixels we subtracted at the beginning
    image = image[0] # the image
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def glob_recursive(pathname, pattern):
    files = []
    for root, dirnames, filenames in os.walk(pathname):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def get_images(in_size):
    filenames = glob_recursive('/home/xzhou/for_AI/wikipedia_dataset/images/history', '*.jpg')
    number_image = len(filenames)
    image_data = {}
    image_list = []
    start = time.time()

    for i in range(number_image):
        img_data = [imresize(imread(filenames[i], mode='RGB'), (in_size, in_size))]#这里虽然reshape,但是因为规定了为RGB形式，所以得到的仍是三维的
        image_data[splitext(basename(filenames[i]))[0]] = img_data
        image_list.append(splitext(basename(filenames[i]))[0])
        
    pickle.dump(image_data, open('/home/xzhou/for_AI/wikipedia_dataset/raw_history_images.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    print('Finished')
    return image_list,number_image

def load_images():
    data = pickle.load(open('/home/xzhou/for_AI/wikipedia_dataset/raw_history_images.pkl', 'rb'))#得到的字典，数值为list形式
    return data

def main():
    filenames,number = get_images(200)
    load_images(filenames)

##if __name__ == '__main__':
##    main()

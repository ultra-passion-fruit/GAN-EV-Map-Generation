import os
import cv2 
from random import shuffle
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def load_data(test_pc=0.3, img_size=256):
    data = []
    images_directory = "routes-generated/final"

    # image file names are as follows: day-##-hour-##.png
        # ex. day-12-hour-3.png
        # image for the 12th day at 3am
        # label is 3, day is ignored

    # going through images in folder
    for image_name in tqdm(os.listdir(images_directory)):
        # image path
        image_path = os.path.join(images_directory, image_name)
        
        # get label (hour) from file name
        label = image_name.split('-')[3].split('.')[0]

        # resizing each image to 256x256
        image = cv2.resize(cv2.imread(image_path), (img_size, img_size))
        
        # appending image along with label to list
        data.append([np.array(image), np.array(np.uint8(label))])
    
    # shuffling images
    shuffle(data)

    # get all images into separate list
    images = [pair[0] for pair in data]
    # get all labels into separate list
    labels = [pair[1] for pair in data]

    # gets index at which test_pc (say 0.3) happens, takes ceiling to avoid decimal index and casts to int
    test_index_split = int(np.ceil(len(data)*test_pc))

    # split test and train
    # note: already shuffled so ok to slice as beginning:end
    trainX = images[test_index_split:]
    trainy = labels[test_index_split:]
    testX = images[:test_index_split]
    testy = labels[test_index_split:]

    # return as (trainX, trainy), (testX, testy) tuple 
    return (np.array(trainX), np.array(trainy)), (np.array(testX), np.array(testy))
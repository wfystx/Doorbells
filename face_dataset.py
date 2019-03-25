import os
import sys
import numpy as np
import cv2

IMAGE_SIZE = 64
path_name = "./new/"   # where to find photos

# resize images
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    h, w, _ = image.shape
    longest_edge = max(h, w)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return cv2.resize(constant, (height, width))

def load_dataset():
    images = []
    labels = []
    for dir_item in os.listdir(path_name):
        # find directories and photos inside
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        for photos in os.listdir(full_path):
            photo_name = os.path.abspath(os.path.join(full_path, photos))
            if photo_name.endswith('.jpg'):
                image = cv2.imread(photo_name)
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                #print (image)
                images.append(image)
                labels.append(full_path)
                #print (labels)
                #print (len(images))
    images = np.array(images)
    print(images.shape)
    labels = np.array([0 if label.endswith('Hao') else 1 for label in labels])    # labels...
    #print (labels)
    #print (len(labels))
    return images, labels


if __name__ == '__main__':
    images, labels = load_dataset()



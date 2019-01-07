import os
from PIL import Image
import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import imageio
from tqdm import tqdm

def show_img(x):
    cv2.imshow('image', x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def data_aug(data_path):
    sometimes = lambda aug: iaa.Sometimes(0.8, aug)

    seq = iaa.Sequential([
        iaa.Crop(percent=(0, 0.1)),
        iaa.Fliplr(0.5), # horizontally flip 50% of the images
        iaa.Flipud(0.5), # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
        sometimes(iaa.Affine(
                rotate=(-45, 45), # rotate by -45 to +45 degrees
            ))
    ], random_order=True)

    label_list = []
    img_list = []
    label_idx = 0

    for root, dirs, files in os.walk(data_path):
        if not files:
            continue
        for filename in files:
            img_path = os.path.join(root, filename)

            try:
                img = cv2.imread(img_path, 1)
                img = img[250:750, 250:750]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
            except:
                continue
            label_list.append((root, filename))
            img_list.append(img)
        label_idx += 1

    for i in range(20):
        images_aug = seq.augment_images(img_list)

        for img, meta in tqdm(zip(images_aug, label_list)):
            # show_img(img)
            imageio.imwrite(os.path.join(meta[0], str(i) + meta[1]), img)

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

matplotlib.use('TkAgg')


def get_weighted_sum(channel1, channel2, value):
    return (value*channel1 + (1-value)*channel2).astype(np.uint8)


def get_weighted_images(image1, image2,
                        values: list = [0, 0.25, 0.5, 0.75, 1]):
    B1, G1, R1 = cv2.split(image1)
    B2, G2, R2 = cv2.split(image2)

    images = []
    for value in values:
        image = [get_weighted_sum(B1, B2, value),
                 get_weighted_sum(G1, G2, value),
                 get_weighted_sum(R1, R2, value)]
        image = cv2.merge(image)
        images.append(image)

    return images


image1 = cv2.imread('avatar.jpg')
image2 = cv2.imread('avatar2.jpg')

fig, ax = plt.subplots(1, 5, figsize=(15, 10))
plt.subplots_adjust(top=0.95, left=0.046, bottom=0.18, right=0.95, wspace=0.1)

# 5 images
values = [0, 0.25, 0.5, 0.75, 1]
images = get_weighted_images(image1, image2, values)

for i, image in enumerate(images):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax[i].imshow(image)
    ax[i].axis('off')
    ax[i].set_title(f'a = {values[i]}')


plt.show()

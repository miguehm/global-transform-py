import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.widgets import CheckButtons

matplotlib.use('TkAgg')


def apply_threshold(image, threshold: int, invert=False):
    """
    Return image with threshold and its mask
    """

    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(image_bw.shape, dtype=np.uint8)

    if invert:
        mask[image_bw < threshold] = 1
    else:
        mask[image_bw >= threshold] = 1

    B, G, R = cv2.split(image)

    channel_B = np.zeros(image_bw.shape, dtype=np.uint8)
    channel_G = np.zeros(image_bw.shape, dtype=np.uint8)
    channel_R = np.zeros(image_bw.shape, dtype=np.uint8)

    channel_B[mask == 1] = B[mask == 1]
    channel_G[mask == 1] = G[mask == 1]
    channel_R[mask == 1] = R[mask == 1]

    image_container = cv2.merge([channel_B, channel_G, channel_R])
    image_container = cv2.cvtColor(image_container, cv2.COLOR_BGR2RGB)

    mask = mask * 255

    return image_container, mask


def main():
    image = cv2.imread('avatar.jpg')

    threshold = 0.1

    image_container, mask = apply_threshold(image, threshold, invert=False)

    data = {}

    # plt.figure(figsize=(15, 10))
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    plt.subplots_adjust(top=0.95, left=0.1, bottom=0.18)

    data['mask'] = ax[0].imshow(mask, cmap='coolwarm')
    ax[0].set_title('Mask')

    data['image_container'] = ax[1].imshow(image_container, cmap='gray')
    ax[1].set_title('Result')

    # Configuración de la barra deslizante
    ax_slider = plt.axes([0.1, 0.1, 0.3, 0.03])
    slider = Slider(ax_slider, 'Threshold', 0, 255, valinit=threshold)

    # Crear las casillas de verificación
    ax_check = plt.axes([0.45, 0.1, 0.05, 0.05])
    check = CheckButtons(ax_check, labels=['Invert'])

    def update(val):
        threshold = slider.val
        invert = check.get_status()[0]

        image_container, mask = apply_threshold(image, threshold, invert)
        data['mask'].set_data(mask)
        data['image_container'].set_data(image_container)
        # img_result.set_data(image_threshold)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def check_update(label):
        update(None)

    check.on_clicked(check_update)

    plt.show()


if __name__ == '__main__':
    main()

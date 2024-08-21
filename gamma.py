import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def create_lut(gamma: float):
    """
    Create a lut with gamma transformation
    """

    gamma_inv = 1.0/gamma
    table = np.arange(256)  # array with values 0..255

    # print(f'Gamma: {gamma}')
    # print(f'Original table: \n{table}')

    table = np.array([np.round(((i / 255) ** gamma_inv) * 255.0)
                     for i in table]).astype(np.uint8)
    # print(f'Lut table: \n{table}')

    for i in range(len(table)):
        value_normalized = table[i] / 255.0  # 0..1 range
        result = value_normalized ** gamma_inv
        result *= 255  # 0..255 range
        table[i] = np.round(result)

    return table


def get_gamma(image, gamma: float):
    """
    Return an image with gamma transformation
    """

    B, G, R = cv2.split(image)

    # pasar la información de la matriz B a la matriz container
    container_B = B
    container_G = G
    container_R = R

    lut = create_lut(gamma)

    for i in range(len(container_B)):
        value_B = container_B[i]
        container_B[i] = lut[value_B]

    for i in range(len(container_G)):
        value_G = container_G[i]
        container_G[i] = lut[value_G]

    for i in range(len(container_R)):
        value_R = container_R[i]
        container_R[i] = lut[value_R]

    image_merge = cv2.merge([container_B, container_G, container_R])

    return image_merge


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    data = {}

    image = cv2.imread('avatar.jpg')

    gamma = 1.7
    image_gamma = get_gamma(image, gamma)
    B, G, R = cv2.split(image_gamma)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(left=0.1, bottom=0.134, right=0.9,
                        top=0.952, wspace=0.076, hspace=0.114)

    data['red'] = ax[0, 0].imshow(R, cmap='gray')
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Red Channel')

    data['green'] = ax[0, 1].imshow(G, cmap='gray')
    ax[0, 1].axis('off')
    ax[0, 1].set_title('Green Channel')

    data['blue'] = ax[0, 2].imshow(B, cmap='gray')
    ax[0, 2].axis('off')
    ax[0, 2].set_title('Blue Channel')

    image_gamma = cv2.cvtColor(image_gamma, cv2.COLOR_BGR2RGB)

    data['image'] = ax[1, 1].imshow(image_gamma)
    ax[1, 1].axis('off')
    ax[1, 1].set_title('Channels merged')

    ax[1, 0].axis('off')
    ax[1, 2].axis('off')

    # Configuración de la barra deslizante
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider = Slider(ax_slider, 'Gamma', 0.1, 6.0, valinit=gamma)

    def update(val):
        gamma = slider.val
        image_gamma = get_gamma(image, gamma)
        B, G, R = cv2.split(image_gamma)

        image_gamma = cv2.cvtColor(image_gamma, cv2.COLOR_BGR2RGB)
        data['image'].set_data(image_gamma)
        data['blue'].set_data(B)
        data['green'].set_data(G)
        data['red'].set_data(R)

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

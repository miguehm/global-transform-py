# read image
# get threshold
# apply gamma to threshold
# merge result

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.widgets import CheckButtons

from gamma import get_gamma, create_lut
from threshold import apply_threshold
from exposure import increase_exposure

matplotlib.use('TkAgg')


def main():
    image = cv2.imread('./dog.png')

    gamma = 1.0
    threshold = 150
    alpha = 1
    beta = 0

    def gamma_threshold(image, gamma, threshold, alpha, beta, invert=False):
        image_threshold, _ = apply_threshold(image, threshold, False)
        image_threshold_invert, _ = apply_threshold(image, threshold, True)

        result = np.zeros(image_threshold.shape, dtype=np.uint8)

        # TODO:
        # Exposure only affects mask
        if invert:
            image_gamma = get_gamma(image_threshold_invert, gamma)
            image_gamma = increase_exposure(image_gamma, alpha, beta)
            result = image_gamma + image_threshold
        else:
            image_gamma = get_gamma(image_threshold, gamma)
            image_gamma = increase_exposure(image_gamma, alpha, beta)
            result = image_gamma + image_threshold_invert
        return result

    # TODO:
    # - Verify whitch function is converting BGR2RGB by default

    # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    result = gamma_threshold(image, gamma, threshold, alpha, beta, False)

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    plt.subplots_adjust(left=0.1, bottom=0.134, right=0.9,
                        top=0.952, wspace=0.076, hspace=0.114)

    data = {}

    data['result'] = ax.imshow(result)

    # Configuración de la barra deslizante
    ax_slider1 = plt.axes([0.1, 0.2, 0.3, 0.03])
    slider1 = Slider(ax_slider1, 'Threshold', 0, 255, valinit=threshold)

    ax_slider2 = plt.axes([0.1, 0.15, 0.3, 0.03])
    slider2 = Slider(ax_slider2, 'Gamma', 0.1, 6, valinit=gamma)

    ax_slider3 = plt.axes([0.1, 0.1, 0.3, 0.03])
    slider3 = Slider(ax_slider3, 'alpha', 0, 40, valinit=alpha)

    ax_slider4 = plt.axes([0.1, 0.05, 0.3, 0.03])
    slider4 = Slider(ax_slider4, 'alpha', 0, 100, valinit=beta)

    # Crear las casillas de verificación
    ax_check = plt.axes([0.45, 0.1, 0.05, 0.05])
    check = CheckButtons(ax_check, labels=['Invert'])

    def update(val):
        threshold = slider1.val
        gamma = slider2.val
        alpha = slider3.val
        beta = slider4.val

        invert = check.get_status()[0]

        result = gamma_threshold(image, gamma, threshold, alpha, beta, invert)
        data['result'].set_data(result)
        # img_result.set_data(image_threshold)
        fig.canvas.draw_idle()

    slider1.on_changed(update)
    slider2.on_changed(update)
    slider3.on_changed(update)
    slider4.on_changed(update)

    def check_update(label):
        update(None)

    check.on_clicked(check_update)

    plt.show()


if __name__ == '__main__':
    main()

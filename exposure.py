import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.widgets import CheckButtons


def increase_exposure(image, alpha: float, beta: float):
    image.astype(np.float32)

    result = alpha * image + beta

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def main():
    matplotlib.use('TkAgg')

    image = cv2.imread('./avatar.jpg')
    alpha = 1.5
    beta = 0

    result = increase_exposure(image, alpha, beta)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    plt.subplots_adjust(left=0.1, bottom=0.134, right=0.9,
                        top=0.952, wspace=0.076, hspace=0.114)

    data = {}

    data['result'] = ax.imshow(result)

    # Configuración de la barra deslizante
    ax_slider1 = plt.axes([0.1, 0.15, 0.3, 0.03])
    slider1 = Slider(ax_slider1, 'Alpha', 0, 10, valinit=alpha)

    ax_slider2 = plt.axes([0.1, 0.1, 0.3, 0.03])
    slider2 = Slider(ax_slider2, 'Beta', 0, 100, valinit=beta)

    def update(val):
        alpha = slider1.val
        beta = slider2.val

        result = increase_exposure(image, alpha, beta)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        data['result'].set_data(result)
        # img_result.set_data(image_threshold)
        fig.canvas.draw_idle()

    slider1.on_changed(update)
    slider2.on_changed(update)

    plt.show()


if __name__ == '__main__':
    main()

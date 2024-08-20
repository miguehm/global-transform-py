import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')

imagen = cv2.imread('avatar.jpg')

# calcular histograma de los tres canales rgb
hist_red = cv2.calcHist([imagen], [2], None, [256], [0, 256])
hist_green = cv2.calcHist([imagen], [1], None, [256], [0, 256])
hist_blue = cv2.calcHist([imagen], [0], None, [256], [0, 256])

x = np.arange(256)

# Mostrar la imagen en el canal red
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(imagen[:, :, 2], cmap='Reds')
plt.title('Canal Red')
plt.colorbar()

plt.subplot(2, 3, 2)
plt.imshow(imagen[:, :, 1], cmap='Greens')
plt.title('Canal Green')
plt.colorbar()

plt.subplot(2, 3, 3)
plt.imshow(imagen[:, :, 0], cmap='Blues')
plt.title('Canal Blue')
plt.colorbar()

# Mostrar los tres histogramas en una pantalla
plt.subplot(2, 3, 4)
plt.bar(x, hist_red[:, 0], width=1.0, edgecolor='black')
plt.xlim([0, 256])
plt.title('Canal Red')
plt.xlabel('Intensidad de píxeles')
plt.ylabel('Número de píxeles')

plt.subplot(2, 3, 5)
plt.bar(x, hist_green[:, 0], width=1.0, edgecolor='black')
plt.xlim([0, 256])
plt.title('Canal Green')
plt.xlabel('Intensidad de píxeles')
plt.ylabel('Número de píxeles')

plt.subplot(2, 3, 6)
plt.bar(x, hist_blue[:, 0], width=1.0, edgecolor='black')
plt.xlim([0, 256])
plt.title('Canal Blue')
plt.xlabel('Intensidad de píxeles')
plt.ylabel('Número de píxeles')

plt.tight_layout()
plt.show()

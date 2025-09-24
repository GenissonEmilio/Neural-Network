import cv2
import matplotlib.pyplot as plt
import numpy as np

# Carregar imagem
img = cv2.imread("imagem.png")

# Dimensões da imagem
row, col, ch = img.shape
mean = 0
sigma = 25

# visualizar como matriz
print(f"Imagem normal: {img.shape}")

# Converter para gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Imagem em gray_scale: {img_gray.shape}")

# Vizualização do histograma da imagem
plt.hist(img_gray.ravel(), bins=256, range=[0, 256])
plt.title("Histograma da imagem em Grayscale")
plt.xlabel("Intensidade")
plt.ylabel("Número de pixels")
plt.show()

# Adicionando ruido na imagem
gauss = np.random.normal(mean, sigma, (row, col, ch)).astype('uint8')
noisy = cv2.add(img, gauss)

plt.imshow(cv2.cvtColor(noisy, cv2.COLOR_BGRA2RGB))
plt.title("Imagem com Ruido")
plt.axis('off')
plt.show()

# Filtro Gaussiano para reduzir ruido na imagem
gaussian_filtered = cv2.GaussianBlur(noisy, (5,5), 2)

# Filtro mediano
median_filtered = cv2.medianBlur(noisy, 3)

# Imagem filtrada
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2RGB))
plt.title("Filtro Gaussiano")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(median_filtered, cv2.COLOR_BGR2RGB))
plt.title("Filtro Mediano")
plt.axis('off')

plt.show()

# Fazendo Segmentação na imagem
threshold_value = 128 # Limite da binarização (valores acima de 128 serão considerados claros)

_, thresholded = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY) # Aplica o Threshold

plt.subplot(1, 2, 1)
plt.imshow(thresholded, cmap='gray')
plt.title("Imagem Thresholding")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_gray, cmap='gray')
plt.title("Imagem gray scale")
plt.axis('off')

plt.show()
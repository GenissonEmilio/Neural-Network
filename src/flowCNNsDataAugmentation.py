import tensorflow as tf
from tensorflow import keras
import time

# --- Configuração ---
image_size = (299, 299)
batch_size = 32
num_classes = 3  # Baseado no seu dataset original (COVID, Normal, Pneumonia)

start_time = time.time()

# --- Carregamento dos Dados (Mantido) ---
path_train = "../data/treinamento400"
path_test = "../data/teste400"

# O 'input_shape' para o Rescaling agora está na primeira camada.
train_ds = keras.utils.image_dataset_from_directory(
    path_train, validation_split=0.2, subset="training", seed=1337,
    image_size=image_size, batch_size=batch_size
)
val_ds = keras.utils.image_dataset_from_directory(
    path_train, validation_split=0.2, subset="validation", seed=1337,
    image_size=image_size, batch_size=batch_size
)
test_ds = keras.utils.image_dataset_from_directory(
    path_test, image_size=image_size, batch_size=batch_size
)

# --- NOVO: DATA AUGMENTATION ---
# Criamos uma camada que faz a rotação e zoom aleatoriamente DURANTE o treinamento.
data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),  # Vira a imagem horizontalmente
        keras.layers.RandomRotation(0.1),  # Rotação aleatória de até 10%
        keras.layers.RandomZoom(0.1),  # Zoom aleatório de até 10%
    ]
)

# --- NOVO MODELO CNN AVANÇADO ---
model = keras.Sequential([
    # 1. Aumento de Dados
    data_augmentation,

    # 2. Pré-processamento: Normalização
    keras.layers.Rescaling(1. / 255),

    # Bloco 1: Convolução e Regularização
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),  # Dropout após o pooling

    # Bloco 2: Aumentando a profundidade
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    # Bloco 3: Camada mais profunda
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),  # Mais um Dropout

    # Camadas Densa para Classificação
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(num_classes, activation='softmax')
])

# --- Compilação do modelo ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Treinamento e Avaliação ---
print("Iniciando o treinamento do modelo CNN...")
model.fit(
    train_ds,
    epochs=10,
    verbose=2,
    validation_data=val_ds
)

print("\nAvaliando o modelo no conjunto de teste...")
score = model.evaluate(test_ds, verbose=2)

end_time = time.time()

print("\n-------------------------")
print(f"Loss final: {score[0]:.4f}")
print(f"Acurácia final: {score[1]:.4f}")
print(f"Tempo total de execução: {(end_time - start_time):.2f} segundos")
import tensorflow as tf
from tensorflow import keras
import time

# --- Configuraçã ---
image_size = (299, 299)
batch_size = 32

start_time = time.time()

# --- Carregando a Base de dados ---
path_train = "../data/treinamento400"
path_test = "../data/teste400"

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

# --- NOVO: DEFINIÇÃO DO MODELO CNN ---
model = keras.Sequential([
    # 1. Pré-processamento: Normaliza os valores de pixel de 0-255 para 0-1
    # Isso é essencial para o treinamento de deep learning.
    keras.layers.Rescaling(1. / 255, input_shape=(image_size[0], image_size[1], 3)),

    # 2. Primeira Camada de Convolução e Pooling
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    # 3. Segunda Camada de Convolução e Pooling
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    # 4. TERCEIRA Camada de Convolução e Pooling
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    # 5. Camadas Densa para Classificação
    keras.layers.Flatten(),
    # Adicionando Dropout para combater Overfitting (Boa prática!)
    keras.layers.Dropout(0.5),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(3, activation='softmax')  # 3 neurônios de saída (COVID, Normal, Pneumonia)
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
print(f"Acurácia final: {(score[1] * 100):.4f}")
print(f"Tempo total de execução: {(end_time - start_time):.2f} segundos")
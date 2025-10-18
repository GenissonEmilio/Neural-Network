import tensorflow as tf
from tensorflow import keras
import time

# Definição das variaveis
image_size = (299, 299)
batch_size = 32

# Contagem do tempo
start_time = time.time()

# --- CARREGANDO O DATASET PÚBLICO (EX: CHEST X-RAY PNEUMONIA) ---
path_base = "../data/chest_xray/"
path_train = path_base + "train"
path_val = path_base + "val"
path_test = path_base + "test"

# Conjunto de treinamento
train_ds = keras.utils.image_dataset_from_directory(
    path_train,
    seed=1337,
    image_size=image_size,
    batch_size=batch_size
)

# Conjunto de validação
val_ds = keras.utils.image_dataset_from_directory(
    path_val,
    seed=1337,
    image_size=image_size,
    batch_size=batch_size
)

# Conjunto de teste
test_ds = keras.utils.image_dataset_from_directory(
    path_test,
    image_size=image_size,
    batch_size=batch_size
)

# --- DEFINIÇÃO DO MODELO ---
model = keras.Sequential([
    # Redimensiona as imagens 256x256
    keras.layers.Resizing(256, 256),
    # Converte a imagem 2d em um vetor 1d
    keras.layers.Flatten(),
    # Camada densa com 128 neurônios e função de ativação ReLu
    keras.layers.Dense(128, activation='relu'),
    # Camada densa com 64 neurônios e função de ativação ReLu
    keras.layers.Dense(64, activation='relu'),
    # Camada de saida com 2 neurônios (PARA 2 CLASSES: Normal/Pneumonia)
    keras.layers.Dense(2, activation='softmax')
])

# Compilação do modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Treinamendo do modelo
print("Iniciando o treinamento do modelo com dataset de Pneumonia...")
model.fit(
    train_ds,
    epochs=10,
    verbose=2,
    validation_data=val_ds
)

# Avaliação do Modelo
print("\nAvaliando o modelo no conjunto de teste...")
score = model.evaluate(test_ds, verbose=2)

print("\n-------------------------")
print(f"Loss final: {score[0]:.4f}")
print(f"Acurácia final: {score[1]:.4f}")

# Fim da contagem de tempo
end_time = time.time()
print(f"Tempo total de execução: {(end_time - start_time):.2f} segundos")
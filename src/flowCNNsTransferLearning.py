import tensorflow as tf
from tensorflow import keras
import time

# --- Configuração ---
image_size = (299, 299)  # Mantemos 299x299, pois InceptionV3 prefere esse tamanho
batch_size = 32
num_classes = 3

start_time = time.time()

# --- Carregamento dos Dados (Mantido) ---
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

# --- NOVO: BASE MODEL PRÉ-TREINADO (TRANSFER LEARNING) ---

# 1. Carregar o modelo base (InceptionV3) sem as camadas de classificação (include_top=False)
base_model = keras.applications.InceptionV3(
    weights="imagenet",  # Usa pesos treinados no dataset ImageNet
    include_top=False,  # Remove a camada de classificação final do Inception
    input_shape=(image_size[0], image_size[1], 3)
)

# 2. Congelar as camadas do modelo base para que seus pesos não sejam alterados
base_model.trainable = False

# 3. Construir o novo modelo (com o InceptionV3 na base)
model = keras.Sequential([
    # O InceptionV3 já inclui a normalização de pixel
    base_model,

    # Camadas de Classificação (nossas novas camadas)
    keras.layers.GlobalAveragePooling2D(),  # Reduz o mapa de features
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

# --- Compilação do modelo (Usamos uma taxa de aprendizado menor, pois estamos 'refinando' um modelo existente) ---
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Treinamento e Avaliação ---
print("Iniciando o treinamento com Transfer Learning (InceptionV3)...")
# Reduzimos as épocas, pois Transfer Learning geralmente converge mais rápido
model.fit(
    train_ds,
    epochs=5,
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
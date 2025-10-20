import tensorflow as tf
from tensorflow import keras
import time

# --- Configuração ---
image_size = (299, 299)
batch_size = 32

start_time = time.time()

# --- Carregamento dos Dados (Mantido) ---
path_base = "../data/chest_xray/"
path_train = path_base + "train"
path_test = path_base + "test"
path_val = path_base + "val"

train_ds = keras.utils.image_dataset_from_directory(
    path_train, seed=1337,
    image_size=image_size, batch_size=batch_size
)
val_ds = keras.utils.image_dataset_from_directory(
    path_val, seed=1337,
    image_size=image_size, batch_size=batch_size
)
test_ds = keras.utils.image_dataset_from_directory(
    path_test, image_size=image_size, batch_size=batch_size
)

# =========================================================
# NOVAS CAMADAS: DATA AUGMENTATION (AUMENTO DE DADOS)
# Criamos uma sequência que aplica transformações aleatórias
# APENAS nos dados de TREINAMENTO.
# =========================================================
data_augmentation = keras.Sequential(
    [
        # 1. Espelhamento Horizontal (útil em imagens médicas)
        keras.layers.RandomFlip("horizontal"),
        # 2. Rotação Aleatória (até 10% do ângulo)
        keras.layers.RandomRotation(0.1),
        # 3. Zoom Aleatório (até 10%)
        keras.layers.RandomZoom(0.1),
        # 4. Ajuste de Contraste (ajuda na variação de raios-X)
        keras.layers.RandomContrast(0.2),
    ],
    name="Data_Augmentation_Layer"
)

# --- NOVO: DEFINIÇÃO DO MODELO CNN REFORÇADO ---
model = keras.Sequential([
    # 1. AUMENTO DE DADOS: Aplica as transformações na imagem antes do treino
    data_augmentation,

    # 2. Pré-processamento: Normaliza os valores de pixel
    keras.layers.Rescaling(1. / 255, input_shape=(image_size[0], image_size[1], 3)),

    # 3. Bloco 1: Convolução, Pooling e REGULARIZAÇÃO
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),  # NOVO: Dropout após o primeiro bloco

    # 4. Bloco 2
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),  # NOVO: Dropout após o segundo bloco

    # 5. Bloco 3
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    # 6. Camadas Densa
    keras.layers.Flatten(),

    # Camada Densa com Regularização Adicional
    keras.layers.Dense(128, activation='relu',
                       # NOVO: Regularização L2 nos pesos (incentiva pesos menores)
                       kernel_regularizer=keras.regularizers.l2(0.01)),

    # Aumentar a taxa de dropout para esta camada final densa
    keras.layers.Dropout(0.5),

    keras.layers.Dense(2, activation='softmax')
])

# --- Compilação do modelo (Taxa de aprendizado menor para estabilidade) ---
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Taxa de aprendizado mais baixa
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Treinamento e Avaliação ---
# Aumentamos as épocas para 15 ou 20, já que Data Augmentation exige mais tempo
# Mas usaremos um Callback para Parar o Treinamento Cedo (Early Stopping)
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitore a perda de validação
        patience=5,  # Pare se não houver melhora após 5 épocas
        restore_best_weights=True  # Restaure os melhores pesos
    )
]

print("Iniciando o treinamento do modelo CNN REFORÇADO...")
model.fit(
    train_ds,
    epochs=5,  # Tentamos 20 épocas, mas o EarlyStopping irá parar
    verbose=2,
    validation_data=val_ds,
    callbacks=callbacks  # Adiciona o Early Stopping
)

print("\nAvaliando o modelo no conjunto de teste...")
score = model.evaluate(test_ds, verbose=2)

end_time = time.time()

print("\n-------------------------")
print(f"Loss final: {score[0]:.4f}")
print(f"Acurácia final: {(score[1] * 100):.4f}")
print(f"Tempo total de execução: {(end_time - start_time):.2f} segundos")
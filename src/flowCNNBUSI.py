import tensorflow as tf
from tensorflow import keras
import time

# --- Configuração ---
image_size = (299, 299)
batch_size = 32
NUM_CLASSES = 3
TOTAL_SPLIT = 0.2

start_time = time.time()

# --- Carregando e Dividindo a Base de Dados (3 classes na raiz) ---
path_base = "../data/Dataset_BUSI_with_GT/"

train_ds = keras.utils.image_dataset_from_directory(
    path_base, validation_split=TOTAL_SPLIT, subset="training", seed=1337,
    image_size=image_size, batch_size=batch_size
)
val_ds = keras.utils.image_dataset_from_directory(
    path_base, validation_split=TOTAL_SPLIT, subset="validation", seed=1337,
    image_size=image_size, batch_size=batch_size
)
test_ds = val_ds

# --- Contagem de Imagens ---
print("\n--- Contagem de Imagens ---")
total_train = train_ds.cardinality().numpy() * batch_size
total_val = val_ds.cardinality().numpy() * batch_size
print(f"Total de Imagens de Treinamento: {total_train}")
print(f"Total de Imagens de Validação/Teste: {total_val}")

# --- DATA AUGMENTATION (Adicionado para robustez) ---
data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomContrast(0.2),
    ]
)

# --- DEFINIÇÃO DO MODELO CNN ROBUSTA ---
model = keras.Sequential([
    data_augmentation,
    keras.layers.Rescaling(1. / 255, input_shape=(image_size[0], image_size[1], 3)),

    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),

    # Camada Densa com Regularização L2
    keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),

    keras.layers.Dropout(0.5),
    keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# --- Compilação do modelo (Learning Rate mais baixo) ---
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- Treinamento e Avaliação (Com Early Stopping) ---
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

print("Iniciando o treinamento do modelo CNN Robusta (BUSI)...")
model.fit(train_ds, epochs=20, verbose=2, validation_data=val_ds, callbacks=callbacks)

print("\nAvaliando o modelo no conjunto de teste...")
score = model.evaluate(test_ds, verbose=2)

end_time = time.time()
print("\n-------------------------")
print(f"Loss final: {score[0]:.4f}")
print(f"Acurácia final: {(score[1] * 100):.4f}")
print(f"Tempo total de execução: {(end_time - start_time):.2f} segundos")
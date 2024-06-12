import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Impostazioni di base
image_size = (96, 96)
batch_size = 4

# Creare i generatori di immagini per train, validation e test
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Directory dei dati
train_dir = 'dataset/train'
valid_dir = 'dataset/valid'
test_dir = 'dataset/test'

# Generator per il training
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary'
)

# Generator per la validazione
valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary'
)

# Generator per il test
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary'
)

# Creare un modello estremamente leggero
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(4, (3, 3), activation='relu', input_shape=(96, 96, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilare il modello
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Addestrare il modello
model.fit(
    train_generator,
    epochs=10,
    validation_data=valid_generator
)

# Valutare il modello
test_loss, test_acc = model.evaluate(test_generator)
print('\nTest accuracy:', test_acc)

# Salvare il modello come SavedModel
model.export('model/saved_model')

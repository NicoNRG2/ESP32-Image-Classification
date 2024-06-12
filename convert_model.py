import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Impostazioni di base
image_size = (96, 96)

# Directory dei dati
train_dir = 'dataset/train'

# Creare il generatore di immagini per train
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=1,  # Usare batch_size 1 per rappresentazione
    class_mode='binary',
    shuffle=True
)

# Funzione di rappresentazione dei dati
def representative_data_gen():
    for _ in range(len(train_generator)):  # Utilizza tutte le immagini del dataset
        data, _ = next(train_generator)
        yield [data]

# Caricare il modello SavedModel
saved_model_dir = 'model/saved_model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Applicare la quantizzazione completa
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convertire il modello
tflite_model = converter.convert()

# Salvare il modello convertito
with open('model/model_quant.tflite', 'wb') as f:
    f.write(tflite_model)

print("Modello quantizzato salvato come 'model_quant.tflite'")

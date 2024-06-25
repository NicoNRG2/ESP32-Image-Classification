import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# Impostazioni di base
image_size = (96, 96)
batch_size = 1

# Directory dei dati
test_dir = 'dataset/test'

# Creare il generatore di immagini per il test
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Caricare il modello quantizzato
interpreter = tf.lite.Interpreter(model_path="model/model_quant.tflite")
interpreter.allocate_tensors()

# Ottenere dettagli sugli input e gli output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Funzione per preprocessare i dati di input
def preprocess_image(image):
    image = (image * 255).astype(np.uint8)  # Convertire l'immagine a uint8
    return image

# Funzione per verificare l'output del modello TFLite
def verify_tflite_model(interpreter, test_generator):
    correct_predictions = 0
    total_predictions = 0

    for i in range(len(test_generator)):
        data, labels = next(test_generator)
        data = preprocess_image(data)
        
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Convertire l'output in etichette binarie
        predicted_label = (output > 127).astype(int)  # Confronta con la metà della gamma uint8
        correct_predictions += (predicted_label == labels).sum()
        total_predictions += labels.size

        # Stampa alcune previsioni per verifica
        if i < 14:
            print(f"True Label: {labels}, Predicted Label: {predicted_label}, Raw Output: {output}")

    accuracy = correct_predictions / total_predictions
    return accuracy

# Valutare l'accuratezza del modello quantizzato
start_time = time.time()
accuracy = verify_tflite_model(interpreter, test_generator)
end_time = time.time()

print(f'Accuracy of the TFLite model: {accuracy * 100:.2f}%')
print(f'Total Inference Time: {end_time - start_time:.2f} seconds')

#Found 14 images belonging to 2 classes.
#True Label: [0.], Predicted Label: [[0]], Raw Output: [[2]]
#True Label: [0.], Predicted Label: [[0]], Raw Output: [[2]]
#True Label: [0.], Predicted Label: [[0]], Raw Output: [[10]]
#True Label: [0.], Predicted Label: [[0]], Raw Output: [[0]]
#True Label: [0.], Predicted Label: [[0]], Raw Output: [[19]]
#True Label: [0.], Predicted Label: [[0]], Raw Output: [[40]]
#True Label: [0.], Predicted Label: [[0]], Raw Output: [[70]]
#True Label: [1.], Predicted Label: [[1]], Raw Output: [[200]]
#True Label: [1.], Predicted Label: [[1]], Raw Output: [[143]]
#True Label: [1.], Predicted Label: [[1]], Raw Output: [[255]]
#True Label: [1.], Predicted Label: [[0]], Raw Output: [[89]]
#True Label: [1.], Predicted Label: [[1]], Raw Output: [[255]]
#True Label: [1.], Predicted Label: [[1]], Raw Output: [[177]]
#True Label: [1.], Predicted Label: [[1]], Raw Output: [[255]]
#Accuracy of the TFLite model: 92.86%
#Total Inference Time: 0.02 seconds
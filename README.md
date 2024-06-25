# ESP32-Image-Classification

This repository contains the code and resources for my bachelor's thesis project. The goal of this project is to develop a lightweight image classification model that can identify two classes (0 or 1) in grayscale images, and deploy this model on an ESP32 microcontroller.

## Project Overview

The project is divided into three main parts:

1. **Model Training with TensorFlow**
   - Images are resized to 96x96 and converted to grayscale.
   - A convolutional neural network (CNN) is trained to classify the images into two classes.
   - The trained model is then exported in save_model format.

2. **Model Conversion to TensorFlow Lite**
   - The trained model is then converted to TensorFlow Lite format with quantization.
   - The TensorFlow model is quantized to optimize it for the resource-constrained environment of the ESP32.
   - The quantized model is converted to a C array for deployment on the ESP32.

4. **Deployment on ESP32 using TensorFlow Lite Micro**
   - The TensorFlow Lite model is integrated into the ESP32 firmware.
   - The ESP32 captures grayscale images, normalizes the pixel values, and runs the inference using the quantized model.

## Repository Structure

```plaintext
├── dataset
│   ├── train
│   │   ├── 0
│   │   └── 1
│   ├── test
│   │   ├── 0
│   │   └── 1
│   └── valid
│       ├── 0
│       └── 1
├── model
│   ├── save_model
│   └── model_quant.tflite
├── train_model.py
├── convert_model.py
├── esp32-classification-project/
│   ├── .devcontainer
│   ├── .vscode
│   ├── build
│   ├── main
│   │   ├── CMakeLists.txt
│   │   ├── idf_component.yml
│   │   ├── main_functions.cc
│   │   ├── main_functions.h
│   │   ├── main.c
│   │   ├── main.cc
│   │   ├── model_data.cc
│   │   ├── model_data.h
│   │   ├── take_picture.cc
│   │   └── take_picture.h
│   ├── managed_components
│   ├── CMakeLists.txt
│   ├── dependencies.lock
│   ├── README.md
│   └── dkconfig
└── README.md
```

## Getting Started
### Prerequisites
 - Python 3.11 or higher
 - TensorFlow 2.16.1
 - ESP-IDF (Espressif IoT Development Framework)

### Training the Model
1. **Set up the dataset:**
   Ensure the dataset is structured as follows:
   ```plaintext
   dataset
   ├── train
   │   ├── 0
   │   └── 1
   ├── test
   │   ├── 0
   │   └── 1
   └── valid
       ├── 0
       └── 1
   ```

2. **Run the training script:**

   ```bash
   python train_model.py
   ```

### Converting the Model

1. **Run the conversion script:**
   ```bash
   python convert_model.py
   ```

3. **Generate the C array:**
   ```bash
   cd model
   xxd -i model_quant.tflite > ../esp32-classification-project/main/model_data.cc
   ```

### Deploying on ESP32
1. **Set up the ESP32 development environment:**
   - Install the ESP-IDF (Espressif IoT Development Framework)
   - Connect your ESP32 board to your computer

2. **Compile and upload the firmware:**
   - Ensure model_data.cc is included in the project.
   - Build the project:
     ```bash
     idf.py build
     ```
   - Flash the firmware to the ESP32 and monitor the output:
     ```bash
     idf.py flash monitor
     ```
3. **Run the inference on ESP32:**
   The ESP32 will capture grayscale images, normalize the pixel values, and perform the classification.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
 - TensorFlow Team for providing the tools and resources.
 - Espressif Systems for the ESP32 platform.
 - My university and professors for their guidance and support.

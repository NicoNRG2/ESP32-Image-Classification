#include "main_functions.h"
#include "model_data.h"
#include "take_picture.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_timer.h"

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 80 * 1024;  // 80 KB
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

void setup() {
    model = tflite::GetModel(model_quant_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model provided is schema version %d not equal to supported "
                    "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Specify only the necessary operators using MicroMutableOpResolver
    static tflite::MicroMutableOpResolver<8> resolver;
    if (resolver.AddConv2D() != kTfLiteOk) {
        return;
    }
    if (resolver.AddMaxPool2D() != kTfLiteOk) {
        return;
    }
    if (resolver.AddFullyConnected() != kTfLiteOk) {
        return;
    }
    if (resolver.AddRelu() != kTfLiteOk) {
        return;
    }
    if (resolver.AddLogistic() != kTfLiteOk) {
        return;
    }
    if (resolver.AddQuantize() != kTfLiteOk) {
        return;
    }
    if (resolver.AddDequantize() != kTfLiteOk) {
        return;
    }
    if (resolver.AddReshape() != kTfLiteOk) {
        return;
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    if(ESP_OK != init_camera()) {
        MicroPrintf("Failed to initialize camera");
        return;
    }
}

void loop() {
    // Buffer to store the captured image
    uint8_t* image_data = (uint8_t*)malloc(96 * 96);

    if (!image_data) {
        MicroPrintf("Failed to allocate memory for image buffer");
        return;
    }

    // Capture the image
    if (grayscale_image(image_data) != ESP_OK) {
        MicroPrintf("Image capture failed");
        free(image_data);
        return;
    }

    // Place the quantized input in the model's input tensor
    memcpy(input->data.uint8, image_data, input->bytes);

    // Measure inference time
    int64_t start_time = esp_timer_get_time();

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed\n");
        return;
    }

    int64_t end_time = esp_timer_get_time();
    int64_t inference_time = end_time - start_time;

    // Print inference time
    MicroPrintf("Inference time: %lld microseconds", inference_time);

    // Obtain the raw output from model's output tensor
    MicroPrintf("Raw output: %d", output->data.uint8[0]);

    // Real value prediction
    double prediction = (output->data.uint8[0] - output->params.zero_point) * output->params.scale;

    // Output the results
    MicroPrintf("Prediction: %f", prediction);

    int predicted_class = prediction > 0.5 ? 1 : 0;

    MicroPrintf("Predicted class: %d", predicted_class);

    free(image_data);
}

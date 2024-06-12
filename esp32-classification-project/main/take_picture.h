#ifndef TAKE_PICTURE_H_
#define TAKE_PICTURE_H_

#include "esp_err.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t init_camera(void);
esp_err_t grayscale_image(uint8_t *image_array);

#ifdef __cplusplus
}
#endif

#endif  // TAKE_PICTURE_H_

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "main_functions.h"

extern "C" void app_main(void) {
  setup();
  while (true) {
    loop();

    // trigger one inference every 1 seconds
    vTaskDelay(pdMS_TO_TICKS(5000));
  }
}

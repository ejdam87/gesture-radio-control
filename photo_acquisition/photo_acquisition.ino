#include "esp_camera.h"

// --- Camera module pin numbers
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define FLASHLIGHT         4
// ---


// Capture period (ms)
#define CAPTURE_INTERVAL 5000

// time of last photo capture (ms)
unsigned long last_capture = 0;


void setup()
{

    // setup boud rate
    Serial.begin(115200);

    // --- config the camera
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    if (psramFound())
    {
        config.frame_size = FRAMESIZE_UXGA;
        config.jpeg_quality = 10;
        config.fb_count = 2;
    }
    else
    {
        config.frame_size = FRAMESIZE_SVGA;
        config.jpeg_quality = 12;
        config.fb_count = 1;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK)
    {
        Serial.println("Camera init failed");
    }
    else
    {
        Serial.println("Camera init succeeded");
    }
    // ---
}


camera_fb_t* capture()
{
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb)
    {
        Serial.println("Camera capture failed");
        return NULL;
    }
    return fb;
}

void send_photo(camera_fb_t* fb)
{
    if (fb == NULL)
    {
        Serial.println("Invalid frame");
    }

    Serial.println("PHOTO_START");  
    // Serial.write(fb->buf, fb->len);  
    Serial.println("PHOTO_END");
    esp_camera_fb_return(fb);
}

void loop()
{
    unsigned long current_time = millis();
    if (current_time - last_capture >= CAPTURE_INTERVAL)
    {
        last_capture = current_time;
        auto* fb = capture();
        send_photo(fb);
    }
}

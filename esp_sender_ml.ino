/*
  esp_sender_ml.ino
  ------------------
  Reads DHT11 + MQ-2, sends JSON to Python via Serial.
  Listens for alert digit from Python, forwards via ESP-NOW.

  WIRING:
    DHT11 VCC  → 3.3V
    DHT11 GND  → GND
    DHT11 DATA → D4

    MQ-2  VCC  → Vin (5V)
    MQ-2  GND  → GND
    MQ-2  AO   → A0

  Libraries: DHT sensor library (Adafruit), ArduinoJson
*/

#include <ESP8266WiFi.h>
#include <espnow.h>
#include <DHT.h>
#include <ArduinoJson.h>

// ── Pins ──────────────────────────────────────────
#define DHTPIN   D4       // back to D4 — this was working before
#define DHTTYPE  DHT11
#define MQ2_PIN  A0

DHT dht(DHTPIN, DHTTYPE);

// ── Receiver MAC ──────────────────────────────────
uint8_t receiverMac[] = {0x68, 0xC6, 0x3A, 0xDE, 0x87, 0xAE};

typedef struct struct_message {
  int  alertType;
  char message[50];
} struct_message;

struct_message myData;
bool peerAdded = false;

unsigned long lastSensorRead = 0;
const unsigned long SENSOR_INTERVAL = 3000;

// ─────────────────────────────────────────────────
void OnDataSent(uint8_t *mac_addr, uint8_t sendStatus) {
  Serial.println(sendStatus == 0 ? "ESP_NOW:SUCCESS" : "ESP_NOW:FAIL");
}

void forwardAlert(int alertType) {
  if (alertType < 0 || alertType > 5) return;
  myData.alertType = alertType;
  switch(alertType) {
    case 0: strcpy(myData.message, "ALL GOOD");   break;
    case 1: strcpy(myData.message, "MEDICAL");    break;
    case 2: strcpy(myData.message, "FIRE");       break;
    case 3: strcpy(myData.message, "FLOOD");      break;
    case 4: strcpy(myData.message, "EARTHQUAKE"); break;
    case 5: strcpy(myData.message, "GAS LEAK");   break;
  }
  esp_now_send(receiverMac, (uint8_t*)&myData, sizeof(myData));
  Serial.print("FORWARD:"); Serial.println(alertType);
}

// ─────────────────────────────────────────────────
void readAndSendSensors() {
  // Simple single read — no retry loop, no blocking delay
  // This is what worked before
  float temperature = dht.readTemperature();
  float humidity    = dht.readHumidity();
  int   mq2_raw     = analogRead(MQ2_PIN);

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("SENSOR_ERR:DHT_FAIL");
    temperature = -1;
    humidity    = -1;
  }

  // Send JSON
  StaticJsonDocument<128> doc;
  doc["T"]  = temperature;
  doc["H"]  = humidity;
  doc["MQ"] = mq2_raw;
  serializeJson(doc, Serial);
  Serial.println();

  // Debug line
  Serial.print("DBG:T="); Serial.print(temperature);
  Serial.print(" H=");     Serial.print(humidity);
  Serial.print(" MQ=");    Serial.println(mq2_raw);
}

// ─────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);

  dht.begin();
  pinMode(MQ2_PIN, INPUT);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  Serial.print("ESP_MAC:"); Serial.println(WiFi.macAddress());

  if (esp_now_init() != 0) {
    Serial.println("ESP_ERR:ESPNOW_INIT"); return;
  }
  esp_now_set_self_role(ESP_NOW_ROLE_CONTROLLER);
  esp_now_register_send_cb(OnDataSent);

  if (esp_now_add_peer(receiverMac, ESP_NOW_ROLE_SLAVE, 1, NULL, 0) == 0) {
    peerAdded = true;
    Serial.println("ESP_READY");
  } else {
    Serial.println("ESP_ERR:PEER_FAILED");
  }
  Serial.println("SENSOR_MODE:START");
}

// ─────────────────────────────────────────────────
void loop() {
  if (millis() - lastSensorRead >= SENSOR_INTERVAL) {
    readAndSendSensors();
    lastSensorRead = millis();
  }

  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      int alertType = input.toInt();
      if (alertType >= 0 && alertType <= 5) forwardAlert(alertType);
    }
  }
}

/*
  receiver.ino — FIXED VERSION
  ─────────────────────────────
  FIXES:
    1. Handles alertType 0 → shows "ALL GOOD / Air is clean"
    2. volatile bool for newAlert
    3. No delay() in callback

  Wiring (unchanged):
    LCD SDA → D2 (GPIO4)
    LCD SCL → D1 (GPIO5)
*/

#include <ESP8266WiFi.h>
#include <espnow.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

typedef struct struct_message {
  int  alertType;
  char message[50];
} struct_message;

struct_message myData;

volatile bool newAlert    = false;
int           lastAlertType = -1;
char          lastMessage[50] = "";

// ─────────────────────────────────────────────────
void OnDataRecv(uint8_t *mac, uint8_t *incomingData, uint8_t len) {
  memcpy(&myData, incomingData, sizeof(myData));
  lastAlertType = myData.alertType;
  strcpy(lastMessage, myData.message);
  newAlert = true;
  Serial.print("RECV:"); Serial.println(myData.alertType);
}

// ─────────────────────────────────────────────────
void blinkLED(int times, int onMs, int offMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(2, LOW);  delay(onMs);
    digitalWrite(2, HIGH); delay(offMs);
  }
}

// ─────────────────────────────────────────────────
void updateLCD() {
  lcd.clear();
  lcd.setCursor(0, 0);

  switch (lastAlertType) {
    case 0:
      lcd.print("ALL GOOD");
      lcd.setCursor(0, 1);
      lcd.print("Air is clean");
      blinkLED(1, 100, 0);   // single short blink
      break;

    case 1:
      lcd.print("MEDICAL ALERT");
      lcd.setCursor(0, 1);
      lcd.print("CALL 108 NOW");
      blinkLED(3, 500, 500); // slow blink x3
      break;

    case 2:
      lcd.print("FIRE DETECTED!");
      lcd.setCursor(0, 1);
      lcd.print("EVACUATE NOW!!");
      blinkLED(10, 100, 100); // fast blink x10
      break;

    case 3:
      lcd.print("FLOOD WARNING");
      lcd.setCursor(0, 1);
      lcd.print("GO TO HIGHER GND");
      blinkLED(5, 300, 200);
      break;

    case 4:
      lcd.print("EARTHQUAKE!");
      lcd.setCursor(0, 1);
      lcd.print("DROP AND COVER");
      blinkLED(8, 150, 150);
      break;

    case 5:
      lcd.print("GAS LEAK!");
      lcd.setCursor(0, 1);
      lcd.print("OPEN WINDOWS!");
      blinkLED(6, 200, 200);
      break;

    default:
      lcd.print("UNKNOWN ALERT");
      lcd.setCursor(0, 1);
      lcd.print("Stay alert");
      break;
  }
}

// ─────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(2, OUTPUT);
  digitalWrite(2, HIGH);  // LED off (active LOW)

  Wire.begin(4, 5);       // SDA=GPIO4(D2), SCL=GPIO5(D1)
  lcd.begin();
  lcd.backlight();
  lcd.clear();
  lcd.print("RECEIVER READY");
  delay(1500);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  if (esp_now_init() != 0) {
    lcd.clear(); lcd.print("ESP-NOW FAIL");
    return;
  }

  esp_now_set_self_role(ESP_NOW_ROLE_SLAVE);
  esp_now_register_recv_cb(OnDataRecv);

  lcd.clear();
  lcd.print("ALL GOOD");
  lcd.setCursor(0, 1);
  lcd.print("Monitoring...");

  Serial.print("MAC: "); Serial.println(WiFi.macAddress());
  Serial.println("READY");
}

// ─────────────────────────────────────────────────
void loop() {
  if (newAlert) {
    newAlert = false;
    updateLCD();
  }
  delay(50);
}

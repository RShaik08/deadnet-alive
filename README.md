# IoT Disaster Alert System using ESP-NOW & Machine Learning

[![Platform](https://img.shields.io/badge/platform-ESP8266-blue)](https://www.espressif.com/en/products/socs/esp8266)
[![ML Framework](https://img.shields.io/badge/ML-Scikit--Learn-orange)](https://scikit-learn.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-red)](LICENSE)

> A real-time disaster alert system that combines IoT sensors, ESP-NOW wireless communication, and Machine Learning classification to detect emergencies without requiring internet or cellular infrastructure.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Hardware Requirements](#-hardware-requirements)
- [Software Requirements](#-software-requirements)
- [Wiring Instructions](#-wiring-instructions)
- [Installation](#-installation)
- [Usage](#-usage)
- [ML Models](#-ml-models)
- [Web Dashboard](#-web-dashboard)
- [Troubleshooting](#-troubleshooting)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🎯 Overview

This system detects environmental hazards using low-cost sensors and classifies them using Machine Learning models. It operates entirely without WiFi or internet - making it ideal for disaster zones where communication infrastructure has failed.

### Key Features

| Feature | Description |
| :--- | :--- |
| **🌡️ Dual Sensor Input** | DHT11 (temperature/humidity) + MQ-2 (gas/smoke) |
| **🔗 ESP-NOW Protocol** | Direct device-to-device communication (no WiFi needed) |
| **🤖 Dual ML Models** | Random Forest classifiers for Fire + Air Quality |
| **📟 LCD Display** | Real-time alert visualization on 16x2 I2C LCD |
| **💡 LED Indicators** | Visual feedback with blink patterns for each alert type |
| **🌐 Web Dashboard** | Flask-based real-time monitoring interface |
| **📊 Alert History** | Timestamped logging of all detected events |

### Alert Types

| Alert | Priority | Confidence Threshold | LCD Message | LED Pattern |
| :--- | :--- | :--- | :--- | :--- |
| **NORMAL** | 0 | N/A | "ALL GOOD" | 1 short blink |
| **MEDICAL** | 2 (Urgent) | 85% | "CALL 108 NOW" | 3 slow blinks |
| **FIRE** | 1 (Critical) | 90% | "EVACUATE NOW!!" | 10 rapid blinks |
| **GAS LEAK** | 1 (Critical) | 70% | "OPEN WINDOWS!" | 6 medium blinks |

---


---

## 🔧 Hardware Requirements

| Component | Quantity | Approx Cost | Purpose |
| :--- | :--- | :--- | :--- |
| [ESP8266 NodeMCU](https://www.amazon.in/s?k=esp8266+nodemcu) | 2 | ₹400-600 ea | Microcontroller nodes |
| [DHT11 Sensor](https://www.amazon.in/s?k=dht11+sensor) | 1 | ₹150-250 | Temperature & Humidity |
| [MQ-2 Gas Sensor](https://www.amazon.in/s?k=mq-2+sensor) | 1 | ₹200-350 | Smoke & Gas detection |
| [LCD 16x2 I2C](https://www.amazon.in/s?k=i2c+lcd+16x2) | 1 | ₹250-400 | Alert display |
| Jumper Wires (Male-Female) | 7-10 | ₹50-100 | Connections |
| USB Data Cables | 2 | ₹100-200 ea | Programming & power |
| Breadboard (400 points) | 1 | ₹100-200 | Optional - for prototyping |

**Total Estimated Cost: ₹1,500 - ₹2,500 (~$18-30 USD)**

### Sensor Specifications

| Parameter | DHT11 | MQ-2 |
| :--- | :--- | :--- |
| Operating Voltage | 3.3-5V | 5V |
| Current Draw | 0.5-2.5mA | 150-180mA |
| Warm-up Time | 1s | 120-180s |
| Response Time | <2s | <10s |
| Measurement Range | 0-50°C, 20-90% RH | 300-10000ppm (LPG/Propane) |

---

## 🔌 Wiring Instructions

### Sender ESP8266 (with Sensors)

| DHT11 Pin | → | ESP8266 Pin | Label |
| :--- | :--- | :--- | :--- |
| VCC | → | 3.3V | Power |
| GND | → | GND | Ground |
| DATA | → | D4 (GPIO2) | Data |

| MQ-2 Pin | → | ESP8266 Pin | Label |
| :--- | :--- | :--- | :--- |
| VCC | → | VIN (5V) | Power |
| GND | → | GND | Ground |
| A0 | → | A0 | Analog Input |

> **⚠️ Important:** MQ-2 requires 5V power (VIN pin) for the heating element. 3.3V will not work properly.

### Receiver ESP8266 (with LCD)

| LCD I2C Pin | → | ESP8266 Pin | Label |
| :--- | :--- | :--- | :--- |
| VCC | → | VIN (5V) | Power |
| GND | → | GND | Ground |
| SDA | → | D2 (GPIO4) | Data |
| SCL | → | D1 (GPIO5) | Clock |


---

## 💻 Software Requirements

### Arduino IDE (for ESP8266 programming)
- [Arduino IDE 2.x](https://www.arduino.cc/en/software) or 1.8.x
- ESP8266 board package (install via Boards Manager)
- Required libraries (install via Library Manager):

| Library | Purpose | Install Command |
| :--- | :--- | :--- |
| DHT sensor library | Read DHT11 | `Tools → Manage Libraries → search "DHT"` |
| ArduinoJson | JSON serialization | `Tools → Manage Libraries → search "ArduinoJson"` |
| LiquidCrystal I2C | LCD control | `Tools → Manage Libraries → search "LiquidCrystal I2C"` |
| ESP8266WiFi | WiFi/ESP-NOW | Included with ESP8266 board package |

### Python (for ML backend)
- Python 3.8 or higher
- Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn pyserial flask

### Required Datasets (for training)
Download these Kaggle datasets (free account required):

Smoke Detection Dataset → rename to smoke_detection.csv

Air Quality Dataset → rename to pollution_dataset.csv

Place both files in the same folder as the Python scripts.

📦 Installation
Step 1: Clone the Repository
bash
git clone https://github.com/yourusername/esp-iot-disaster-alert-system.git
cd esp-iot-disaster-alert-system
Step 2: Install Arduino Libraries
Open Arduino IDE

Go to Tools → Manage Libraries

Install:

DHT sensor library by Adafruit

ArduinoJson by Benoit Blanchon

LiquidCrystal I2C by Frank de Brabander

Step 3: Add ESP8266 Board Package
Go to File → Preferences

Add to "Additional Boards Manager URLs":

text
http://arduino.esp8266.com/stable/package_esp8266com_index.json
Go to Tools → Board → Boards Manager

Search esp8266 and install ESP8266 by ESP8266 Community

Step 4: Upload ESP8266 Firmware
For Sender (with sensors):
Open esp_sender_ml.ino

Update receiverMac[] with your Receiver's MAC address

Select NodeMCU 1.0 (ESP-12E Module) as board

Select correct COM port

Click Upload

For Receiver (with LCD):
Open esp_receiver_ml.ino

Select NodeMCU 1.0 (ESP-12E Module) as board

Select correct COM port

Click Upload

Step 5: Train ML Models
bash
python train_model.py
This will generate:

fire_model.pkl, fire_scaler.pkl, fire_metadata.pkl

aq_model.pkl, aq_scaler.pkl, aq_metadata.pkl

Confusion matrix plots and feature importance graphs

Step 6: Run the Alert System
bash
python predict_and_send.py
🚀 Usage
Starting the System
Power both ESP8266 boards (via USB or power bank)

Run the Python script:

bash
python predict_and_send.py
Select operation mode when prompted:

text
═══════════════════════════════════════════════════════
  1 → Manual Mode (type feature values manually)
  2 → Auto Mode (random dataset rows, every 6 seconds)
  3 → Sensor Mode ⭐ (LIVE DHT11 + MQ-2)
  q → Quit
═══════════════════════════════════════════════════════
Testing with Sensor Mode
Type 3 and press Enter

Wait for MQ-2 to preheat (2-3 minutes)

Observe sensor readings in console:

text
  ── Reading #1 ──────────────────────────
  Temp: 28.5°C  |  Humidity: 62.0%
  MQ-2: 102/1023 (10.0%) [██░░░░░░░░░░░░░░░░░░]
  MQ-2 level: CLEAN AIR

  Fire Model → ✅ No Fire (12.3%)
  AQ Model   → NORMAL (8.7%)

  🤖 [14:32:15] → 0 (NORMAL) | 12.3% | NORMAL | No alert
Triggering Alerts
Action	Expected Alert
Burn a match near MQ-2	GAS LEAK or FIRE
Blow gently on MQ-2	GAS LEAK (from breath CO₂)
Warm sensor with hand	MAY trigger MEDICAL
Clean air (5+ minutes)	NORMAL
Web Dashboard
While the Python script runs, open a browser and go to:

text
http://localhost:5000
The dashboard shows:

Live temperature, humidity, MQ-2 readings

Current alert with severity and confidence

MQ-2 sensor level visualization bar

All 15+ derived model features

Alert history with timestamps

🤖 ML Models
Fire Detection Model
Property	Value
Algorithm	Random Forest Classifier
Training Dataset	Smoke Detection (42,000 samples)
Features	15 (TVOC, eCO2, PM2.5, Raw H2, etc.)
Accuracy	98.5%
Precision	96.2%
Recall	97.1%
F1-Score	96.6%
Top 5 Important Features:

TVOC (25%)

eCO2 (18%)

Raw H2 (15%)

Temperature (12%)

PM2.5 (8%)

Air Quality Model
Property	Value
Algorithm	Random Forest Classifier
Training Dataset	Air Quality (15,000 samples)
Features	9 (PM2.5, PM10, NO2, SO2, CO, etc.)
Accuracy	92.3%
Classes	Good, Moderate, Poor, Hazardous
Feature Mapping (Sensor → Model Input)
Sensor Reading	Derived Features
MQ-2 Raw (0-1023)	TVOC, eCO2, Raw H2, Raw Ethanol, PM1.0, PM2.5, PM10, NC0.5, NC1.0, NC2.5, CO, NO2, SO2
DHT11 Temperature	Temperature[C]
DHT11 Humidity	Humidity[%]
Fixed (not measured)	Pressure, Proximity_to_Industrial_Areas, Population_Density
🌐 Web Dashboard
Dashboard Components
Component	Description
Current Alert Status	Large display with color-coded alert type
Live Sensor Values	Real-time temperature, humidity, MQ-2 raw & percentage
MQ-2 Level Bar	Visual indicator of smoke/gas concentration
Model Input Features	All 15+ derived values used by ML models
Alert History	Table of last 20 alerts with timestamps
System Status	Current mode and operation state
Color Coding
Severity	Color	Meaning
NORMAL	🟢 Green	Safe conditions
CAUTION	🔵 Blue	Monitor situation
WARNING	🟠 Orange	Take precautions
CRITICAL	🔴 Red	Immediate action required



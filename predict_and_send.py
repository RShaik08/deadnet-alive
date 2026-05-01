"""
predict_and_send.py  — FIXED VERSION
──────────────────────────────────────
FIXES:
  1. mq2_to_tvoc  — now scales to 60000 properly (was capped at 6000)
  2. mq2_to_eco2  — now scales to 65000 (was capped at 6000)
  3. mq2_to_raw_h2 — fixed direction (decreases with smoke, was increasing)
  4. mq2_to_raw_ethanol — fixed scaling range
  5. Added MQ2 percentage printout so you can see sensor responding
  6. Sends 0 to ESP for no-alert → LCD shows ALL GOOD
"""

import pickle, serial, time, threading, json
import pandas as pd, numpy as np
import os, sys, random
from datetime import datetime
from flask import Flask, jsonify, render_template_string

# ══════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════
SERIAL_PORT   = "COM4"
BAUD_RATE     = 115200
AUTO_INTERVAL = 6
WEB_PORT      = 5000

FIRE_CONFIDENCE_THRESHOLD     = 90
GAS_LEAK_CONFIDENCE_THRESHOLD = 70
MEDICAL_CONFIDENCE_THRESHOLD  = 85

ALERT_NAMES = {0:"NORMAL", 1:"MEDICAL", 2:"FIRE", 5:"GAS LEAK"}

# ══════════════════════════════════════════════════
# SHARED STATE
# ══════════════════════════════════════════════════
state = {
    "latest": {
        "alert_type":0,"alert_name":"NORMAL","confidence":100.0,
        "severity":"NORMAL","source":"—","model_used":"—",
        "timestamp":"—","fire_readings":{},"aq_readings":{}
    },
    "history":[], "total_sent":0, "system_status":"Starting..."
}
state_lock = threading.Lock()

# ══════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════
def load_models():
    files = ['fire_model.pkl','fire_scaler.pkl','fire_metadata.pkl',
             'aq_model.pkl','aq_scaler.pkl','aq_metadata.pkl']
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing: {missing}\n   Run train_model.py first!")
        sys.exit(1)

    with open('fire_model.pkl','rb')    as f: fire_model  = pickle.load(f)
    with open('fire_scaler.pkl','rb')   as f: fire_scaler = pickle.load(f)
    with open('fire_metadata.pkl','rb') as f: fire_meta   = pickle.load(f)
    with open('aq_model.pkl','rb')      as f: aq_model    = pickle.load(f)
    with open('aq_scaler.pkl','rb')     as f: aq_scaler   = pickle.load(f)
    with open('aq_metadata.pkl','rb')   as f: aq_meta     = pickle.load(f)

    print("✅ Both models loaded")
    print(f"   Fire accuracy : {fire_meta['accuracy']*100:.1f}%")
    print(f"   AQ accuracy   : {aq_meta['accuracy']*100:.1f}%")
    print(f"   Fire features : {fire_meta['feature_cols']}")
    print(f"   AQ features   : {aq_meta['feature_cols']}")
    return fire_model, fire_scaler, fire_meta, aq_model, aq_scaler, aq_meta

# ══════════════════════════════════════════════════
# LOAD DATASETS
# ══════════════════════════════════════════════════
def load_datasets(fire_meta, aq_meta):
    df_fire = df_aq = None
    for name in ['smoke_detection.csv','smoke.csv','fire_detection.csv']:
        if os.path.exists(name):
            df = pd.read_csv(name)
            cols = [c for c in fire_meta['feature_cols'] if c in df.columns]
            df_fire = df[cols].dropna()
            print(f"✅ Fire dataset: {len(df_fire)} rows")
            break
    for name in ['pollution_dataset.csv','air_quality.csv','dataset.csv']:
        if os.path.exists(name):
            df = pd.read_csv(name)
            cols = [c for c in aq_meta['feature_cols'] if c in df.columns]
            df_aq = df[cols].dropna()
            print(f"✅ AQ dataset: {len(df_aq)} rows")
            break
    return df_fire, df_aq

# ══════════════════════════════════════════════════
# SERIAL
# ══════════════════════════════════════════════════
def connect_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)
        print(f"✅ Connected to {SERIAL_PORT}")
        return ser
    except serial.SerialException as e:
        print(f"❌ Cannot open {SERIAL_PORT}: {e}")
        print("   Close Arduino Serial Monitor first!")
        sys.exit(1)

# ══════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════
def predict_fire(fire_model, fire_scaler, fire_meta, values):
    arr_s = fire_scaler.transform(np.array(values).reshape(1,-1))
    pred  = int(fire_model.predict(arr_s)[0])
    conf  = max(fire_model.predict_proba(arr_s)[0]) * 100
    return pred, conf

def predict_aq(aq_model, aq_scaler, aq_meta, values):
    arr_s = aq_scaler.transform(np.array(values).reshape(1,-1))
    pred  = int(aq_model.predict(arr_s)[0])
    conf  = max(aq_model.predict_proba(arr_s)[0]) * 100
    return pred, conf

def get_severity(alert_type, confidence):
    if alert_type == 0: return "NORMAL"
    if confidence >= 90: return "CRITICAL"
    if confidence >= 70: return "WARNING"
    return "CAUTION"

def decide_alert(fire_pred, fire_conf, aq_pred, aq_conf):
    if fire_pred == 1 and fire_conf >= FIRE_CONFIDENCE_THRESHOLD:
        return 2, fire_conf, "Fire Detection Model"
    if aq_pred == 5 and aq_conf >= GAS_LEAK_CONFIDENCE_THRESHOLD:
        return 5, aq_conf, "Air Quality Model"
    if aq_pred == 1 and aq_conf >= MEDICAL_CONFIDENCE_THRESHOLD:
        return 1, aq_conf, "Air Quality Model"
    return 0, max(fire_conf, aq_conf, 50), "No alert"

# ══════════════════════════════════════════════════
# MQ2 CONVERSION FUNCTIONS — FIXED SCALING
# ══════════════════════════════════════════════════

def mq2_percent(raw):
    """How saturated is the MQ-2? 0% = clean air, 100% = max reading"""
    return round((raw / 1023.0) * 100, 1)

def mq2_to_ppm(raw):
    """Rough smoke PPM estimate"""
    voltage = raw * (3.3 / 1023.0)
    if voltage < 0.01: voltage = 0.01
    rs_ro = max((3.3 - voltage) / voltage, 0.01)
    return max(0, round(613.9 * pow(rs_ro, -2.074), 1))

# ── FIXED: these now scale to the ranges the fire model was trained on ──

def mq2_to_tvoc(raw):
    # Fire dataset TVOC range: 0 (clean) to ~60000 (heavy fire)
    # Use power curve so mid-range values map realistically
    ratio = raw / 1023.0
    return round(ratio ** 1.5 * 60000, 1)

def mq2_to_eco2(raw):
    # Fire dataset eCO2 range: ~400 (clean) to ~65000 (fire)
    ratio = raw / 1023.0
    return round(400 + ratio ** 1.5 * 64600, 1)

def mq2_to_raw_h2(raw):
    # FIXED: Raw H2 DECREASES as combustion gases displace H2
    # Clean air: ~13000-14000, Fire: ~10000-11000
    ratio = raw / 1023.0
    return round(14000 - ratio * 3000, 1)

def mq2_to_raw_ethanol(raw):
    # FIXED: Ethanol sensor reading
    # Clean: ~18000-19000, Fire: ~20000-30000 (increases with combustion)
    ratio = raw / 1023.0
    return round(18520 + ratio ** 1.2 * 13000, 1)

def mq2_to_pm1(raw):
    ratio = raw / 1023.0
    return round(ratio ** 1.5 * 150, 2)

def mq2_to_pm25(raw):
    ratio = raw / 1023.0
    return round(ratio ** 1.5 * 300, 2)

def mq2_to_pm10(raw):
    ratio = raw / 1023.0
    return round(ratio ** 1.5 * 400, 2)

def mq2_to_nc05(raw):
    ratio = raw / 1023.0
    return round(ratio ** 1.5 * 5000, 1)

def mq2_to_nc10(raw):
    ratio = raw / 1023.0
    return round(ratio ** 1.5 * 2000, 1)

def mq2_to_nc25(raw):
    ratio = raw / 1023.0
    return round(ratio ** 1.5 * 500, 1)

def mq2_to_co(raw):
    # CO range: 0 (clean) to 10 mg/m³ (hazardous)
    ratio = raw / 1023.0
    return round(ratio ** 1.2 * 10, 3)

def mq2_to_no2(raw):
    ratio = raw / 1023.0
    return round(ratio * 200, 2)

def mq2_to_so2(raw):
    ratio = raw / 1023.0
    return round(ratio * 100, 2)

DEFAULT_PRESSURE   = 939.0
DEFAULT_PROXIMITY  = 10.0
DEFAULT_POPULATION = 500.0

# ══════════════════════════════════════════════════
# FEATURE BUILDERS
# ══════════════════════════════════════════════════
def build_fire_features(temp, humidity, mq2_raw, feature_cols):
    feature_map = {}
    for col in feature_cols:
        cl = col.lower().replace(' ','').replace('_','').replace('[','').replace(']','')
        if 'temp' in cl:
            feature_map[col] = temp
        elif 'humid' in cl:
            feature_map[col] = humidity
        elif 'tvoc' in cl:
            feature_map[col] = mq2_to_tvoc(mq2_raw)
        elif 'eco2' in cl or ('co2' in cl and 'e' in cl):
            feature_map[col] = mq2_to_eco2(mq2_raw)
        elif 'rawh2' in cl or ('raw' in cl and 'h2' in cl):
            feature_map[col] = mq2_to_raw_h2(mq2_raw)
        elif 'rawethanol' in cl or ('raw' in cl and 'ethanol' in cl):
            feature_map[col] = mq2_to_raw_ethanol(mq2_raw)
        elif 'pressure' in cl:
            feature_map[col] = DEFAULT_PRESSURE
        elif 'pm1' in cl and '0' in cl and '2' not in cl:
            feature_map[col] = mq2_to_pm1(mq2_raw)
        elif 'pm2' in cl or 'pm25' in cl:
            feature_map[col] = mq2_to_pm25(mq2_raw)
        elif 'pm10' in cl:
            feature_map[col] = mq2_to_pm10(mq2_raw)
        elif 'nc0' in cl or 'nc05' in cl:
            feature_map[col] = mq2_to_nc05(mq2_raw)
        elif 'nc1' in cl:
            feature_map[col] = mq2_to_nc10(mq2_raw)
        elif 'nc2' in cl:
            feature_map[col] = mq2_to_nc25(mq2_raw)
        else:
            feature_map[col] = 0.0
    return [feature_map[c] for c in feature_cols], feature_map

def build_aq_features(temp, humidity, mq2_raw, feature_cols):
    feature_map = {}
    for col in feature_cols:
        cl = col.lower().replace('_','').replace(' ','')
        if 'temp' in cl:
            feature_map[col] = temp
        elif 'humid' in cl:
            feature_map[col] = humidity
        elif 'pm2' in cl:
            feature_map[col] = mq2_to_pm25(mq2_raw)
        elif 'pm10' in cl:
            feature_map[col] = mq2_to_pm10(mq2_raw)
        elif 'no2' in cl:
            feature_map[col] = mq2_to_no2(mq2_raw)
        elif 'so2' in cl:
            feature_map[col] = mq2_to_so2(mq2_raw)
        elif cl == 'co' or col.upper() == 'CO':
            feature_map[col] = mq2_to_co(mq2_raw)
        elif 'proximity' in cl or 'industrial' in cl:
            feature_map[col] = DEFAULT_PROXIMITY
        elif 'population' in cl or 'density' in cl:
            feature_map[col] = DEFAULT_POPULATION
        else:
            feature_map[col] = 0.0
    return [feature_map[c] for c in feature_cols], feature_map

# ══════════════════════════════════════════════════
# SEND ALERT
# ══════════════════════════════════════════════════
def send_alert(ser, alert_type, confidence, source, fire_readings, aq_readings, model_used):
    ts       = datetime.now().strftime("%H:%M:%S")
    severity = get_severity(alert_type, confidence)
    entry = {
        "timestamp": ts, "alert_type": alert_type,
        "alert_name": ALERT_NAMES.get(alert_type, "?"),
        "confidence": round(confidence, 1), "severity": severity,
        "source": source, "model_used": model_used,
        "fire_readings": fire_readings, "aq_readings": aq_readings
    }
    with state_lock:
        state["latest"] = entry
        state["history"].insert(0, entry)
        if len(state["history"]) > 20: state["history"].pop()
        if alert_type != 0: state["total_sent"] += 1

    name = ALERT_NAMES.get(alert_type, '?')
    print(f"\n  🤖 [{ts}] → {alert_type} ({name}) | {confidence:.1f}% | {severity} | {model_used}")

    try:
        ser.write((str(alert_type) + '\n').encode())
        ser.flush()
        time.sleep(0.3)
        while ser.in_waiting:
            line = ser.readline().decode(errors='ignore').strip()
            if line and not line.startswith('{') and not line.startswith('DBG'):
                print(f"  ESP: {line}")
    except Exception as e:
        print(f"  ⚠️  Serial error: {e}")

# ══════════════════════════════════════════════════
# SENSOR MODE
# ══════════════════════════════════════════════════
def sensor_mode(fire_model, fire_scaler, fire_meta,
                aq_model, aq_scaler, aq_meta, ser):

    print("\n" + "="*55)
    print("  SENSOR MODE — LIVE DHT11 + MQ-2")
    print("  Ctrl+C to stop")
    print("="*55)
    print("\n  Waiting for ESP8266 sensor data...\n")
    print("  To trigger FIRE:     light a match or lighter near MQ-2")
    print("  To trigger GAS LEAK: blow into MQ-2 from mouth closely")
    print("  To trigger MEDICAL:  warm up the MQ-2 slightly with hand")
    print("  Normal air:          LCD shows ALL GOOD\n")

    ser.flushInput()
    count = 0

    with state_lock:
        state["system_status"] = "Sensor mode active"

    try:
        while True:
            try:
                raw_line = ser.readline().decode(errors='ignore').strip()
            except Exception:
                time.sleep(1); continue

            if not raw_line: continue

            # Print non-JSON debug lines
            if not raw_line.startswith('{'):
                if any(raw_line.startswith(p) for p in ['ESP_','SENSOR_','DBG:','FORWARD:','ESP_NOW:']):
                    print(f"  [ESP] {raw_line}")
                continue

            # Parse JSON
            try:
                data     = json.loads(raw_line)
                temp     = float(data['T'])
                humidity = float(data['H'])
                mq2_raw  = int(data['MQ'])
            except Exception as e:
                print(f"  ⚠️  Parse error: {e} — line: {raw_line}")
                continue

            # DHT11 fallback — use ambient room defaults if sensor fails
            # MQ-2 still works perfectly, temp/humidity have minor effect on prediction
            DHT_FAILED = (temp == -1 or humidity == -1)
            if DHT_FAILED:
                temp     = 28.0   # typical indoor room temperature
                humidity = 60.0   # typical indoor humidity
                print(f"  ⚠️  DHT11 unavailable — using room defaults (28°C, 60%)")
                print(f"      MQ-2 is still live: {mq2_raw} ({mq2_percent(mq2_raw)}%)")

            count += 1
            mq_pct = mq2_percent(mq2_raw)

            print(f"\n  ── Reading #{count} ──────────────────────────")
            dht_tag = " (room default)" if DHT_FAILED else " (DHT11 live)"
            print(f"  Temp: {temp}°C  |  Humidity: {humidity}%{dht_tag}")
            print(f"  MQ-2: {mq2_raw}/1023 ({mq_pct}%) ", end="")

            # Visual bar for MQ2 level
            bars = int(mq_pct / 5)
            print(f"[{'█'*bars}{'░'*(20-bars)}]")

            if mq_pct < 10:
                print(f"  MQ-2 level: CLEAN AIR")
            elif mq_pct < 30:
                print(f"  MQ-2 level: SLIGHTLY ELEVATED")
            elif mq_pct < 60:
                print(f"  MQ-2 level: HIGH — smoke/gas detected")
            else:
                print(f"  MQ-2 level: VERY HIGH — strong smoke/gas!")

            # Build features
            fire_vals, fire_readings = build_fire_features(
                temp, humidity, mq2_raw, fire_meta['feature_cols'])
            aq_vals, aq_readings = build_aq_features(
                temp, humidity, mq2_raw, aq_meta['feature_cols'])

            # Predict
            fire_pred, fire_conf = predict_fire(fire_model, fire_scaler, fire_meta, fire_vals)
            aq_pred,   aq_conf   = predict_aq(aq_model, aq_scaler, aq_meta, aq_vals)

            print(f"\n  Fire Model → {'⚠️  FIRE' if fire_pred==1 else '✅ No Fire'} ({fire_conf:.1f}%)")
            print(f"  AQ Model   → {ALERT_NAMES.get(aq_pred,'?')} ({aq_conf:.1f}%)")

            # Key derived values
            print(f"  Derived    → TVOC:{mq2_to_tvoc(mq2_raw):.0f}  eCO2:{mq2_to_eco2(mq2_raw):.0f}  CO:{mq2_to_co(mq2_raw):.2f}")

            final_alert, final_conf, model_used = decide_alert(
                fire_pred, fire_conf, aq_pred, aq_conf)

            send_alert(ser, final_alert, final_conf,
                       "DHT11 + MQ-2 sensors",
                       {"Temp(°C)": temp,
                        "Temp source": "room default" if DHT_FAILED else "DHT11 live",
                        "Humidity(%)": humidity,
                        "MQ2_raw": mq2_raw, "MQ2_%": mq_pct,
                        **{k: round(v,2) for k,v in fire_readings.items()
                           if k.lower() not in ['temperature','humidity']}},
                       aq_readings, model_used)

            with state_lock:
                state["system_status"] = (
                    f"Reading #{count} | MQ2:{mq_pct}% | "
                    f"{ALERT_NAMES.get(final_alert,'?')} @ {final_conf:.1f}%")

    except KeyboardInterrupt:
        print("\n\n⏹  Sensor mode stopped.")
        with state_lock: state["system_status"] = "Idle"

# ══════════════════════════════════════════════════
# MANUAL MODE
# ══════════════════════════════════════════════════
def manual_mode(fire_model, fire_scaler, fire_meta,
                aq_model, aq_scaler, aq_meta, ser):
    print("\n" + "="*50)
    print("  MANUAL MODE")
    print("="*50)

    while True:
        print("\n--- Fire Detection Readings ---")
        fire_vals = []; fire_readings = {}; skip = False

        for col in fire_meta['feature_cols']:
            while True:
                raw = input(f"  {col}: ").strip()
                if raw.lower() == 'back': skip = True; break
                try:
                    v = float(raw); fire_vals.append(v); fire_readings[col] = v; break
                except ValueError: print("  Enter a number")
            if skip: break
        if skip: break

        print("\n--- Air Quality Readings ---")
        aq_vals = []; aq_readings = {}; skip = False

        for col in aq_meta['feature_cols']:
            while True:
                raw = input(f"  {col}: ").strip()
                if raw.lower() == 'back': skip = True; break
                try:
                    v = float(raw); aq_vals.append(v); aq_readings[col] = v; break
                except ValueError: print("  Enter a number")
            if skip: break
        if skip: break

        fire_pred, fire_conf = predict_fire(fire_model, fire_scaler, fire_meta, fire_vals)
        aq_pred,   aq_conf   = predict_aq(aq_model, aq_scaler, aq_meta, aq_vals)

        print(f"\n  Fire → {'FIRE' if fire_pred==1 else 'No Fire'} ({fire_conf:.1f}%)")
        print(f"  AQ   → {ALERT_NAMES.get(aq_pred,'?')} ({aq_conf:.1f}%)")

        final_alert, final_conf, model_used = decide_alert(
            fire_pred, fire_conf, aq_pred, aq_conf)
        send_alert(ser, final_alert, final_conf, "Manual",
                   fire_readings, aq_readings, model_used)

        if input("\n  Another? (y/n): ").strip().lower() != 'y': break

# ══════════════════════════════════════════════════
# AUTO MODE
# ══════════════════════════════════════════════════
def auto_mode(fire_model, fire_scaler, fire_meta,
              aq_model, aq_scaler, aq_meta,
              ser, df_fire, df_aq):
    print(f"\n  AUTO MODE — every {AUTO_INTERVAL}s | Ctrl+C to stop")
    count = 0
    try:
        while True:
            count += 1
            if df_fire is not None:
                row = df_fire.sample(1).iloc[0]
                fire_vals = [float(row[c]) for c in fire_meta['feature_cols']]
                fire_readings = {c: round(float(row[c]),2) for c in fire_meta['feature_cols']}
            else:
                fire_vals = [0.0] * len(fire_meta['feature_cols']); fire_readings = {}

            if df_aq is not None:
                row = df_aq.sample(1).iloc[0]
                aq_vals = [float(row[c]) for c in aq_meta['feature_cols']]
                aq_readings = {c: round(float(row[c]),2) for c in aq_meta['feature_cols']}
            else:
                aq_vals = [0.0] * len(aq_meta['feature_cols']); aq_readings = {}

            fire_pred, fire_conf = predict_fire(fire_model, fire_scaler, fire_meta, fire_vals)
            aq_pred,   aq_conf   = predict_aq(aq_model, aq_scaler, aq_meta, aq_vals)
            final_alert, final_conf, model_used = decide_alert(
                fire_pred, fire_conf, aq_pred, aq_conf)
            send_alert(ser, final_alert, final_conf, f"Auto #{count}",
                       fire_readings, aq_readings, model_used)
            print(f"  ⏱  Next in {AUTO_INTERVAL}s... (Ctrl+C to stop)")
            time.sleep(AUTO_INTERVAL)
    except KeyboardInterrupt:
        print("\n⏹  Auto mode stopped.")

# ══════════════════════════════════════════════════
# FLASK DASHBOARD
# ══════════════════════════════════════════════════
app = Flask(__name__)

HTML_PAGE = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ESP8266 ML Alert System</title>
<meta http-equiv="refresh" content="3">
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Courier New',monospace;background:#080c12;color:#c8d8f0;
       min-height:100vh;padding:16px}
  /* ── header ── */
  .header{display:flex;align-items:center;justify-content:space-between;
          padding:14px 20px;background:#0d1420;border:1px solid #1a2540;
          border-top:2px solid #00d4ff;margin-bottom:14px}
  .header h1{color:#00d4ff;font-size:1.2rem;letter-spacing:3px}
  .header-right{display:flex;gap:20px;align-items:center}
  .hdot{display:flex;align-items:center;gap:6px;font-size:0.75rem;color:#4a6080}
  .dot{width:8px;height:8px;border-radius:50%;background:#22c55e;
       box-shadow:0 0 6px #22c55e;animation:pulse 2s infinite}
  .dot.off{background:#ef4444;box-shadow:0 0 6px #ef4444;animation:none}
  @keyframes pulse{50%{opacity:0.4}}
  /* ── main grid ── */
  .grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;width:100%}
  .span2{grid-column:span 2}
  .span3{grid-column:1/-1}
  /* ── cards ── */
  .card{background:#0d1420;border:1px solid #1a2540;padding:18px;
        border-top:2px solid #1a2540}
  .card-title{font-size:0.65rem;letter-spacing:3px;color:#4a6080;margin-bottom:14px}
  /* ── alert display ── */
  .alert-main{text-align:center;padding:10px 0}
  .alert-icon{font-size:3.5rem;line-height:1;margin-bottom:10px}
  .alert-name{font-size:2.8rem;font-weight:bold;letter-spacing:4px;color:#fff;
              text-shadow:0 0 20px currentColor;margin-bottom:8px}
  .alert-action{font-size:0.85rem;color:#8090a8;margin-bottom:12px}
  .badge{display:inline-block;padding:3px 12px;font-size:0.7rem;
         letter-spacing:2px;border:1px solid}
  .sev-NORMAL{color:#22c55e;border-color:#22c55e}
  .sev-CAUTION{color:#00d4ff;border-color:#00d4ff}
  .sev-WARNING{color:#f97316;border-color:#f97316}
  .sev-CRITICAL{color:#ef4444;border-color:#ef4444;
                animation:blink 0.4s infinite}
  @keyframes blink{50%{opacity:0.2}}
  .model-tag{font-size:0.7rem;color:#4a6080;margin-top:6px}
  .conf-wrap{margin:12px 0 4px}
  .conf-bg{background:#1a2540;height:6px;border-radius:3px}
  .conf-fill{height:6px;border-radius:3px;background:#00d4ff;transition:width 0.6s}
  .conf-label{font-size:0.75rem;color:#4a6080;margin-top:4px}
  /* fire */
  .c-fire .card{border-top-color:#ef4444}
  .c-fire .alert-name{color:#ef4444}
  .c-fire .conf-fill{background:#ef4444}
  /* gas */
  .c-gas .card{border-top-color:#a855f7}
  .c-gas .alert-name{color:#a855f7}
  .c-gas .conf-fill{background:#a855f7}
  /* medical */
  .c-med .card{border-top-color:#3b82f6}
  .c-med .alert-name{color:#3b82f6}
  .c-med .conf-fill{background:#3b82f6}
  /* normal */
  .c-ok .card{border-top-color:#22c55e}
  .c-ok .alert-name{color:#22c55e}
  .c-ok .conf-fill{background:#22c55e}
  /* ── sensor boxes ── */
  .sensor-row{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:12px}
  .sbox{background:#080c12;border:1px solid #1a2540;padding:10px;text-align:center}
  .sval{font-size:1.3rem;font-weight:bold;color:#00d4ff}
  .slbl{font-size:0.6rem;color:#4a6080;letter-spacing:1px;margin-top:3px}
  /* mq bar */
  .mq-bg{background:#1a2540;height:10px;border-radius:2px;margin-top:8px}
  .mq-fill{height:10px;border-radius:2px;transition:width 0.5s;
           background:linear-gradient(90deg,#22c55e 0%,#f97316 60%,#ef4444 100%)}
  /* ── stat boxes ── */
  .stat-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px}
  .stat-box{background:#080c12;border:1px solid #1a2540;padding:10px;text-align:center}
  .stat-val{font-size:1.5rem;color:#00d4ff;font-weight:bold}
  .stat-lbl{font-size:0.6rem;color:#4a6080;letter-spacing:1px;margin-top:2px}
  .model-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .model-box{background:#080c12;border:1px solid #1a2540;padding:8px}
  .model-lbl{font-size:0.6rem;color:#4a6080;letter-spacing:1px}
  .model-val{font-size:1rem;font-weight:bold;margin-top:2px}
  .status-txt{font-size:0.7rem;color:#4a6080;margin-top:10px;padding:6px 8px;
              background:#080c12;border:1px solid #1a2540;word-break:break-all}
  /* ── features grid ── */
  .feat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
  .feat-item{background:#080c12;border:1px solid #1a2540;padding:7px}
  .feat-key{font-size:0.6rem;color:#4a6080;letter-spacing:1px}
  .feat-val{font-size:0.85rem;color:#00d4ff;margin-top:2px}
  /* ── table ── */
  table{width:100%;border-collapse:collapse;font-size:0.8rem}
  th{color:#4a6080;padding:7px 8px;border-bottom:1px solid #1a2540;
     text-align:left;font-size:0.65rem;letter-spacing:2px}
  td{padding:6px 8px;border-bottom:1px solid #111824}
  tr:hover td{background:#0d1420}
</style></head>
<body>

<!-- HEADER -->
<div class="header">
  <h1>⚡ ESP8266 ML EMERGENCY ALERT SYSTEM</h1>
  <div class="header-right">
    <div class="hdot"><div class="dot"></div>SERVER ONLINE</div>
    <div class="hdot" style="font-size:0.7rem;color:#4a6080">
      DHT11 + MQ-2 → Random Forest → ESP-NOW → LCD
    </div>
  </div>
</div>

<!-- MAIN GRID -->
<div id="wrapper">
<div class="grid c-ok" id="mainGrid">

  <!-- ALERT CARD (spans 2 cols) -->
  <div class="span2" id="alertWrapper">
    <div class="card">
      <div class="card-title">// CURRENT ALERT STATUS</div>
      <div class="alert-main">
        <div class="alert-icon" id="alertIcon">🟢</div>
        <div class="alert-name" id="alertName">NORMAL</div>
        <div class="alert-action" id="alertAction">All sensors nominal — monitoring environment</div>
        <div>
          <span class="badge sev-NORMAL" id="sevBadge">NORMAL</span>
        </div>
        <div class="model-tag" id="modeBadge">—</div>
        <div class="conf-wrap">
          <div class="conf-bg"><div class="conf-fill" id="confBar" style="width:0%"></div></div>
          <div class="conf-label">Confidence: <span id="confText">—</span> &nbsp;|&nbsp; Last: <span id="lastTime">—</span></div>
        </div>
      </div>
    </div>
  </div>

  <!-- SYSTEM STATS (1 col) -->
  <div>
    <div class="card" style="height:100%">
      <div class="card-title">// SYSTEM</div>
      <div class="stat-row">
        <div class="stat-box">
          <div class="stat-val" id="totalSent">0</div>
          <div class="stat-lbl">ALERTS SENT</div>
        </div>
        <div class="stat-box">
          <div class="stat-val" id="totalReadings">0</div>
          <div class="stat-lbl">READINGS</div>
        </div>
      </div>
      <div class="model-row">
        <div class="model-box">
          <div class="model-lbl">FIRE MODEL</div>
          <div class="model-val" style="color:#ef4444" id="fireAcc">—</div>
        </div>
        <div class="model-box">
          <div class="model-lbl">AQ MODEL</div>
          <div class="model-val" style="color:#22c55e" id="aqAcc">—</div>
        </div>
      </div>
      <div class="status-txt" id="statusBar">Starting...</div>
    </div>
  </div>

  <!-- LIVE SENSORS (full width) -->
  <div class="span3">
    <div class="card">
      <div class="card-title">// LIVE SENSOR VALUES</div>
      <div class="sensor-row">
        <div class="sbox"><div class="sval" id="liveTemp">—</div><div class="slbl">TEMP °C</div></div>
        <div class="sbox"><div class="sval" id="liveHumid">—</div><div class="slbl">HUMIDITY %</div></div>
        <div class="sbox"><div class="sval" id="liveMQ">—</div><div class="slbl">MQ-2 RAW</div></div>
        <div class="sbox"><div class="sval" id="livePct" style="color:#f97316">—</div><div class="slbl">MQ-2 %</div></div>
        <div class="sbox"><div class="sval" id="livePPM">—</div><div class="slbl">EST. PPM</div></div>
      </div>
      <div style="font-size:0.65rem;color:#4a6080;margin-bottom:4px;letter-spacing:2px">MQ-2 SENSOR LEVEL</div>
      <div class="mq-bg"><div class="mq-fill" id="mqBarFill" style="width:0%"></div></div>
    </div>
  </div>

  <!-- MODEL FEATURES (2 cols) -->
  <div class="span2">
    <div class="card">
      <div class="card-title">// MODEL INPUT FEATURES</div>
      <div id="readingsContainer" style="color:#4a6080;font-size:0.8rem">Waiting for sensor data...</div>
    </div>
  </div>

  <!-- ALERT HISTORY (1 col) -->
  <div>
    <div class="card">
      <div class="card-title">// ALERT HISTORY</div>
      <table><thead><tr>
        <th>TIME</th><th>ALERT</th><th>CONF</th><th>SEV</th>
      </tr></thead>
      <tbody id="historyBody">
        <tr><td colspan="4" style="text-align:center;color:#4a6080;padding:20px">No history</td></tr>
      </tbody></table>
    </div>
  </div>

</div>
</div>

<script>
const ICONS  = {0:'🟢',1:'🔵',2:'🔴',5:'🟡'};
const COLORS = {0:'c-ok',1:'c-med',2:'c-fire',5:'c-gas'};
const ACTIONS= {0:'All sensors nominal',1:'Call 108 immediately',
                2:'Evacuate now!',5:'Open windows, evacuate!'};
const SEV_COL= {NORMAL:'#22c55e',CAUTION:'#00d4ff',WARNING:'#f97316',CRITICAL:'#ef4444'};
const PPM_MAP= {'Temp(°C)':'liveTemp','Humidity(%)':'liveHumid',
               'MQ2_raw':'liveMQ','MQ2_%':'livePct',
               'DHT11 Temp (°C)':'liveTemp','DHT11 Humidity (%)':'liveHumid',
               'MQ-2 Raw ADC':'liveMQ','MQ-2 ~PPM (smoke)':'livePPM'};

fetch('/api/models').then(r=>r.json()).then(d=>{
  document.getElementById('fireAcc').textContent=d.fire_acc+'%';
  document.getElementById('aqAcc').textContent=d.aq_acc+'%';
});

function updateUI(data){
  const l=data.latest;
  // Color theme
  const grid=document.getElementById('mainGrid');
  grid.className='grid '+(COLORS[l.alert_type]||'c-ok');
  // Alert panel
  document.getElementById('alertIcon').textContent=ICONS[l.alert_type]||'⚪';
  document.getElementById('alertName').textContent=l.alert_name;
  document.getElementById('alertAction').textContent=ACTIONS[l.alert_type]||'';
  document.getElementById('lastTime').textContent=l.timestamp!=='—'?l.timestamp:'waiting...';
  const sb=document.getElementById('sevBadge');
  sb.textContent=l.severity; sb.className='badge sev-'+l.severity;
  document.getElementById('modeBadge').textContent=l.model_used||'—';
  document.getElementById('confBar').style.width=(l.confidence||0)+'%';
  document.getElementById('confText').textContent=l.confidence>0?l.confidence.toFixed(1)+'%':'—';
  // Stats
  document.getElementById('totalSent').textContent=data.total_sent;
  document.getElementById('totalReadings').textContent=data.history.length;
  document.getElementById('statusBar').textContent=data.system_status;
  // Sensors
  const fr=l.fire_readings||{};
  const mq_pct=fr['MQ2_%']||0;
  document.getElementById('mqBarFill').style.width=mq_pct+'%';
  document.getElementById('livePct').textContent=mq_pct?mq_pct.toFixed(1)+'%':'—';
  Object.entries(PPM_MAP).forEach(([k,id])=>{
    const v=fr[k];
    if(v!==undefined) document.getElementById(id).textContent=typeof v==='number'?v.toFixed(1):v;
  });
  // Features grid
  const all={...fr,...(l.aq_readings||{})};
  const keys=Object.keys(all);
  document.getElementById('readingsContainer').innerHTML=keys.length?
    '<div class="feat-grid">'+keys.map(k=>
      `<div class="feat-item"><div class="feat-key">${k}</div>
       <div class="feat-val">${typeof all[k]==='number'?all[k].toFixed(2):all[k]}</div></div>`
    ).join('')+'</div>':'<div style="color:#4a6080">No readings yet</div>';
  // History
  document.getElementById('historyBody').innerHTML=data.history.length?
    data.history.slice(0,15).map(e=>
      `<tr><td>${e.timestamp}</td>
       <td style="color:${SEV_COL[e.severity]||'#fff'}">${e.alert_name}</td>
       <td>${e.confidence}%</td>
       <td><span style="color:${SEV_COL[e.severity]}">${e.severity}</span></td>
       </tr>`).join(''):
    '<tr><td colspan="4" style="text-align:center;color:#4a6080">No history</td></tr>';
}

fetch('/api/state').then(r=>r.json()).then(updateUI);
</script></body></html>"""

@app.route('/')
def index(): return render_template_string(HTML_PAGE)

@app.route('/api/state')
def api_state():
    with state_lock: return jsonify(state)

@app.route('/api/models')
def api_models():
    fa = aq = "—"
    if os.path.exists('fire_metadata.pkl'):
        with open('fire_metadata.pkl','rb') as f: fa = f"{pickle.load(f)['accuracy']*100:.1f}"
    if os.path.exists('aq_metadata.pkl'):
        with open('aq_metadata.pkl','rb') as f: aq = f"{pickle.load(f)['accuracy']*100:.1f}"
    return jsonify({"fire_acc": fa, "aq_acc": aq})

def run_web():
    import logging; logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False)

# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════
def main():
    print("\n" + "═"*55)
    print("  ESP8266 ML ALERT SYSTEM — SENSOR VERSION")
    print("═"*55)

    fire_model, fire_scaler, fire_meta, \
    aq_model, aq_scaler, aq_meta = load_models()
    df_fire, df_aq = load_datasets(fire_meta, aq_meta)
    ser = connect_serial()

    threading.Thread(target=run_web, daemon=True).start()
    print(f"\n🌐 Dashboard: http://localhost:{WEB_PORT}\n")

    with state_lock: state["system_status"] = "Idle"

    while True:
        print("\n" + "─"*42)
        print("  1 → Manual Mode")
        print("  2 → Auto Mode (dataset rows)")
        print("  3 → Sensor Mode  (DHT11 + MQ-2)")
        print("  q → Quit")
        print("─"*42)
        choice = input("  Choice: ").strip().lower()

        if   choice == '1':
            manual_mode(fire_model, fire_scaler, fire_meta, aq_model, aq_scaler, aq_meta, ser)
        elif choice == '2':
            auto_mode(fire_model, fire_scaler, fire_meta, aq_model, aq_scaler, aq_meta, ser, df_fire, df_aq)
        elif choice == '3':
            sensor_mode(fire_model, fire_scaler, fire_meta, aq_model, aq_scaler, aq_meta, ser)
        elif choice == 'q':
            print("\n👋 Bye!"); ser.close(); break
        else:
            print("⚠️  Enter 1, 2, 3, or q")

if __name__ == "__main__":
    main()
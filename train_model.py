"""
train_model.py
--------------
Trains TWO separate ML models:
  1. Fire Detection Model     → trained on Smoke Detection Dataset (actual fire sensor data)
  2. Air Quality Model        → trained on Air Quality Dataset (pollution levels)

Datasets needed (put in same folder):
  - smoke_detection.csv     → kaggle.com/datasets/deepcontractor/smoke-detection-dataset
  - pollution_dataset.csv   → kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment

Run this ONCE. Generates:
  fire_model.pkl, fire_scaler.pkl, fire_metadata.pkl
  aq_model.pkl,   aq_scaler.pkl,   aq_metadata.pkl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys

# ══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════

def save_confusion_matrix(y_test, y_pred, labels, title, filename):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  📊 Saved: {filename}")

def save_feature_importance(model, feature_cols, title, filename):
    plt.figure(figsize=(10, 5))
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    sns.barplot(x=imp.values, y=imp.index, palette='viridis')
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  📊 Saved: {filename}")

def find_csv(candidates):
    for name in candidates:
        if os.path.exists(name):
            return name
    return None

# ══════════════════════════════════════════════════════════
# MODEL 1 — FIRE DETECTION
# Dataset: Smoke Detection Dataset (deepcontractor, Kaggle)
# Target: Fire Alarm column (0 = no fire, 1 = fire)
# This is REAL fire sensor data — no mapping needed
# ══════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  MODEL 1 — FIRE DETECTION")
print("  Dataset: Smoke Detection (IoT sensor readings)")
print("═"*60)

fire_csv = find_csv([
    "smoke_detection.csv",
    "smoke_detection_iot.csv",
    "smoke.csv",
    "fire_detection.csv"
])

if fire_csv is None:
    print("❌ Smoke Detection dataset not found!")
    print("   Download from: kaggle.com/datasets/deepcontractor/smoke-detection-dataset")
    print("   Rename to: smoke_detection.csv")
    sys.exit(1)

df_fire = pd.read_csv(fire_csv)
print(f"✅ Loaded: {fire_csv}")
print(f"   Shape: {df_fire.shape}")
print(f"   Columns: {list(df_fire.columns)}")

# Find target column — Fire Alarm
fire_target = None
for col in df_fire.columns:
    if 'fire' in col.lower() and ('alarm' in col.lower() or 'alert' in col.lower()):
        fire_target = col
        break
if fire_target is None:
    for col in df_fire.columns:
        if 'alarm' in col.lower() or 'label' in col.lower():
            fire_target = col
            break

if fire_target is None:
    print(f"   Available columns: {list(df_fire.columns)}")
    fire_target = input("   Enter target column name: ").strip()

print(f"\n🎯 Target column: '{fire_target}'")
print(f"   Value counts:\n{df_fire[fire_target].value_counts()}")

# Drop non-feature columns
drop_cols = [fire_target]
# Drop UTC/timestamp and index-like columns
for col in df_fire.columns:
    cl = col.lower()
    if any(x in cl for x in ['utc', 'time', 'index', 'unnamed', 'cnt']):
        drop_cols.append(col)

feature_cols_fire = [c for c in df_fire.columns
                     if c not in drop_cols and df_fire[c].dtype in ['float64', 'int64']]

print(f"\n📐 Features ({len(feature_cols_fire)}): {feature_cols_fire}")

X_fire = df_fire[feature_cols_fire].copy()
y_fire = df_fire[fire_target].copy()

# Clean
X_fire = X_fire.dropna()
y_fire = y_fire[X_fire.index]

# Convert target to int if needed
y_fire = y_fire.astype(int)

print(f"   Samples: {len(X_fire)} | Fire={sum(y_fire==1)} | No Fire={sum(y_fire==0)}")

# Split
X_tr, X_te, y_tr, y_te = train_test_split(
    X_fire, y_fire, test_size=0.2, random_state=42, stratify=y_fire
)

# Scale
scaler_fire = StandardScaler()
X_tr_s = scaler_fire.fit_transform(X_tr)
X_te_s  = scaler_fire.transform(X_te)

# Train
print("\n🌲 Training Fire Detection Model...")
fire_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
fire_model.fit(X_tr_s, y_tr)

# Evaluate
y_pred_fire = fire_model.predict(X_te_s)
acc_fire = accuracy_score(y_te, y_pred_fire)
print(f"\n✅ Fire Model Accuracy: {acc_fire*100:.2f}%")
print(classification_report(y_te, y_pred_fire, target_names=['No Fire', 'FIRE']))

save_confusion_matrix(y_te, y_pred_fire,
    ['No Fire', 'FIRE'],
    f'Fire Detection Confusion Matrix (Acc: {acc_fire*100:.1f}%)',
    'fire_confusion_matrix.png')

save_feature_importance(fire_model, feature_cols_fire,
    'Fire Detection — Feature Importance',
    'fire_feature_importance.png')

# Save
with open('fire_model.pkl', 'wb') as f: pickle.dump(fire_model, f)
with open('fire_scaler.pkl', 'wb') as f: pickle.dump(scaler_fire, f)
fire_metadata = {
    'feature_cols': feature_cols_fire,
    'accuracy': acc_fire,
    'target_col': fire_target
}
with open('fire_metadata.pkl', 'wb') as f: pickle.dump(fire_metadata, f)
print("✅ Saved: fire_model.pkl, fire_scaler.pkl, fire_metadata.pkl")


# ══════════════════════════════════════════════════════════
# MODEL 2 — AIR QUALITY (Gas Leak / Medical / Normal)
# Dataset: Air Quality and Pollution Assessment
# Target: Air Quality column → Poor=Gas Leak, Moderate=Medical, Good=Normal
# ══════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  MODEL 2 — AIR QUALITY (Gas Leak / Medical)")
print("  Dataset: Air Quality & Pollution Assessment")
print("═"*60)

aq_csv = find_csv([
    "pollution_dataset.csv",
    "air_quality.csv",
    "updated_pollution_dataset.csv",
    "dataset.csv"
])

if aq_csv is None:
    print("❌ Air Quality dataset not found!")
    print("   Download from: kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment")
    print("   Rename to: pollution_dataset.csv")
    sys.exit(1)

df_aq = pd.read_csv(aq_csv)
print(f"✅ Loaded: {aq_csv}")
print(f"   Shape: {df_aq.shape}")

# Find target
aq_target = None
for col in df_aq.columns:
    if 'quality' in col.lower() or 'air' in col.lower():
        aq_target = col
        break

print(f"🎯 Target column: '{aq_target}'")

# Map labels → alert types
# Good = 0 (no alert), Moderate = 1 (Medical), Poor = 5 (Gas Leak)
# We EXCLUDE Hazardous from this model — fire model handles extreme danger
df_aq[aq_target] = df_aq[aq_target].astype(str).str.strip()

label_map = {}
for label in df_aq[aq_target].unique():
    l = label.lower()
    if 'good' in l:
        label_map[label] = 0      # No alert
    elif 'moderate' in l:
        label_map[label] = 1      # Medical alert
    elif 'poor' in l or 'unhealthy' in l:
        label_map[label] = 5      # Gas Leak alert
    elif 'hazard' in l:
        label_map[label] = 5      # Also Gas Leak (severe pollution, not fire)

print(f"\n🔗 Label mapping:")
alert_names = {0:'No Alert', 1:'MEDICAL', 2:'FIRE', 5:'GAS LEAK'}
for k, v in label_map.items():
    print(f"   '{k}' → {v} ({alert_names[v]})")

df_aq['alert_type'] = df_aq[aq_target].map(label_map)

feature_cols_aq = df_aq.select_dtypes(include=[np.number]).columns.tolist()
feature_cols_aq = [c for c in feature_cols_aq
                   if c != 'alert_type' and 'id' not in c.lower()]

print(f"\n📐 Features ({len(feature_cols_aq)}): {feature_cols_aq}")

X_aq = df_aq[feature_cols_aq].dropna()
y_aq = df_aq.loc[X_aq.index, 'alert_type']

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_aq, y_aq, test_size=0.2, random_state=42, stratify=y_aq
)

scaler_aq = StandardScaler()
X_tr2_s = scaler_aq.fit_transform(X_tr2)
X_te2_s  = scaler_aq.transform(X_te2)

print("\n🌲 Training Air Quality Model...")
aq_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
aq_model.fit(X_tr2_s, y_tr2)

y_pred_aq = aq_model.predict(X_te2_s)
acc_aq = accuracy_score(y_te2, y_pred_aq)
present = sorted(y_aq.unique())
print(f"\n✅ Air Quality Model Accuracy: {acc_aq*100:.2f}%")
print(classification_report(y_te2, y_pred_aq,
      target_names=[f"{k}:{alert_names[k]}" for k in present]))

save_confusion_matrix(y_te2, y_pred_aq,
    [alert_names[k] for k in present],
    f'Air Quality Confusion Matrix (Acc: {acc_aq*100:.1f}%)',
    'aq_confusion_matrix.png')

save_feature_importance(aq_model, feature_cols_aq,
    'Air Quality — Feature Importance',
    'aq_feature_importance.png')

with open('aq_model.pkl', 'wb') as f: pickle.dump(aq_model, f)
with open('aq_scaler.pkl', 'wb') as f: pickle.dump(scaler_aq, f)
aq_metadata = {
    'feature_cols': feature_cols_aq,
    'label_map': label_map,
    'alert_names': alert_names,
    'accuracy': acc_aq
}
with open('aq_metadata.pkl', 'wb') as f: pickle.dump(aq_metadata, f)
print("✅ Saved: aq_model.pkl, aq_scaler.pkl, aq_metadata.pkl")

print("\n" + "═"*60)
print("  TRAINING COMPLETE")
print(f"  Fire Detection Accuracy : {acc_fire*100:.2f}%")
print(f"  Air Quality Accuracy    : {acc_aq*100:.2f}%")
print("  Now run: python predict_and_send.py")
print("═"*60)
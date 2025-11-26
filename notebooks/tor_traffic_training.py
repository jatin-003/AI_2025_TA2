import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 

import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "data" / "tor_nontor_flows.csv"


synthetic_flows = [
    {
        "Source IP": "10.0.0.1",
        " Source Port": 54321,
        " Destination IP": "151.101.1.69",
        " Destination Port": 443,
        " Protocol": 6,  # TCP
        " Flow Duration": 12000000,
        " Flow Bytes/s": 150000.0,
        " Flow Packets/s": 120.0,
        " Flow IAT Mean": 100000.0,
        " Flow IAT Std": 15000.0,
        " Flow IAT Max": 200000.0,
        " Flow IAT Min": 50000.0,
        "Fwd IAT Mean": 90000.0,
        " Fwd IAT Std": 14000.0,
        " Fwd IAT Max": 180000.0,
        " Fwd IAT Min": 40000.0,
        "Bwd IAT Mean": 110000.0,
        " Bwd IAT Std": 16000.0,
        " Bwd IAT Max": 210000.0,
        " Bwd IAT Min": 60000.0,
        "Active Mean": 500000.0,
        " Active Std": 80000.0,
        " Active Max": 600000.0,
        " Active Min": 300000.0,
        "Idle Mean": 200000.0,
        " Idle Std": 30000.0,
        " Idle Max": 260000.0,
        " Idle Min": 150000.0,
        "label": "tor"
    },
    {
        "Source IP": "10.0.0.5",
        " Source Port": 51515,
        " Destination IP": "142.250.184.46",
        " Destination Port": 80,
        " Protocol": 6,
        " Flow Duration": 3000000,
        " Flow Bytes/s": 500000.0,
        " Flow Packets/s": 300.0,
        " Flow IAT Mean": 30000.0,
        " Flow IAT Std": 8000.0,
        " Flow IAT Max": 55000.0,
        " Flow IAT Min": 10000.0,
        "Fwd IAT Mean": 28000.0,
        " Fwd IAT Std": 7000.0,
        " Fwd IAT Max": 50000.0,
        " Fwd IAT Min": 9000.0,
        "Bwd IAT Mean": 32000.0,
        " Bwd IAT Std": 9000.0,
        " Bwd IAT Max": 60000.0,
        " Bwd IAT Min": 12000.0,
        "Active Mean": 150000.0,
        " Active Std": 20000.0,
        " Active Max": 190000.0,
        " Active Min": 110000.0,
        "Idle Mean": 50000.0,
        " Idle Std": 10000.0,
        " Idle Max": 70000.0,
        " Idle Min": 30000.0,
        "label": "nontor"
    },
    {
        "Source IP": "192.168.1.10",
        " Source Port": 60000,
        " Destination IP": "104.26.3.2",
        " Destination Port": 443,
        " Protocol": 6,
        " Flow Duration": 25000000,
        " Flow Bytes/s": 90000.0,
        " Flow Packets/s": 80.0,
        " Flow IAT Mean": 180000.0,
        " Flow IAT Std": 30000.0,
        " Flow IAT Max": 300000.0,
        " Flow IAT Min": 90000.0,
        "Fwd IAT Mean": 170000.0,
        " Fwd IAT Std": 28000.0,
        " Fwd IAT Max": 280000.0,
        " Fwd IAT Min": 80000.0,
        "Bwd IAT Mean": 190000.0,
        " Bwd IAT Std": 32000.0,
        " Bwd IAT Max": 320000.0,
        " Bwd IAT Min": 100000.0,
        "Active Mean": 700000.0,
        " Active Std": 90000.0,
        " Active Max": 820000.0,
        " Active Min": 550000.0,
        "Idle Mean": 250000.0,
        " Idle Std": 40000.0,
        " Idle Max": 310000.0,
        " Idle Min": 180000.0,
        "label": "tor"
    },
    {
        "Source IP": "192.168.1.15",
        " Source Port": 49152,
        " Destination IP": "172.217.160.78",
        " Destination Port": 443,
        " Protocol": 6,
        " Flow Duration": 5000000,
        " Flow Bytes/s": 400000.0,
        " Flow Packets/s": 260.0,
        " Flow IAT Mean": 40000.0,
        " Flow IAT Std": 10000.0,
        " Flow IAT Max": 80000.0,
        " Flow IAT Min": 15000.0,
        "Fwd IAT Mean": 38000.0,
        " Fwd IAT Std": 9000.0,
        " Fwd IAT Max": 75000.0,
        " Fwd IAT Min": 14000.0,
        "Bwd IAT Mean": 42000.0,
        " Bwd IAT Std": 11000.0,
        " Bwd IAT Max": 85000.0,
        " Bwd IAT Min": 16000.0,
        "Active Mean": 200000.0,
        " Active Std": 30000.0,
        " Active Max": 260000.0,
        " Active Min": 150000.0,
        "Idle Mean": 60000.0,
        " Idle Std": 15000.0,
        " Idle Max": 90000.0,
        " Idle Min": 35000.0,
        "label": "nontor"
    },
    {
        "Source IP": "10.1.0.3",
        " Source Port": 55000,
        " Destination IP": "185.220.101.1",
        " Destination Port": 9001,
        " Protocol": 6,
        " Flow Duration": 18000000,
        " Flow Bytes/s": 130000.0,
        " Flow Packets/s": 95.0,
        " Flow IAT Mean": 130000.0,
        " Flow IAT Std": 20000.0,
        " Flow IAT Max": 230000.0,
        " Flow IAT Min": 70000.0,
        "Fwd IAT Mean": 125000.0,
        " Fwd IAT Std": 19000.0,
        " Fwd IAT Max": 220000.0,
        " Fwd IAT Min": 65000.0,
        "Bwd IAT Mean": 135000.0,
        " Bwd IAT Std": 21000.0,
        " Bwd IAT Max": 240000.0,
        " Bwd IAT Min": 75000.0,
        "Active Mean": 550000.0,
        " Active Std": 75000.0,
        " Active Max": 650000.0,
        " Active Min": 420000.0,
        "Idle Mean": 210000.0,
        " Idle Std": 35000.0,
        " Idle Max": 270000.0,
        " Idle Min": 160000.0,
        "label": "tor"
    },
]


df = pd.read_csv(CSV_PATH)

print("Columns in dataset:")
print(df.columns.tolist()[:50])  
print("Shape:", df.shape)


synthetic_df = pd.DataFrame(synthetic_flows)
df = pd.concat([df, synthetic_df], ignore_index=True)

print("Shape after adding synthetic rows:", df.shape)
print("Tail with synthetic examples:")
print(df.tail())

df = df.drop_duplicates()

df = df.dropna(how="all")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()  

print("Shape after cleaning:", df.shape)

TARGET_COLUMN = "label"   

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Set TARGET_COLUMN correctly. Current value '{TARGET_COLUMN}' not found in columns.")

print("Unique label values:", df[TARGET_COLUMN].unique())


def label_to_binary(x: str) -> int:
    x_str = str(x).lower()
    if "tor" in x_str and "non" not in x_str:
        return 1
    else:
        return 0

df["target"] = df[TARGET_COLUMN].apply(label_to_binary)

numeric_df = df.select_dtypes(include=[np.number])

FEATURE_COLUMNS = [
    ' Source Port',
    ' Destination Port',
    ' Protocol',
    ' Flow Duration',
    ' Flow Bytes/s',
    ' Flow Packets/s',
    ' Flow IAT Mean',
    ' Flow IAT Std',
    ' Flow IAT Max',     
    ' Flow IAT Min',    
    'Fwd IAT Mean',
    ' Fwd IAT Std',      
    ' Fwd IAT Max',     
    ' Fwd IAT Min',     
    'Bwd IAT Mean',
    ' Bwd IAT Std',     
    ' Bwd IAT Max',     
    ' Bwd IAT Min',     
    'Active Mean',
    ' Active Std',      
    ' Active Max',      
    ' Active Min',      
    'Idle Mean',
    ' Idle Std',        
    ' Idle Max',        
    ' Idle Min'         
]

print("Number of numeric features:", len(FEATURE_COLUMNS))
print("Sample feature columns:", FEATURE_COLUMNS[:15])


X = numeric_df[FEATURE_COLUMNS]
y = df["target"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Sample features:", FEATURE_COLUMNS[:10])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("Train size:", X_train.shape, "Test size:", X_test.shape)


models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "SVC_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True))
    ]),
    "RandomForest": Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "XGBoost": Pipeline([ 
        ("scaler", StandardScaler()), 
        ("clf", XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            n_estimators=200, 
            random_state=42,
            n_jobs=-1
        ))
    ])
}

results = {}


for name, model in models.items():
    print("\n" + "="*60)
    print(f"Training model: {name}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    results[name] = {
        "model": model,
        "accuracy": acc,
        "y_pred": y_pred
    }


best_name = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_name]["model"]
best_acc = results[best_name]["accuracy"]

print("\n" + "#"*60)
print(f"Best model: {best_name} with accuracy = {best_acc:.4f}")
print("#"*60)


models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

artifact = {
    "model_name": best_name,
    "model": best_model,
    "feature_columns": FEATURE_COLUMNS,
    "target_mapping": {"NonTor": 0, "Tor": 1}
}

models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)

pkl_path = models_dir / "best_tor_model.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(artifact, f)

print(f"Saved best model to: {pkl_path}")
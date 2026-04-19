import numpy as np, pandas as pd, joblib, json, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

with open("models/model_metadata.json") as f:
    meta = json.load(f)

df = pd.read_csv("generated_data/training_dataset_5000_orders_yards.csv")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

CM = {
    "Order_Quantity": "order_quantity",
    "Fabric_Width_cm": "fabric_width_cm",
    "Marker_Efficiency_%": "marker_efficiency",
    "Expected_Defect_Rate_%": "defect_rate",
    "Operator_Experience_Years": "operator_experience",
    "Garment_Type": "garment_type",
    "Fabric_Type": "fabric_type",
    "Pattern_Complexity": "pattern_complexity",
    "Season": "season",
    "Actual_Consumption_yards": "fabric_consumption_yards",
}
df = df.rename(columns=CM)
dc = [
    "order_quantity",
    "fabric_width_cm",
    "marker_efficiency",
    "defect_rate",
    "operator_experience",
    "fabric_consumption_yards",
    "garment_type",
    "fabric_type",
    "pattern_complexity",
    "season",
]
df = df.dropna(subset=[c for c in dc if c in df.columns])

CATS = {
    "garment_type": ["Dress", "Jacket", "Pants", "Shirt", "T-Shirt"],
    "fabric_type": ["Cotton", "Cotton-Blend", "Denim", "Polyester", "Silk"],
    "pattern_complexity": ["Complex", "Medium", "Simple"],
    "season": ["Fall", "Spring", "Summer", "Winter"],
}
for feat, cats in CATS.items():
    enc = LabelEncoder()
    enc.fit(cats)
    df[f"{feat}_encoded"] = enc.transform(df[feat])

FEATS = [
    "order_quantity",
    "fabric_width_cm",
    "marker_efficiency",
    "defect_rate",
    "operator_experience",
    "garment_type_encoded",
    "fabric_type_encoded",
    "pattern_complexity_encoded",
    "season_encoded",
]

X = df[FEATS].values.astype(np.float32)
y = df["fabric_consumption_yards"].values.astype(np.float32)

indices = np.arange(len(df))
X_tmp, X_te, y_tmp, y_te, idx_tmp, idx_te = train_test_split(
    X, y, indices, test_size=0.20, random_state=42
)
X_tr, X_v, y_tr, y_v, idx_tr, idx_v = train_test_split(
    X_tmp, y_tmp, idx_tmp, test_size=0.20, random_state=42
)
print(f"Train {len(X_tr)}  Val {len(X_v)}  Test {len(X_te)}")

scaler = joblib.load("models/scaler.pkl")
X_te_s = scaler.transform(X_te)

lr_m = joblib.load("models/linear_regression_model.pkl")
rf_m = joblib.load("models/random_forest_model.pkl")
xgb_m = joblib.load("models/xgboost_model.pkl")
ens_spec = joblib.load("models/ensemble_model.pkl")
lstm_m = tf.keras.models.load_model("models/lstm_model.h5", compile=False)

p_lr = lr_m.predict(X_te_s)
p_rf = rf_m.predict(X_te_s)
p_xgb = xgb_m.predict(X_te_s)
p_lstm = lstm_m.predict(X_te_s.reshape(-1, 1, X_te_s.shape[1]), verbose=0).flatten()
w = ens_spec["weights"]
p_ens = (
    w["xgboost"] * p_xgb
    + w["random_forest"] * p_rf
    + w["lstm"] * p_lstm
    + w["linear_regression"] * p_lr
)

yt = y_te.astype(np.float64)
models = {
    "Linear Reg.": p_lr,
    "Random Forest": p_rf,
    "XGBoost": p_xgb,
    "LSTM": p_lstm,
    "Ensemble": p_ens,
}

print("\n" + "=" * 70)
print("1. ERROR PERCENTILES (pred - actual, yards)")
print("=" * 70)
print(f"{'Model':<16} {'Min':>8} {'P10':>8} {'Median':>8} {'P90':>8} {'Max':>8}")
print("-" * 70)
for n, p in models.items():
    e = p - yt
    print(
        f"{n:<16} {np.min(e):>8.0f} {np.percentile(e, 10):>8.0f} {np.median(e):>8.0f} {np.percentile(e, 90):>8.0f} {np.max(e):>8.0f}"
    )

print("\nFigure values:")
print("  Linear Reg.:  -4821  -1204   -142  +1843  +6217")
print("  Random Forest: -2841   -921    -88   +932  +3184")
print("  XGBoost:      -2012   -614    -51   +584  +2841")
print("  LSTM:         -1841   -521    -44   +501  +2412")
print("  Ensemble:     -1694   -498    -42   +487  +2314")

tdf = df.iloc[idx_te]

print("\n" + "=" * 70)
print("2. SEASONAL MAPE — XGBoost")
print("=" * 70)
for s in ["Spring", "Summer", "Fall", "Winter"]:
    m = tdf["season"].values == s
    if m.sum() > 0:
        a = yt[m]
        pr = p_xgb[m]
        mp = np.mean(np.abs((a - pr) / np.where(a == 0, 1, a))) * 100
        rm = np.sqrt(np.mean((a - pr) ** 2))
        print(f"  {s:<10} MAPE={mp:>6.1f}%  RMSE={rm:>8.1f}  n={m.sum()}")
print("Figure values: Spring=5.1  Summer=4.9  Fall=5.3  Winter=5.6")

print("\n" + "=" * 70)
print("3. FABRIC MAPE — XGBoost")
print("=" * 70)
for f in ["Polyester", "Denim", "Cotton", "Silk", "Cotton-Blend"]:
    m = tdf["fabric_type"].values == f
    if m.sum() > 0:
        a = yt[m]
        pr = p_xgb[m]
        mp = np.mean(np.abs((a - pr) / np.where(a == 0, 1, a))) * 100
        rm = np.sqrt(np.mean((a - pr) ** 2))
        print(f"  {f:<14} MAPE={mp:>6.1f}%  RMSE={rm:>8.1f}  n={m.sum()}")
print("Figure values: Polyester=4.8  Denim=5.2  Cotton=5.4  Silk=5.4  Blend=5.5")

print("\n" + "=" * 70)
print("4. VARIANCE BY ORDER QTY (full 5000)")
print("=" * 70)
pc = [c for c in df.columns if "Planned_BOM" in c and "yard" in c.lower()][0]
qty = df["order_quantity"].values
pln = df[pc].values
act = df["fabric_consumption_yards"].values
for lo, hi, lb in [
    (100, 500, "100-500"),
    (501, 1500, "501-1.5k"),
    (1501, 3000, "1.5k-3k"),
    (3001, 5000, "3k-5k"),
]:
    m = (qty >= lo) & (qty <= hi)
    if m.sum() > 0:
        ap = pln[m].mean()
        aa = act[m].mean()
        av = (act[m] - pln[m]).mean()
        avp = ((act[m] - pln[m]) / pln[m] * 100).mean()
        print(
            f"  {lb:<12} n={m.sum():>5}  Plan={ap:>8.1f}  Act={aa:>8.1f}  Var={av:>+8.1f} yd  Var%={avp:>+6.2f}%"
        )
ov = ((act - pln) / pln * 100).mean()
print(f"  {'All':<12} n={len(df):>5}  Var%={ov:>+6.2f}%")
print("Figure values: 100-500=+2.23%  501-1.5k=+2.16%  1.5k-3k=+1.68%  3k-5k=+1.07%")

print("\n" + "=" * 70)
print("5. CI COVERAGE (90% target)")
print("=" * 70)
nkm = {
    "Linear Reg.": "linear_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "LSTM": "lstm",
    "Ensemble": "ensemble",
}
for n, p in models.items():
    cf = meta["models"][nkm[n]]["ci_bounds"]["ci_fraction"]
    ae = np.abs(yt - p)
    cw = cf * yt
    cov = (ae <= cw).mean() * 100
    print(f"  {n:<16} coverage={cov:>5.1f}%  ci_frac={cf}")
print("Figure values: LR=88.1  RF=91.4  LSTM=90.7  XGB=92.3  Ensemble=93.1")

print("\nDone.")

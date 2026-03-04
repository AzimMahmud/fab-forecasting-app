"""
================================================================================
                    FABRIC CONSUMPTION FORECASTING SYSTEM
                         DATA GENERATION SCRIPT
================================================================================

Generates realistic synthetic fabric consumption datasets for ML training.

Key improvements in v3.0:
- Single 2% noise term replaces 7 compounded noise sources (was ~7% variance)
- Strong deterministic feature signals — each input has clear measurable impact
- Season added as encoded feature (consistent with train_models.py and app.py)
- Operator experience range aligned with app.py UI (1–20 years)
- Defect rate clipped to 0–10% (consistent with training domain)
- Marker efficiency clipped to 70–95% (consistent with training domain)
- Author and version corrected
- Interaction term columns labelled as exploratory-only (not ML features)

Author:         Azim Mahmud
Date:           January 2026
Version:        3.0.0
Thesis:         Optimizing Material Forecasting in Apparel Manufacturing
                Using Machine Learning
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json


# ============================================================================
# CONSTANTS
# ============================================================================

RANDOM_SEED = 42

GARMENT_TYPES    = ["T-Shirt", "Shirt", "Pants", "Dress", "Jacket"]
FABRIC_TYPES     = ["Cotton", "Polyester", "Cotton-Blend", "Silk", "Denim"]
COMPLEXITIES     = ["Simple", "Medium", "Complex"]
SEASONS          = ["Spring", "Summer", "Fall", "Winter"]
PRODUCTION_LINES = ["Line_A", "Line_B", "Line_C", "Line_D"]

# Base fabric consumption per unit (meters) at standard 160 cm width.
# Garment-specific values — must match AppConfig.GARMENT_BASE_CONSUMPTION_M in app.py.
BASE_CONSUMPTION_M = {
    "T-Shirt": 1.20,
    "Shirt":   1.80,
    "Pants":   2.50,
    "Dress":   3.00,
    "Jacket":  3.50,
}

FABRIC_COST_PER_M = {
    "Cotton":       8.5,
    "Polyester":    6.2,
    "Cotton-Blend": 7.0,
    "Silk":        25.0,
    "Denim":        9.5,
}

# Clearly separated complexity multipliers so ML can learn the signal.
COMPLEXITY_MULT = {
    "Simple":  1.00,
    "Medium":  1.15,
    "Complex": 1.35,
}

# Seasonal impact on consumption (small but real).
# Season IS a training feature — season_encoded is added to every CSV row.
SEASONAL_IMPACT = {
    "Spring":  0.010,
    "Summer": -0.008,
    "Fall":    0.005,
    "Winter":  0.018,
}

# Alphabetical encoding — must match sklearn LabelEncoder output in train_models.py.
# LabelEncoder.fit(["Spring","Summer","Fall","Winter"]) sorts → Fall=0, Spring=1, Summer=2, Winter=3
SEASON_ENCODING = {
    "Fall":   0,
    "Spring": 1,
    "Summer": 2,
    "Winter": 3,
}

STANDARD_WIDTH_CM = 160.0
BOM_SAFETY_MARGIN = 1.05


# ============================================================================
# UNIT CONVERTER
# ============================================================================

class UnitConverter:
    YARDS_TO_METERS = 0.9144
    METERS_TO_YARDS = 1.0936132983

    @staticmethod
    def yards_to_meters(yards):
        return yards * UnitConverter.YARDS_TO_METERS

    @staticmethod
    def meters_to_yards(meters):
        return meters * UnitConverter.METERS_TO_YARDS

    @staticmethod
    def inches_to_cm(inches):
        return inches * 2.54

    @staticmethod
    def cm_to_inches(cm):
        return cm / 2.54


# ============================================================================
# DATA GENERATOR
# ============================================================================

class FabricDataGenerator:
    """
    Generate realistic fabric consumption datasets for ML training.

    Signal design (v3.0.0)
    --------------------
    v2.0.0 problem : 7 independent noise sources compounded to ~7% variance.
                   Even a perfect model could only reach ~5.5% MAPE.
    v3.0.0 solution: One deterministic signal layer (learnable) + one 2% noise
                   term (irreducible). Expected Ensemble MAPE: 2–4%.

    Feature → target relationships (all learnable by ML):
        garment_type        — base per-unit consumption (1.2m – 3.5m)
        fabric_width_cm     — width scaling factor vs 160 cm standard
        pattern_complexity  — multiplier 1.00 / 1.15 / 1.35
        marker_efficiency   — ±5% efficiency impact
        defect_rate         — 0–10% additional material needed
        operator_experience — exponential decay (new operators waste ~4% more)
        season              — ±1–2% seasonal adjustment
        order_quantity      — economy of scale (−3% log-scale reduction)

    Single irreducible noise: N(0, 2%) applied once after all deterministic
    factors. This sets the theoretical MAPE floor at ~1.6%.
    """

    def __init__(self, random_seed=RANDOM_SEED):
        self.rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_dataset(self, n_samples=5000, start_date="2023-01-01",
                         unit="meters", include_both_units=True,
                         noise_std=0.020):
        """
        Generate a complete fabric consumption dataset.

        Parameters
        ----------
        n_samples        : number of orders to generate
        start_date       : first order date (YYYY-MM-DD)
        unit             : primary measurement unit ('meters' or 'yards')
        include_both_units : include both _m and _yards columns in output
        noise_std        : std-dev of irreducible operational noise (default 2%)
        """
        print(f"  Generating {n_samples} orders | unit={unit} | noise={noise_std*100:.1f}%")

        df = self._generate_base_features(n_samples, start_date)
        df = self._calculate_consumption(df, noise_std)
        df = self._add_unit_columns(df, unit, include_both_units)
        df = self._add_derived_features(df)

        avg_var = df["Variance_%"].abs().mean()
        cv = df["Actual_Consumption_m"].std() / df["Actual_Consumption_m"].mean()
        print(f"    QA => avg|variance|={avg_var:.2f}%  CV={cv:.3f}  rows={len(df)}")

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_base_features(self, n, start_date):
        start = pd.to_datetime(start_date)

        garment_type    = self.rng.choice(GARMENT_TYPES, n)
        fabric_type     = self.rng.choice(FABRIC_TYPES, n)
        pattern_complex = self.rng.choice(COMPLEXITIES, n)
        season          = self.rng.choice(SEASONS, n)
        prod_line       = self.rng.choice(PRODUCTION_LINES, n)
        width_inches    = self.rng.choice([55, 59, 63, 71], n)

        return pd.DataFrame({
            "Order_ID":   [f"ORD_{i:06d}" for i in range(1, n + 1)],
            "Order_Date": [start + timedelta(days=int(i / 5)) for i in range(n)],

            # ── 8 ML features (must match TrainingConfig.FEATURES in train_models.py) ──
            "Order_Quantity":            self.rng.integers(100, 5001, n),
            "Garment_Type":              garment_type,
            "Fabric_Type":               fabric_type,
            "Fabric_Width_inches":       width_inches,
            "Fabric_Width_cm":           (width_inches * 2.54).astype(float),
            "Pattern_Complexity":        pattern_complex,
            "Marker_Efficiency_%":       self.rng.normal(85.0, 4.0, n).clip(70, 95),
            "Expected_Defect_Rate_%":    self.rng.exponential(1.8, n).clip(0.0, 10.0),
            "Operator_Experience_Years": self.rng.integers(1, 21, n),    # 1–20 (UI max=20)
            "Season":                    season,

            # ── Context columns (not ML features — excluded from TrainingConfig.FEATURES) ──
            # Production_Line, Supplier_ID, and Fabric_GSM are operational metadata.
            # They are retained in the CSV for analysis but are NOT passed to model training.
            "Supplier_ID":     self.rng.choice([f"SUP_{i}" for i in range(1, 11)], n),
            "Production_Line": prod_line,
            "Fabric_GSM":      self.rng.normal(180, 25, n).clip(100, 300),
        })

    def _calculate_consumption(self, df, noise_std):
        """
        Compute planned BOM and actual consumption.

        Signal layer (deterministic, fully learnable):
            actual = planned_BOM
                     × efficiency_factor      (±5%)
                     × defect_factor          (0–10%)
                     × experience_factor      (0–4%)
                     × seasonal_factor        (±1–2%)
                     × economy_factor         (0–3% size reduction)
                     × (1 + ε)               ← single noise N(0, noise_std)

        Physical clip: ±15% of planned BOM.
        """
        n = len(df)

        # Base consumption per unit adjusted for width and complexity
        base = df["Garment_Type"].map(BASE_CONSUMPTION_M).astype(float)
        base *= (STANDARD_WIDTH_CM / df["Fabric_Width_cm"])
        base *= df["Pattern_Complexity"].map(COMPLEXITY_MULT).astype(float)
        df["Base_Consumption_Per_Unit_m"] = base

        # Traditional planned BOM (industry rule: qty × garment_base × 1.05 buffer)
        df["Planned_BOM_m"] = df["Order_Quantity"] * base * BOM_SAFETY_MARGIN

        # ── Deterministic adjustment factors ──────────────────────────────────
        # 1. Marker efficiency: +1% efficiency → -0.4% consumption
        efficiency_factor = 1.0 - (df["Marker_Efficiency_%"] - 85.0) / 100.0 * 0.40

        # 2. Defect rate: each 1% defect → 1% extra material needed
        defect_factor = 1.0 + df["Expected_Defect_Rate_%"] / 100.0

        # 3. Operator experience: new operators waste ~4% more, decays with experience
        exp_factor = 1.0 + np.exp(-df["Operator_Experience_Years"] / 15.0) * 0.04

        # 4. Seasonal variation (small but real; Season IS a training feature)
        seasonal_factor = 1.0 + df["Season"].map(SEASONAL_IMPACT).astype(float)

        # 5. Order size economy of scale (larger orders → marginally less waste)
        size_factor = 1.0 - 0.03 * np.log1p(df["Order_Quantity"] / 1000.0)

        # ── Single irreducible noise term ──────────────────────────────────────
        noise = self.rng.normal(0.0, noise_std, n)

        actual_m = (
            df["Planned_BOM_m"]
            * efficiency_factor
            * defect_factor
            * exp_factor
            * seasonal_factor
            * size_factor
            * (1.0 + noise)
        )

        # Physical sanity clip: ±15% of planned BOM
        df["Actual_Consumption_m"] = actual_m.clip(
            lower=df["Planned_BOM_m"] * 0.85,
            upper=df["Planned_BOM_m"] * 1.15,
        )
        return df

    def _add_unit_columns(self, df, unit, include_both):
        m2y = UnitConverter.METERS_TO_YARDS

        df["Planned_BOM_yards"]               = df["Planned_BOM_m"]               * m2y
        df["Actual_Consumption_yards"]         = df["Actual_Consumption_m"]         * m2y
        df["Base_Consumption_Per_Unit_yards"]  = df["Base_Consumption_Per_Unit_m"]  * m2y

        if unit.lower() == "yards":
            df["Planned_BOM"]               = df["Planned_BOM_yards"]
            df["Actual_Consumption"]        = df["Actual_Consumption_yards"]
            df["Base_Consumption_Per_Unit"] = df["Base_Consumption_Per_Unit_yards"]
            df["Measurement_Unit"]          = "yards"
        else:
            df["Planned_BOM"]               = df["Planned_BOM_m"]
            df["Actual_Consumption"]        = df["Actual_Consumption_m"]
            df["Base_Consumption_Per_Unit"] = df["Base_Consumption_Per_Unit_m"]
            df["Measurement_Unit"]          = "meters"

        df["Variance_m"]     = df["Actual_Consumption_m"]     - df["Planned_BOM_m"]
        df["Variance_yards"] = df["Actual_Consumption_yards"] - df["Planned_BOM_yards"]
        df["Variance_%"]     = (df["Variance_m"] / df["Planned_BOM_m"]) * 100

        return df

    def _add_derived_features(self, df):
        # Season encoded (alphabetical = sklearn LabelEncoder order).
        # Fall=0, Spring=1, Summer=2, Winter=3
        df["season_encoded"] = df["Season"].map(SEASON_ENCODING).astype(int)

        df["Order_Size_Category"] = pd.cut(
            df["Order_Quantity"],
            bins=[0, 500, 1500, 3000, 10_000],
            labels=["Small", "Medium", "Large", "XLarge"],
        )

        df["Fabric_Cost_Per_m"]    = df["Fabric_Type"].map(FABRIC_COST_PER_M)
        df["Fabric_Cost_Per_yard"] = df["Fabric_Cost_Per_m"] * UnitConverter.METERS_TO_YARDS

        df["Planned_Cost_USD"]      = df["Planned_BOM_m"]        * df["Fabric_Cost_Per_m"]
        df["Actual_Cost_USD"]       = df["Actual_Consumption_m"] * df["Fabric_Cost_Per_m"]
        df["Waste_Cost_USD"]        = df["Variance_m"].clip(lower=0) * df["Fabric_Cost_Per_m"]
        df["Savings_Potential_USD"] = df["Variance_m"].abs()          * df["Fabric_Cost_Per_m"]

        # NOTE: The following interaction terms are included for exploratory
        # analysis only. They are NOT part of the ML feature vector.
        df["Efficiency_x_Experience"] = (
            df["Marker_Efficiency_%"] * df["Operator_Experience_Years"]
        )
        df["Defect_x_Complexity"] = (
            df["Expected_Defect_Rate_%"]
            * df["Pattern_Complexity"].map(COMPLEXITY_MULT)
        )
        df["Order_Value_USD"] = (
            df["Order_Quantity"]
            * df["Fabric_Cost_Per_m"]
            * df["Base_Consumption_Per_Unit_m"]
        )
        return df


# ============================================================================
# EXPORT UTILITIES
# ============================================================================

def export_csv(df, path, essential_only=False):
    """Export dataset to CSV."""
    if essential_only:
        keep = [
            "Order_ID", "Order_Date", "Order_Quantity",
            "Garment_Type", "Fabric_Type",
            "Fabric_Width_inches", "Fabric_Width_cm",
            "Pattern_Complexity", "Season", "season_encoded",
            "Marker_Efficiency_%", "Expected_Defect_Rate_%",
            "Operator_Experience_Years",
            "Planned_BOM_m", "Planned_BOM_yards",
            "Actual_Consumption_m", "Actual_Consumption_yards",
            "Variance_%", "Waste_Cost_USD",
        ]
        df = df[[c for c in keep if c in df.columns]]

    df.to_csv(path, index=False)
    print(f"    -> {path}  ({len(df):,} rows × {len(df.columns)} cols)")


def export_summary(df, path):
    """Write human-readable statistics summary."""
    lines = []

    def section(t):
        lines.append(f"\n{t}")
        lines.append("=" * 60)

    section("Dataset Overview")
    lines += [
        f"Total orders  : {len(df):,}",
        f"Date range    : {df['Order_Date'].min().date()} -> {df['Order_Date'].max().date()}",
        f"Primary unit  : {df['Measurement_Unit'].iloc[0]}",
    ]

    section("Consumption (Meters)")
    lines += [
        f"  Avg Planned BOM       : {df['Planned_BOM_m'].mean():>10.2f} m",
        f"  Avg Actual            : {df['Actual_Consumption_m'].mean():>10.2f} m",
        f"  Avg |Variance|        : {df['Variance_m'].abs().mean():>10.2f} m",
        f"  Avg Variance %        : {df['Variance_%'].mean():>+10.2f}%",
        f"  Std Variance %        : {df['Variance_%'].std():>10.2f}%",
        f"  Avg |Variance %|      : {df['Variance_%'].abs().mean():>10.2f}%",
        f"  Avg |Variance %|      : {df['Variance_%'].abs().mean():>10.2f}%  (|diff|/Planned)",
        f"  MAPE vs Actual        : {(df['Variance_m'].abs()/df['Actual_Consumption_m']).mean()*100:>10.2f}%  (|diff|/Actual, true MAPE)",
    ]

    section("Consumption (Yards)")
    lines += [
        f"  Avg Planned BOM       : {df['Planned_BOM_yards'].mean():>10.2f} yd",
        f"  Avg Actual            : {df['Actual_Consumption_yards'].mean():>10.2f} yd",
        f"  Avg |Variance|        : {df['Variance_yards'].abs().mean():>10.2f} yd",
    ]

    section("Economic Impact")
    lines += [
        f"  Total planned cost    : ${df['Planned_Cost_USD'].sum():>12,.2f}",
        f"  Total actual cost     : ${df['Actual_Cost_USD'].sum():>12,.2f}",
        f"  Total waste cost      : ${df['Waste_Cost_USD'].sum():>12,.2f}",
        f"  Avg waste / order     : ${df['Waste_Cost_USD'].mean():>12.2f}",
    ]

    section("Variance Distribution")
    lines += [
        f"  Over-consumption      : {(df['Variance_m'] > 0).mean()*100:.1f}%",
        f"  Under-consumption     : {(df['Variance_m'] < 0).mean()*100:.1f}%",
    ]

    section("Feature-Target Correlations (Actual_Consumption_m)")
    for feat in [
        "Order_Quantity", "Marker_Efficiency_%",
        "Expected_Defect_Rate_%", "Operator_Experience_Years", "Fabric_Width_cm",
    ]:
        if feat in df.columns:
            r = df[feat].corr(df["Actual_Consumption_m"])
            lines.append(f"  r( {feat:<40} ) = {r:+.3f}")

    section("Season Distribution")
    if "Season" in df.columns:
        for s, cnt in df["Season"].value_counts().items():
            lines.append(f"  {s:<8}: {cnt} orders ({cnt/len(df)*100:.1f}%)")

    section("Expected ML Performance (1000+ rows, single 2% noise)")
    vs = df["Variance_%"].std()
    # For Gaussian noise N(0, sigma), E[|e|] = sigma * sqrt(2/pi) ~= sigma * 0.7979
    import math as _math
    half_normal_factor = _math.sqrt(2.0 / _math.pi)  # ~= 0.7979
    lines += [
        f"  Noise std (irreducible) : {vs:.2f}%",
        f"  Theoretical MAPE floor  : {vs * half_normal_factor:.2f}%  (sigma * sqrt(2/pi))",
        f"  Expected XGBoost MAPE   : {vs * 1.0:.2f} - {vs * 1.5:.2f}%",
        f"  Expected Ensemble MAPE  : {vs * half_normal_factor:.2f} - {vs * 1.2:.2f}%",
        f"  Expected Ensemble R2    : > 0.97  (>= 1000 rows)",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"    -> {path}")


def create_batch_template(path, n_samples=10):
    """Create CSV template for Streamlit batch prediction uploads."""
    template = {
        "Order_ID":  [f"ORD_{i:06d}" for i in range(1, n_samples + 1)],
        "Order_Quantity":            [1000, 1500, 2000, 800,  2500, 1200, 1800, 900,  3000, 1100][:n_samples],
        "Garment_Type":              ["T-Shirt","Shirt","Pants","Dress","Jacket",
                                      "T-Shirt","Shirt","Pants","Dress","Jacket"][:n_samples],
        "Fabric_Type":               ["Cotton","Polyester","Denim","Silk","Cotton-Blend",
                                      "Cotton","Polyester","Denim","Silk","Cotton"][:n_samples],
        "Fabric_Width_inches":       [63, 59, 63, 55, 71, 63, 59, 63, 55, 71][:n_samples],
        "Pattern_Complexity":        ["Simple","Medium","Complex","Medium","Simple",
                                      "Medium","Complex","Simple","Medium","Simple"][:n_samples],
        "Season":                    ["Spring","Summer","Fall","Winter","Spring",
                                      "Summer","Fall","Winter","Spring","Summer"][:n_samples],
        "Marker_Efficiency_%":       [85, 88, 82, 90, 86, 84, 87, 83, 91, 85][:n_samples],
        "Expected_Defect_Rate_%":    [2.0, 3.0, 4.0, 1.5, 2.5, 2.0, 3.5, 2.0, 1.0, 2.5][:n_samples],
        "Operator_Experience_Years": [5,  8,  3, 12,  6,  7, 10,  4, 15,  5][:n_samples],
    }
    pd.DataFrame(template).to_csv(path, index=False)
    print(f"    -> {path}  (batch upload template, {n_samples} rows)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("FABRIC CONSUMPTION DATA GENERATOR v3.0.0")
    print("=" * 70)
    print(f"Author           : Azim Mahmud")
    print(f"Random seed      : {RANDOM_SEED}")
    print(f"Noise std        : 2.0%  (single irreducible term — v3.0 design)")
    print(f"Season           : now a training feature (season_encoded column)")
    print(f"Operator exp range: 1–20 years (aligned with app.py UI)")

    out = "generated_data"
    os.makedirs(out, exist_ok=True)
    print(f"\nOutput : {out}/\n")

    gen = FabricDataGenerator(random_seed=RANDOM_SEED)

    # ── Dataset 1: 100-row demo (yards) ─────────────────────────────────────
    print("-" * 70)
    print("DATASET 1 — Demo  100 orders  (yards)")
    df1 = gen.generate_dataset(100, "2024-01-01", "yards")
    export_csv(df1, f"{out}/demo_dataset_100_orders_yards.csv")
    export_csv(df1, f"{out}/demo_dataset_essential_columns.csv", essential_only=True)
    export_summary(df1, f"{out}/demo_dataset_100_orders_yards_summary.txt")

    # ── Dataset 2: 1 000-row primary training set (yards) ─────────────────────
    print("\n" + "-" * 70)
    print("DATASET 2 — Training  1 000 orders  (yards)  [PRIMARY]")
    df2 = gen.generate_dataset(1000, "2023-01-01", "yards")
    export_csv(df2, f"{out}/training_dataset_1000_orders_yards.csv")
    export_summary(df2, f"{out}/training_dataset_1000_orders_yards_summary.txt")

    # ── Dataset 3: 5 000-row production set (yards) ───────────────────────────
    print("\n" + "-" * 70)
    print("DATASET 3 — Production  5 000 orders  (yards)")
    df3 = gen.generate_dataset(5000, "2022-01-01", "yards")
    export_csv(df3, f"{out}/production_dataset_5000_orders_yards.csv")
    export_summary(df3, f"{out}/production_dataset_5000_orders_yards_summary.txt")

    # ── Dataset 4: Batch upload template ──────────────────────────────────────
    print("\n" + "-" * 70)
    print("DATASET 4 — Batch Prediction Template")
    create_batch_template(f"{out}/batch_prediction_template.csv", n_samples=10)

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print("""
Files written to generated_data/:

  Training (use with train_models.py):
    training_dataset_1000_orders_yards.csv    (1 000 rows) <- PRIMARY

  Production:
    production_dataset_5000_orders_yards.csv  (5 000 rows)

  Demo:
    demo_dataset_100_orders_yards.csv   (100 rows)
    demo_dataset_essential_columns.csv  (lightweight)

  App batch upload:
    batch_prediction_template.csv       (10 rows, includes Season column)

  Summaries:
    *_summary.txt

Next steps:
  1. python train_models.py
       Expected MAPE  : XGBoost 2–4%,  Ensemble 1.5–3%
       Expected R²    : XGBoost >0.96, Ensemble >0.97
  2. streamlit run app.py
""")


if __name__ == "__main__":
    main()
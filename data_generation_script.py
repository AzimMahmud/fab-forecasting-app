"""
FABRIC CONSUMPTION DATA GENERATOR
==================================

Generates a 5,000-order dataset for ML training with yards as the sole
unit of measurement (no meters output columns).

Changes vs v2.0:
  - 5,000 orders (was split across 100/1000 demo runs)
  - Yards-only UOM -- all consumption, BOM and cost columns are in yards
  - Season feature included and label-encoded for direct ML use
  - Single primary output: training_dataset_5000_orders_yards.csv
  - Batch prediction template updated to yards column naming

Developer: Azim Mahmud
Version:   3.0.0
Date:      January 2026
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# CONSTANTS
# ============================================================================

METERS_TO_YARDS = 1.0936132983   # exact ISO ratio

# Base garment consumption per unit at standard 160 cm fabric width (meters).
# Physics stays in meters; all dataset output columns are yards.
BASE_CONSUMPTION_M = {
    "T-Shirt": 1.20,
    "Shirt":   1.80,
    "Pants":   2.50,
    "Dress":   3.00,
    "Jacket":  3.50,
}

FABRIC_COST_PER_M = {
    "Cotton":       8.50,
    "Polyester":    6.20,
    "Cotton-Blend": 7.00,
    "Silk":        25.00,
    "Denim":        9.50,
}

COMPLEXITY_MULTIPLIER = {
    "Simple":  1.00,
    "Medium":  1.15,
    "Complex": 1.35,
}

SEASONAL_IMPACT = {
    "Spring":  0.02,
    "Summer": -0.01,
    "Fall":    0.01,
    "Winter":  0.03,
}

# Alphabetical label-encoder mappings -- must stay in sync with
# EncodingMaps in app.py and TrainingConfig.CATEGORICAL_FEATURES in train_models.py
GARMENT_ENCODING = {"Dress": 0, "Jacket": 1, "Pants": 2, "Shirt": 3, "T-Shirt": 4}
FABRIC_ENCODING  = {"Cotton": 0, "Cotton-Blend": 1, "Denim": 2, "Polyester": 3, "Silk": 4}
COMPLEX_ENCODING = {"Complex": 0, "Medium": 1, "Simple": 2}
SEASON_ENCODING  = {"Fall": 0, "Spring": 1, "Summer": 2, "Winter": 3}

RANDOM_SEED = 42


# ============================================================================
# GENERATOR CLASS
# ============================================================================

class FabricDataGenerator:
    """Generate realistic fabric consumption data -- yards output."""

    def __init__(self, random_seed: int = RANDOM_SEED):
        np.random.seed(random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_dataset(self, n_samples: int = 5000,
                         start_date: str = "2022-01-01") -> pd.DataFrame:
        """
        Generate a complete yards-based fabric consumption dataset.

        Parameters
        ----------
        n_samples : int
            Number of orders to generate (default 5,000).
        start_date : str
            ISO date for the first order (default '2022-01-01').

        Returns
        -------
        pd.DataFrame
        """
        print(f"  Generating {n_samples:,} orders ...")
        df = self._generate_base_data(n_samples, start_date)
        df = self._calculate_consumption_yards(df)
        df = self._add_derived_features(df)
        df = self._add_encoded_columns(df)
        print(f"  [OK] {len(df):,} rows, {len(df.columns)} columns  |  unit: yards")
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_base_data(self, n: int, start_date: str) -> pd.DataFrame:
        start = pd.to_datetime(start_date)
        return pd.DataFrame({
            "Order_ID":
                [f"ORD_{str(i).zfill(6)}" for i in range(1, n + 1)],
            "Order_Date":
                [start + timedelta(days=int(i / 5)) for i in range(n)],
            "Order_Quantity":
                np.random.randint(100, 5001, n),
            "Garment_Type":
                np.random.choice(list(BASE_CONSUMPTION_M), n),
            "Fabric_Type":
                np.random.choice(list(FABRIC_COST_PER_M), n),
            "Fabric_Width_inches":
                np.random.choice([55, 59, 63, 71], n),
            "Pattern_Complexity":
                np.random.choice(list(COMPLEXITY_MULTIPLIER), n),
            "Season":
                np.random.choice(list(SEASONAL_IMPACT), n),
            "Supplier_ID":
                np.random.choice([f"SUP_{i}" for i in range(1, 11)], n),
            "Production_Line":
                np.random.choice(["Line_A", "Line_B", "Line_C", "Line_D"], n),
            "Operator_Experience_Years":
                np.random.randint(1, 21, n),          # 1-20 inclusive
            "Fabric_GSM":
                np.random.normal(180, 30, n).clip(100, 300),
            "Marker_Efficiency_%":
                np.random.normal(85, 5, n).clip(70, 95),
            "Expected_Defect_Rate_%":
                np.random.exponential(2, n).clip(0, 10),
        }).assign(Fabric_Width_cm=lambda d: d["Fabric_Width_inches"] * 2.54)

    def _calculate_consumption_yards(self, df: pd.DataFrame) -> pd.DataFrame:
        """Physics-based consumption model -- all outputs in yards."""
        # 1. Garment base at standard 160 cm width
        base_m = df["Garment_Type"].map(BASE_CONSUMPTION_M)

        # 2. Fabric-width adjustment  (wider -> less fabric per unit)
        base_m = base_m * (160.0 / df["Fabric_Width_cm"])

        # 3. Pattern complexity
        base_m = base_m * df["Pattern_Complexity"].map(COMPLEXITY_MULTIPLIER)

        # 4. Planned BOM with 5 % safety margin
        planned_m = df["Order_Quantity"] * base_m * 1.05

        # -- Variance factors ------------------------------------------
        f_quality = np.random.normal(0, 0.02, len(df))          # ?2 %
        f_eff     = 1.0 - (df["Marker_Efficiency_%"] - 85) / 100 * 0.40
        f_defect  = 1.0 + df["Expected_Defect_Rate_%"] / 100
        f_exp     = 1.0 + np.exp(-df["Operator_Experience_Years"] / 15.0) * 0.04
        f_season  = 1.0 + df["Season"].map(SEASONAL_IMPACT)
        f_size    = 1.0 - 0.05 * np.log(df["Order_Quantity"] / 1000.0 + 1)
        f_noise   = np.random.normal(0, 0.015, len(df))         # ?1.5 %

        actual_m = (
            planned_m * (1 + f_quality) * f_eff * f_defect
            * f_exp * f_season * f_size * (1 + f_noise)
        ).clip(lower=planned_m * 0.80, upper=planned_m * 1.30)

        # -- Convert to yards ------------------------------------------
        df["Base_Consumption_Per_Unit_yards"] = base_m    * METERS_TO_YARDS
        df["Planned_BOM_yards"]               = planned_m * METERS_TO_YARDS
        df["Actual_Consumption_yards"]         = actual_m  * METERS_TO_YARDS

        df["Variance_yards"] = (
            df["Actual_Consumption_yards"] - df["Planned_BOM_yards"]
        )
        df["Variance_%"] = (
            df["Variance_yards"] / df["Planned_BOM_yards"] * 100
        )
        df["Measurement_Unit"] = "yards"
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Economic and interaction features -- yards-denominated."""
        df["Order_Size_Category"] = pd.cut(
            df["Order_Quantity"],
            bins=[0, 500, 1500, 3000, 10001],
            labels=["Small", "Medium", "Large", "XLarge"],
        )
        # Cost per yard = cost_per_meter / meters_per_yard
        df["Fabric_Cost_Per_yard"] = (
            df["Fabric_Type"].map(FABRIC_COST_PER_M) / METERS_TO_YARDS
        )
        df["Planned_Cost_USD"]      = df["Planned_BOM_yards"]        * df["Fabric_Cost_Per_yard"]
        df["Actual_Cost_USD"]       = df["Actual_Consumption_yards"]  * df["Fabric_Cost_Per_yard"]
        df["Waste_Cost_USD"]        = df["Variance_yards"].clip(lower=0) * df["Fabric_Cost_Per_yard"]
        df["Savings_Potential_USD"] = df["Variance_yards"].abs()         * df["Fabric_Cost_Per_yard"]

        df["Efficiency_x_Experience"] = (
            df["Marker_Efficiency_%"] * df["Operator_Experience_Years"]
        )
        df["Defect_x_Complexity"] = (
            df["Expected_Defect_Rate_%"]
            * df["Pattern_Complexity"].map(COMPLEXITY_MULTIPLIER)
        )
        df["Order_Value_USD"] = (
            df["Order_Quantity"]
            * df["Base_Consumption_Per_Unit_yards"]
            * df["Fabric_Cost_Per_yard"]
        )
        return df

    def _add_encoded_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-computed label-encoded columns for direct ML use."""
        df["Garment_Type_Encoded"] = df["Garment_Type"].map(GARMENT_ENCODING)
        df["Fabric_Type_Encoded"]  = df["Fabric_Type"].map(FABRIC_ENCODING)
        df["Complexity_Encoded"]   = df["Pattern_Complexity"].map(COMPLEX_ENCODING)
        df["Season_Encoded"]       = df["Season"].map(SEASON_ENCODING)
        return df


# ============================================================================
# BATCH-PREDICTION TEMPLATE
# ============================================================================

def create_batch_prediction_template(filepath: str, n_samples: int = 10) -> None:
    """Create a CSV upload template for the Streamlit app's Batch Prediction page."""
    rows = [
        ("ORD_000001", 1000, "T-Shirt",  "Cotton",       63, "Simple",  85.0, 2.0,  5, "Spring"),
        ("ORD_000002", 1500, "Shirt",    "Polyester",     59, "Medium",  88.0, 3.0,  8, "Summer"),
        ("ORD_000003", 2000, "Pants",    "Denim",         63, "Complex", 82.0, 4.0,  3, "Fall"),
        ("ORD_000004",  800, "Dress",    "Silk",          55, "Medium",  90.0, 1.5, 12, "Winter"),
        ("ORD_000005", 2500, "Jacket",   "Cotton-Blend",  71, "Simple",  86.0, 2.5,  6, "Spring"),
        ("ORD_000006", 1200, "T-Shirt",  "Cotton",        63, "Medium",  84.0, 2.0,  7, "Summer"),
        ("ORD_000007", 1800, "Shirt",    "Polyester",     59, "Complex", 87.0, 3.5, 10, "Fall"),
        ("ORD_000008",  900, "Pants",    "Denim",         63, "Simple",  83.0, 2.0,  4, "Winter"),
        ("ORD_000009", 3000, "Dress",    "Silk",          55, "Medium",  91.0, 1.0, 15, "Spring"),
        ("ORD_000010", 1100, "Jacket",   "Cotton",        71, "Simple",  85.0, 2.5,  5, "Summer"),
    ]
    pd.DataFrame(
        rows[:n_samples],
        columns=[
            "Order_ID", "Order_Quantity", "Garment_Type", "Fabric_Type",
            "Fabric_Width_inches", "Pattern_Complexity",
            "Marker_Efficiency_%", "Expected_Defect_Rate_%",
            "Operator_Experience_Years", "Season",
        ],
    ).to_csv(filepath, index=False)
    print(f"  [OK] Batch template  -> {filepath}  ({n_samples} rows)")



# ============================================================================
# 100-SAMPLE BATCH PREDICTION DATASET
# ============================================================================

def generate_batch_prediction_dataset(filepath: str, n_samples: int = 100,
                                       random_seed: int = 99) -> None:
    """
    Generate a realistic 100-order batch prediction input CSV.

    Uses a separate random seed from the training data so orders are
    genuinely unseen. Columns match exactly what the Streamlit app
    Batch Prediction page expects for upload.
    """
    rng = np.random.default_rng(random_seed)

    garments     = ["T-Shirt", "Shirt", "Pants", "Dress", "Jacket"]
    fabrics      = ["Cotton", "Polyester", "Cotton-Blend", "Silk", "Denim"]
    complexities = ["Simple", "Medium", "Complex"]
    seasons      = ["Spring", "Summer", "Fall", "Winter"]
    widths_in    = [55, 59, 63, 71]

    order_ids     = [f"BATCH_{str(i).zfill(5)}" for i in range(1, n_samples + 1)]
    quantities    = rng.integers(100, 5001, n_samples)
    garment_types = rng.choice(garments, n_samples)
    fabric_types  = rng.choice(fabrics, n_samples)
    width_inches  = rng.choice(widths_in, n_samples)
    complexities_ = rng.choice(complexities, n_samples)
    seasons_      = rng.choice(seasons, n_samples)
    marker_eff    = np.round(rng.uniform(70.0, 95.0, n_samples), 1)
    defect_rate   = np.round(rng.exponential(2.0, n_samples).clip(0, 10), 1)
    op_exp        = rng.integers(1, 21, n_samples)

    df = pd.DataFrame({
        "Order_ID":                  order_ids,
        "Order_Quantity":            quantities.astype(int),
        "Garment_Type":              garment_types,
        "Fabric_Type":               fabric_types,
        "Fabric_Width_inches":       width_inches.astype(int),
        "Pattern_Complexity":        complexities_,
        "Season":                    seasons_,
        "Marker_Efficiency_%":       marker_eff,
        "Expected_Defect_Rate_%":    defect_rate,
        "Operator_Experience_Years": op_exp.astype(int),
    })

    df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"  [OK] Batch dataset   -> {filepath}  ({len(df)} rows)")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def export_summary_statistics(df: pd.DataFrame, filepath: str) -> None:
    over  = (df["Variance_yards"] > 0).mean() * 100
    under = (df["Variance_yards"] < 0).mean() * 100
    lines = [
        "FABRIC CONSUMPTION DATASET -- SUMMARY STATISTICS",
        "=" * 56,
        "",
        "Dataset Overview",
        "-" * 40,
        f"  Total orders     : {len(df):,}",
        f"  Date range       : {df['Order_Date'].min().date()} -> {df['Order_Date'].max().date()}",
        f"  Measurement unit : yards",
        "",
        "Consumption Statistics (yards)",
        "-" * 40,
        f"  Avg Planned BOM        : {df['Planned_BOM_yards'].mean():>10.2f} yd",
        f"  Avg Actual Consumption : {df['Actual_Consumption_yards'].mean():>10.2f} yd",
        f"  Avg Variance           : {df['Variance_yards'].mean():>10.2f} yd",
        f"  Avg Variance %%         : {df['Variance_%'].mean():>10.2f} %%",
        f"  Std Dev Variance %%     : {df['Variance_%'].std():>10.2f} %%",
        "",
        "Economic Impact",
        "-" * 40,
        f"  Total Planned Cost : ${df['Planned_Cost_USD'].sum():>14,.2f}",
        f"  Total Actual Cost  : ${df['Actual_Cost_USD'].sum():>14,.2f}",
        f"  Total Waste Cost   : ${df['Waste_Cost_USD'].sum():>14,.2f}",
        f"  Avg Waste / Order  : ${df['Waste_Cost_USD'].mean():>14.2f}",
        "",
        "Variance Distribution",
        "-" * 40,
        f"  Over-consumption  : {over:.1f} %%",
        f"  Under-consumption : {under:.1f} %%",
        "",
        "Garment Mix",
        "-" * 40,
    ]
    for g, cnt in df["Garment_Type"].value_counts().items():
        lines.append(f"  {g:<12} : {cnt:,} ({cnt/len(df)*100:.1f} %%)")
    lines += ["", "Fabric Mix", "-" * 40]
    for f, cnt in df["Fabric_Type"].value_counts().items():
        lines.append(f"  {f:<14} : {cnt:,} ({cnt/len(df)*100:.1f} %%)")
    lines += ["", "Season Mix", "-" * 40]
    for s, cnt in df["Season"].value_counts().items():
        lines.append(f"  {s:<8} : {cnt:,} ({cnt/len(df)*100:.1f} %%)")

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  [OK] Summary stats   -> {filepath}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("FABRIC CONSUMPTION DATA GENERATOR  v3.0.0")
    print("Unit of Measure : YARDS only")
    print("Dataset size    : 5,000 orders + 100-order batch prediction file")
    print("=" * 70)

    out = "generated_data"
    os.makedirs(out, exist_ok=True)
    print(f"\n? Output directory: {out}/\n")

    gen = FabricDataGenerator(random_seed=RANDOM_SEED)

    # -- Primary training dataset -----------------------------------------
    print("=" * 70)
    print("PRIMARY DATASET  (5,000 orders -- yards)")
    print("=" * 70)
    df = gen.generate_dataset(n_samples=5000, start_date="2022-01-01")

    primary_csv = f"{out}/training_dataset_5000_orders_yards.csv"
    df.to_csv(primary_csv, index=False)
    print(f"  [OK] Training data   -> {primary_csv}")

    export_summary_statistics(df, f"{out}/training_dataset_5000_orders_yards_summary.txt")

    # -- Small upload template (10 rows) ---------------------------------
    print()
    print("=" * 70)
    print("BATCH TEMPLATE  (10 sample orders)")
    print("=" * 70)
    create_batch_prediction_template(f"{out}/batch_prediction_template.csv", n_samples=10)

    # -- 100-sample batch prediction dataset ------------------------------
    print()
    print("=" * 70)
    print("BATCH PREDICTION DATASET  (100 orders)")
    print("=" * 70)
    generate_batch_prediction_dataset(
        f"{out}/batch_prediction_100_orders_yards.csv",
        n_samples=100,
        random_seed=99,
    )

    # -- Done -------------------------------------------------------------
    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"""
? Files in '{out}/':
   training_dataset_5000_orders_yards.csv         <- main training dataset
   training_dataset_5000_orders_yards_summary.txt <- statistics
   batch_prediction_template.csv                  <- 10-row app upload template
   batch_prediction_100_orders_yards.csv          <- 100-row batch prediction input

? Key ML columns:
   Features : Order_Quantity, Fabric_Width_cm, Marker_Efficiency_%,
              Expected_Defect_Rate_%, Operator_Experience_Years,
              Garment_Type_Encoded, Fabric_Type_Encoded,
              Complexity_Encoded, Season_Encoded
   Target   : Actual_Consumption_yards

? Next step: python train_models.py
""")


if __name__ == "__main__":
    main()
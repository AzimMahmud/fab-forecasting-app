"""
FABRIC CONSUMPTION DATA GENERATOR
==================================

Standalone script to generate realistic fabric consumption datasets
with full yards/meters support for demonstration and testing.

Features:
- Generate any number of orders
- Dual unit support (meters & yards)
- Realistic manufacturing variance
- Export to CSV
- Multiple output formats

Author: Production Implementation
Date: January 2026
Version: 2.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ============================================================================
# UNIT CONVERSION CLASS
# ============================================================================

class UnitConverter:
    """Handle conversions between yards and meters"""
    
    YARDS_TO_METERS = 0.9144
    METERS_TO_YARDS = 1.0936132983
    
    @staticmethod
    def yards_to_meters(yards):
        return yards * UnitConverter.YARDS_TO_METERS
    
    @staticmethod
    def meters_to_yards(meters):
        return meters * UnitConverter.METERS_TO_YARDS

# ============================================================================
# DATA GENERATOR CLASS
# ============================================================================

class FabricDataGenerator:
    """Generate realistic fabric consumption data"""
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        
        # Configuration
        self.garment_types = ['T-Shirt', 'Shirt', 'Pants', 'Dress', 'Jacket']
        self.fabric_types = ['Cotton', 'Polyester', 'Cotton-Blend', 'Silk', 'Denim']
        self.complexities = ['Simple', 'Medium', 'Complex']
        self.seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        self.production_lines = ['Line_A', 'Line_B', 'Line_C', 'Line_D']
        
        # Base consumption per unit (meters)
        self.base_consumption_m = {
            'T-Shirt': 1.2,
            'Shirt': 1.8,
            'Pants': 2.5,
            'Dress': 3.0,
            'Jacket': 3.5
        }
        
        # Fabric costs
        self.fabric_cost_per_m = {
            'Cotton': 8.5,
            'Polyester': 6.2,
            'Cotton-Blend': 7.0,
            'Silk': 25.0,
            'Denim': 9.5
        }
        
        # Complexity multipliers
        self.complexity_multiplier = {
            'Simple': 1.0,
            'Medium': 1.15,
            'Complex': 1.35
        }
        
        # Seasonal impact
        self.seasonal_impact = {
            'Spring': 0.02,
            'Summer': -0.01,
            'Fall': 0.01,
            'Winter': 0.03
        }
    
    def generate_dataset(self, n_samples=1000, start_date='2024-01-01', 
                        unit='meters', include_both_units=True):
        """
        Generate complete fabric consumption dataset
        
        Parameters:
        -----------
        n_samples : int
            Number of orders to generate
        start_date : str
            Start date for order dates (YYYY-MM-DD)
        unit : str
            Primary unit ('meters' or 'yards')
        include_both_units : bool
            Include both meter and yard columns
            
        Returns:
        --------
        pd.DataFrame
            Complete dataset with all features
        """
        
        print(f"Generating {n_samples} orders...")
        print(f"Primary unit: {unit}")
        print(f"Start date: {start_date}")
        
        # Generate base data
        data = self._generate_base_data(n_samples, start_date)
        
        # Generate consumption values
        data = self._calculate_consumption(data)
        
        # Add unit conversions
        data = self._add_unit_conversions(data, unit, include_both_units)
        
        # Add derived features
        data = self._add_derived_features(data)
        
        print(f"✅ Dataset generated successfully!")
        print(f"   Total orders: {len(data)}")
        print(f"   Date range: {data['Order_Date'].min()} to {data['Order_Date'].max()}")
        print(f"   Average variance: {data['Variance_%'].mean():.2f}%")
        
        return data
    
    def _generate_base_data(self, n_samples, start_date):
        """Generate base order information"""
        
        start = pd.to_datetime(start_date)
        
        data = {
            'Order_ID': [f'ORD_{str(i).zfill(6)}' for i in range(1, n_samples + 1)],
            'Order_Date': [start + timedelta(days=int(i/5)) for i in range(n_samples)],
            'Order_Quantity': np.random.randint(100, 5000, n_samples),
            'Garment_Type': np.random.choice(self.garment_types, n_samples),
            'Fabric_Type': np.random.choice(self.fabric_types, n_samples),
            'Fabric_Width_inches': np.random.choice([55, 59, 63, 71], n_samples),
            'Pattern_Complexity': np.random.choice(self.complexities, n_samples),
            'Season': np.random.choice(self.seasons, n_samples),
            'Supplier_ID': np.random.choice([f'SUP_{i}' for i in range(1, 11)], n_samples),
            'Production_Line': np.random.choice(self.production_lines, n_samples),
            'Operator_Experience_Years': np.random.randint(1, 20, n_samples),
            'Fabric_GSM': np.random.normal(180, 30, n_samples).clip(100, 300),
            'Marker_Efficiency_%': np.random.normal(85, 5, n_samples).clip(70, 95),
            'Expected_Defect_Rate_%': np.random.exponential(2, n_samples).clip(0, 10),
        }
        
        df = pd.DataFrame(data)
        
        # Convert width to cm
        df['Fabric_Width_cm'] = df['Fabric_Width_inches'] * 2.54
        
        return df
    
    def _calculate_consumption(self, df):
        """Calculate planned and actual consumption"""
        
        # Base consumption per unit
        df['Base_Consumption_Per_Unit_m'] = df['Garment_Type'].map(self.base_consumption_m)
        
        # Adjust for fabric width
        width_factor = 160 / df['Fabric_Width_cm']
        df['Base_Consumption_Per_Unit_m'] *= width_factor
        
        # Adjust for pattern complexity
        df['Complexity_Multiplier'] = df['Pattern_Complexity'].map(self.complexity_multiplier)
        df['Base_Consumption_Per_Unit_m'] *= df['Complexity_Multiplier']
        
        # Calculate planned BOM (with 5% safety margin)
        safety_margin = 1.05
        df['Planned_BOM_m'] = df['Order_Quantity'] * df['Base_Consumption_Per_Unit_m'] * safety_margin
        
        # Generate actual consumption with realistic variance
        actual_consumption_m = df['Planned_BOM_m'].copy()
        
        # Factor 1: Fabric quality variations (±3%)
        fabric_quality_variance = np.random.normal(0, 0.03, len(df))
        actual_consumption_m *= (1 + fabric_quality_variance)
        
        # Factor 2: Marker efficiency impact
        efficiency_impact = (df['Marker_Efficiency_%'] - 85) / 100
        actual_consumption_m *= (1 - efficiency_impact * 0.5)
        
        # Factor 3: Defect rate impact
        defect_impact = df['Expected_Defect_Rate_%'] / 100
        actual_consumption_m *= (1 + defect_impact)
        
        # Factor 4: Operator experience
        experience_factor = np.exp(-df['Operator_Experience_Years'] / 30) * 0.05
        actual_consumption_m *= (1 + experience_factor)
        
        # Factor 5: Seasonal variations
        actual_consumption_m *= (1 + df['Season'].map(self.seasonal_impact))
        
        # Factor 6: Order size economies
        size_factor = -0.05 * np.log(df['Order_Quantity'] / 1000 + 1)
        actual_consumption_m *= (1 + size_factor)
        
        # Factor 7: Random operational variance (±4%)
        operational_noise = np.random.normal(0, 0.04, len(df))
        actual_consumption_m *= (1 + operational_noise)
        
        # Ensure positive and reasonable
        df['Actual_Consumption_m'] = actual_consumption_m.clip(
            lower=df['Planned_BOM_m'] * 0.8,
            upper=df['Planned_BOM_m'] * 1.3
        )
        
        return df
    
    def _add_unit_conversions(self, df, unit, include_both_units):
        """Add unit conversions"""
        
        # Convert to yards
        df['Planned_BOM_yards'] = df['Planned_BOM_m'] * UnitConverter.METERS_TO_YARDS
        df['Actual_Consumption_yards'] = df['Actual_Consumption_m'] * UnitConverter.METERS_TO_YARDS
        df['Base_Consumption_Per_Unit_yards'] = df['Base_Consumption_Per_Unit_m'] * UnitConverter.METERS_TO_YARDS
        
        # Set primary unit
        if unit.lower() == 'yards':
            df['Planned_BOM'] = df['Planned_BOM_yards']
            df['Actual_Consumption'] = df['Actual_Consumption_yards']
            df['Base_Consumption_Per_Unit'] = df['Base_Consumption_Per_Unit_yards']
            df['Measurement_Unit'] = 'yards'
        else:
            df['Planned_BOM'] = df['Planned_BOM_m']
            df['Actual_Consumption'] = df['Actual_Consumption_m']
            df['Base_Consumption_Per_Unit'] = df['Base_Consumption_Per_Unit_m']
            df['Measurement_Unit'] = 'meters'
        
        # Calculate variance
        df['Variance_m'] = df['Actual_Consumption_m'] - df['Planned_BOM_m']
        df['Variance_yards'] = df['Actual_Consumption_yards'] - df['Planned_BOM_yards']
        df['Variance_%'] = (df['Variance_m'] / df['Planned_BOM_m']) * 100
        
        return df
    
    def _add_derived_features(self, df):
        """Add derived features and calculations"""
        
        # Order size category
        df['Order_Size_Category'] = pd.cut(
            df['Order_Quantity'],
            bins=[0, 500, 1500, 3000, 10000],
            labels=['Small', 'Medium', 'Large', 'XLarge']
        )
        
        # Fabric costs
        df['Fabric_Cost_Per_m'] = df['Fabric_Type'].map(self.fabric_cost_per_m)
        df['Fabric_Cost_Per_yard'] = df['Fabric_Cost_Per_m'] * UnitConverter.METERS_TO_YARDS
        
        # Economic calculations
        df['Planned_Cost_USD'] = df['Planned_BOM_m'] * df['Fabric_Cost_Per_m']
        df['Actual_Cost_USD'] = df['Actual_Consumption_m'] * df['Fabric_Cost_Per_m']
        df['Waste_Cost_USD'] = df['Variance_m'].clip(lower=0) * df['Fabric_Cost_Per_m']
        df['Savings_Potential_USD'] = abs(df['Variance_m']) * df['Fabric_Cost_Per_m']
        
        # Additional features
        df['Efficiency_x_Experience'] = df['Marker_Efficiency_%'] * df['Operator_Experience_Years']
        df['Defect_x_Complexity'] = df['Expected_Defect_Rate_%'] * df['Complexity_Multiplier']
        df['Order_Value_USD'] = df['Order_Quantity'] * df['Fabric_Cost_Per_m'] * df['Base_Consumption_Per_Unit_m']
        
        return df

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_to_csv(df, filename, include_all_columns=True):
    """Export dataset to CSV"""
    
    if not include_all_columns:
        # Export only essential columns
        essential_cols = [
            'Order_ID', 'Order_Date', 'Order_Quantity', 'Garment_Type', 'Fabric_Type',
            'Fabric_Width_inches', 'Fabric_Width_cm', 'Pattern_Complexity', 'Season',
            'Marker_Efficiency_%', 'Expected_Defect_Rate_%', 'Operator_Experience_Years',
            'Planned_BOM_m', 'Planned_BOM_yards', 'Actual_Consumption_m', 
            'Actual_Consumption_yards', 'Variance_%', 'Waste_Cost_USD'
        ]
        df_export = df[[col for col in essential_cols if col in df.columns]]
    else:
        df_export = df
    
    df_export.to_csv(filename, index=False)
    print(f"✅ Exported to {filename}")
    print(f"   Rows: {len(df_export)}, Columns: {len(df_export.columns)}")

def export_summary_statistics(df, filename):
    """Export summary statistics"""
    
    summary = {
        'Dataset Overview': {
            'Total Orders': len(df),
            'Date Range': f"{df['Order_Date'].min()} to {df['Order_Date'].max()}",
            'Primary Unit': df['Measurement_Unit'].iloc[0],
        },
        'Consumption Statistics (Meters)': {
            'Avg Planned BOM': f"{df['Planned_BOM_m'].mean():.2f}",
            'Avg Actual Consumption': f"{df['Actual_Consumption_m'].mean():.2f}",
            'Avg Variance': f"{df['Variance_m'].mean():.2f}",
            'Avg Variance %': f"{df['Variance_%'].mean():.2f}%",
        },
        'Consumption Statistics (Yards)': {
            'Avg Planned BOM': f"{df['Planned_BOM_yards'].mean():.2f}",
            'Avg Actual Consumption': f"{df['Actual_Consumption_yards'].mean():.2f}",
            'Avg Variance': f"{df['Variance_yards'].mean():.2f}",
        },
        'Economic Impact': {
            'Total Planned Cost': f"${df['Planned_Cost_USD'].sum():,.2f}",
            'Total Actual Cost': f"${df['Actual_Cost_USD'].sum():,.2f}",
            'Total Waste Cost': f"${df['Waste_Cost_USD'].sum():,.2f}",
            'Avg Waste Per Order': f"${df['Waste_Cost_USD'].mean():.2f}",
        },
        'Distribution': {
            'Over-consumption %': f"{(df['Variance_m'] > 0).mean() * 100:.1f}%",
            'Under-consumption %': f"{(df['Variance_m'] < 0).mean() * 100:.1f}%",
            'Perfect match %': f"{(df['Variance_m'] == 0).mean() * 100:.1f}%",
        }
    }
    
    with open(filename, 'w') as f:
        for section, stats in summary.items():
            f.write(f"\n{section}\n")
            f.write("=" * 50 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
    
    print(f"✅ Summary exported to {filename}")

def create_batch_prediction_template(filename, n_samples=10):
    """Create template CSV for batch predictions"""
    
    template_data = {
        'Order_ID': [f'ORD_{str(i).zfill(6)}' for i in range(1, n_samples + 1)],
        'Order_Quantity': [1000, 1500, 2000, 800, 2500, 1200, 1800, 900, 3000, 1100][:n_samples],
        'Garment_Type': ['T-Shirt', 'Shirt', 'Pants', 'Dress', 'Jacket', 'T-Shirt', 'Shirt', 'Pants', 'Dress', 'Jacket'][:n_samples],
        'Fabric_Type': ['Cotton', 'Polyester', 'Denim', 'Silk', 'Cotton-Blend', 'Cotton', 'Polyester', 'Denim', 'Silk', 'Cotton'][:n_samples],
        'Fabric_Width_inches': [63, 59, 63, 55, 71, 63, 59, 63, 55, 71][:n_samples],
        'Pattern_Complexity': ['Simple', 'Medium', 'Complex', 'Medium', 'Simple', 'Medium', 'Complex', 'Simple', 'Medium', 'Simple'][:n_samples],
        'Marker_Efficiency_%': [85, 88, 82, 90, 86, 84, 87, 83, 91, 85][:n_samples],
        'Expected_Defect_Rate_%': [2, 3, 4, 1.5, 2.5, 2, 3.5, 2, 1, 2.5][:n_samples],
        'Operator_Experience_Years': [5, 8, 3, 12, 6, 7, 10, 4, 15, 5][:n_samples],
    }
    
    df_template = pd.DataFrame(template_data)
    df_template.to_csv(filename, index=False)
    print(f"✅ Template created: {filename}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("="*70)
    print("FABRIC CONSUMPTION DATA GENERATOR v2.0")
    print("="*70)
    
    # Create output directory
    output_dir = 'generated_data'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}/\n")
    
    # Initialize generator
    generator = FabricDataGenerator(random_seed=42)
    
    # ========================================================================
    # DATASET 1: Small Demo Dataset (100 orders, meters)
    # ========================================================================
    
    print("\n" + "="*70)
    print("DATASET 1: Small Demo Dataset (Meters)")
    print("="*70)
    
    df_demo_meters = generator.generate_dataset(
        n_samples=100,
        start_date='2024-01-01',
        unit='meters',
        include_both_units=True
    )
    
    export_to_csv(
        df_demo_meters, 
        f'{output_dir}/demo_dataset_100_orders_meters.csv',
        include_all_columns=True
    )
    
    export_summary_statistics(
        df_demo_meters,
        f'{output_dir}/demo_dataset_100_orders_meters_summary.txt'
    )
    
    # ========================================================================
    # DATASET 2: Small Demo Dataset (100 orders, yards)
    # ========================================================================
    
    print("\n" + "="*70)
    print("DATASET 2: Small Demo Dataset (Yards)")
    print("="*70)
    
    df_demo_yards = generator.generate_dataset(
        n_samples=100,
        start_date='2024-01-01',
        unit='yards',
        include_both_units=True
    )
    
    export_to_csv(
        df_demo_yards,
        f'{output_dir}/demo_dataset_100_orders_yards.csv',
        include_all_columns=True
    )
    
    # ========================================================================
    # DATASET 3: Medium Training Dataset (1000 orders, meters)
    # ========================================================================
    
    print("\n" + "="*70)
    print("DATASET 3: Medium Training Dataset (Meters)")
    print("="*70)
    
    df_train_medium = generator.generate_dataset(
        n_samples=1000,
        start_date='2023-01-01',
        unit='meters',
        include_both_units=True
    )
    
    export_to_csv(
        df_train_medium,
        f'{output_dir}/training_dataset_1000_orders_meters.csv',
        include_all_columns=True
    )
    
    export_summary_statistics(
        df_train_medium,
        f'{output_dir}/training_dataset_1000_orders_meters_summary.txt'
    )
    
    # ========================================================================
    # DATASET 4: Large Production Dataset (5000 orders, meters)
    # ========================================================================
    
    print("\n" + "="*70)
    print("DATASET 4: Large Production Dataset (Meters)")
    print("="*70)
    
    df_production = generator.generate_dataset(
        n_samples=5000,
        start_date='2022-01-01',
        unit='meters',
        include_both_units=True
    )
    
    export_to_csv(
        df_production,
        f'{output_dir}/production_dataset_5000_orders_meters.csv',
        include_all_columns=True
    )
    
    export_summary_statistics(
        df_production,
        f'{output_dir}/production_dataset_5000_orders_meters_summary.txt'
    )
    
    # ========================================================================
    # DATASET 5: Batch Prediction Template
    # ========================================================================
    
    print("\n" + "="*70)
    print("DATASET 5: Batch Prediction Template")
    print("="*70)
    
    create_batch_prediction_template(
        f'{output_dir}/batch_prediction_template.csv',
        n_samples=10
    )
    
    # ========================================================================
    # DATASET 6: Essential columns only (for lightweight use)
    # ========================================================================
    
    print("\n" + "="*70)
    print("DATASET 6: Essential Columns Only (Lightweight)")
    print("="*70)
    
    export_to_csv(
        df_demo_meters,
        f'{output_dir}/demo_dataset_essential_columns.csv',
        include_all_columns=False
    )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    print(f"\n📊 Generated Files in '{output_dir}/':")
    print("\n1. demo_dataset_100_orders_meters.csv (100 orders, meters)")
    print("2. demo_dataset_100_orders_yards.csv (100 orders, yards)")
    print("3. training_dataset_1000_orders_meters.csv (1,000 orders)")
    print("4. production_dataset_5000_orders_meters.csv (5,000 orders)")
    print("5. batch_prediction_template.csv (template for uploads)")
    print("6. demo_dataset_essential_columns.csv (lightweight version)")
    print("\n📄 Summary Files:")
    print("- demo_dataset_100_orders_meters_summary.txt")
    print("- training_dataset_1000_orders_meters_summary.txt")
    print("- production_dataset_5000_orders_meters_summary.txt")
    
    print("\n✅ All datasets ready for use!")
    print("\n💡 Usage:")
    print("   - Use demo datasets for initial testing")
    print("   - Use training dataset for model development")
    print("   - Use production dataset for final training")
    print("   - Use template for batch prediction uploads")
    
    print("\n🎯 Next Steps:")
    print("   1. Review generated CSV files")
    print("   2. Check summary statistics")
    print("   3. Use datasets in ML training script")
    print("   4. Upload to Streamlit app for predictions")

if __name__ == "__main__":
    main()
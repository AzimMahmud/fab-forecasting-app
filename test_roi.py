#!/usr/bin/env python3

"""
Test script for ROI calculator functionality.
"""

import sys
sys.path.append('.')

from app.services import ROICalculator

def test_roi_calculator():
    """Test the ROI calculator functionality."""
    print("Testing ROI Calculator...")

    # Test basic calculation
    result = ROICalculator.calculate_roi(1000, 900)

    print(f"✓ ROI Calculator works: {result['savings_yards']} yards saved")
    print(f"  Cost savings: ${result['fabric_cost_savings']}")
    print(f"  Savings percentage: {result['savings_percentage']:.1f}%")
    print(f"  Projected annual savings: ${result['projected_annual_savings']:.2f}")
    print(f"  Efficiency improvement: {result['efficiency_improvement']:.1f}%")

    # Test report generation
    report = ROICalculator.generate_roi_report(result)
    print("\nGenerated Report:")
    print(report)

    return True

if __name__ == "__main__":
    test_roi_calculator()
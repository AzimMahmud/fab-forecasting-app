"""
Professional UI theme for Fabric Forecasting System.

Developer: Azim Mahmud | Version 4.0.0
"""

# Color Palette
PRIMARY_COLOR = "#1e3a5f"      # Deep navy blue
SECONDARY_COLOR = "#00b4d8"    # Teal accent
SUCCESS_COLOR = "#10b981"       # Green
WARNING_COLOR = "#f59e0b"       # Amber
ERROR_COLOR = "#ef4444"         # Red
BACKGROUND_COLOR = "#f8fafc"    # Light gray
TEXT_COLOR = "#1f2937"          # Dark gray
BORDER_COLOR = "#e5e7eb"        # Light border

# Typography
FONT_FAMILY = "Inter, Segoe UI, sans-serif"
FONT_SIZE_BASE = "14px"
FONT_SIZE_H1 = "32px"
FONT_SIZE_H2 = "24px"
FONT_SIZE_H3 = "18px"

# Spacing
SPACING_SM = "8px"
SPACING_MD = "16px"
SPACING_LG = "24px"
SPACING_XL = "32px"

# Border Radius
BORDER_RADIUS_SM = "4px"
BORDER_RADIUS_MD = "8px"
BORDER_RADIUS_LG = "12px"

# CSS Styles
CUSTOM_CSS = f"""
<style>
    /* Global Styles */
    .main {{
        font-family: {FONT_FAMILY};
        font-size: {FONT_SIZE_BASE};
        color: {TEXT_COLOR};
        background-color: {BACKGROUND_COLOR};
    }}

    /* Headers */
    h1 {{
        font-size: {FONT_SIZE_H1};
        font-weight: 700;
        color: {PRIMARY_COLOR};
        margin-bottom: {SPACING_MD};
    }}

    h2 {{
        font-size: {FONT_SIZE_H2};
        font-weight: 600;
        color: {PRIMARY_COLOR};
        margin-bottom: {SPACING_SM};
    }}

    h3 {{
        font-size: {FONT_SIZE_H3};
        font-weight: 500;
        color: {TEXT_COLOR};
        margin-bottom: {SPACING_SM};
    }}

    /* Buttons */
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border: none;
        border-radius: {BORDER_RADIUS_MD};
        padding: {SPACING_SM} {SPACING_LG};
        font-weight: 500;
        transition: all 0.2s ease;
    }}

    .stButton>button:hover {{
        background-color: {SECONDARY_COLOR};
        transform: translateY(-1px);
    }}

    /* Cards */
    .metric-card {{
        background: white;
        border: 1px solid {BORDER_COLOR};
        border-radius: {BORDER_RADIUS_LG};
        padding: {SPACING_LG};
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}

    /* Inputs */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>select,
    .stNumberInput>div>div>input {{
        border: 1px solid {BORDER_COLOR};
        border-radius: {BORDER_RADIUS_MD};
        padding: {SPACING_SM};
    }}

    /* Success Messages */
    .success-message {{
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid {SUCCESS_COLOR};
        padding: {SPACING_MD};
        border-radius: {BORDER_RADIUS_MD};
    }}

    /* Error Messages */
    .error-message {{
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid {ERROR_COLOR};
        padding: {SPACING_MD};
        border-radius: {BORDER_RADIUS_MD};
    }}
</style>
"""

def apply_custom_styles():
    """Apply custom CSS styles to the Streamlit app."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
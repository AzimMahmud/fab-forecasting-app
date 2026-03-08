"""
Toast notification system for user feedback.

Developer: Azim Mahmud | Version 4.0.0
"""

import streamlit as st
from typing import Optional
from app.ui_theme import SUCCESS_COLOR, ERROR_COLOR, WARNING_COLOR

def show_success(message: str, duration: int = 5000):
    """
    Display a success toast notification.

    Args:
        message: Success message to display
        duration: Duration in milliseconds (default: 5000)
    """
    st.success(message)
    st.toast(message, icon="✅")

def show_error(message: str, duration: int = 5000):
    """
    Display an error toast notification.

    Args:
        message: Error message to display
        duration: Duration in milliseconds (default: 5000)
    """
    st.error(message)
    st.toast(message, icon="❌")

def show_warning(message: str, duration: int = 5000):
    """
    Display a warning toast notification.

    Args:
        message: Warning message to display
        duration: Duration in milliseconds (default: 5000)
    """
    st.warning(message)
    st.toast(message, icon="⚠️")

def show_info(message: str, duration: int = 5000):
    """
    Display an info toast notification.

    Args:
        message: Info message to display
        duration: Duration in milliseconds (default: 5000)
    """
    st.info(message)
    st.toast(message, icon="ℹ️")
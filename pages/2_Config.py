import streamlit as st
import json

st.title("Config")

st.markdown(
    """
    Configures the default settings of the page.
    """
)

separator = st.selectbox("Select the delimiter", options=[
    ",", 
    ";"
    ], 
    key="select_separator")

if separator == ";":
    st.session_state["separator"] = ";"
    
decimal = st.selectbox("Select the decimal separator", options=[
    ".", 
    ","
    ], 
    key="select_decimal")

if decimal == ",":
    st.session_state["decimal"] = ","

settings = st.selectbox("Select the setting option", options=[
    "Basic", 
    "Advanced"
    ], 
    key="select_settings")

if settings == "Advanced":
    st.session_state["settings"] = "Advanced"
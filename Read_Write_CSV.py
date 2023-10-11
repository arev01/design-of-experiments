import streamlit as st
import pandas as pd

# ==========================================================
# Function for reading a CSV file into a dictionary format
# ==========================================================

def read_variables_csv(csvfile):
    """
    Builds a Python dictionary object from an input CSV file.
    Helper function to read a CSV file on the disk, where user stores the limits/ranges of the process variables.
    """

    dict_key={}
    try:
        data = pd.read_csv(csvfile, sep=st.session_state["separator"], decimal=st.session_state["decimal"])
        lowercase = lambda x: str(x).lower()
        #data.rename(lowercase, axis="columns", inplace=True)

        fields = list(data.columns)
        for field in fields:
            lst = data[field].tolist()
            dict_key[field]=lst
        
        return dict_key
    except:
        st.write("Error in reading the specified file. Please try again.")
        return -1
        
# ===============================================================
# Function for writing the design matrix into an output CSV file
# ===============================================================

def write_csv(df):
    """
    Writes a CSV file on to the disk from the internal Pandas DataFrame object i.e. the computed design matrix
    """
    
    try:
        return df.to_csv(index=False, sep=st.session_state["separator"], decimal=st.session_state["decimal"]).encode("utf-8")
    except:
        return -1

from pyDOE import *
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
#from st_pages import Page, show_pages, add_page_title

from Generate_DOE import *
from Read_Write_CSV import *

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

# Specify what pages should be shown in the sidebar, and what their titles 
# and icons should be
#show_pages(
#    [
#        Page("Main.py", "Home", "🏠"),
#        Page("pages/1_Help.py", "Help", ":question:"),
#        Page("pages/2_Config.py", "Config", ":gear:"),
#    ]
#)

st.write("# Design of Experiments - DoE")

st.markdown(
    """
    Online tool for generation and visualisation of experimental designs.  
    
    ### Getting Started  
    **👈 Need help to get started?** Go to the 'Help' section from the sidebar and learn the easy steps needed to create your first experimental design.  
    **👈 Hungry for more?** Go to the 'Config' section from the sidebar and unlock advanced functionalities.
    """
)

methods = [
    "Full-Factorial", 
    "Box-Behnken", 
    "Simple-Latin-Hypercube", 
    "Uniform-Random"
]

if "separator" not in st.session_state:
    st.session_state["separator"] = ","
    
if "decimal" not in st.session_state:
    st.session_state["decimal"] = "."

if "settings" not in st.session_state:
    st.session_state["settings"] = "Basic"

if st.session_state["settings"] == "Advanced":
    methods.extend(
        [
            "Plackett-Burman", 
            "Central-Composite", 
            "Space-Filling-Latin-Hypercube", 
            "Halton-Sequence"
        ]
    )

in_file = st.file_uploader("Upload file", type="csv")

if in_file:
    dict_vars = read_variables_csv(in_file)
    df = pd.DataFrame(dict_vars)
    
    if st.checkbox("Show raw data"):
        st.write(df)
    
        st.markdown("---")
        
    doe_choice = st.selectbox("Select the experiment design method", options=methods, key="select_method")

    df_updated, filename = generate_DOE(doe_choice, dict_vars)

    if type(df_updated) != int or type(filename) != int:
        out_file = write_csv(df_updated)

        st.write(df_updated)

        if len(df_updated.columns) == 2:
            fig = go.Figure(data=[go.Scatter(
                x=df_updated[df_updated.keys()[0]], 
                y=df_updated[df_updated.keys()[1]],
                mode='markers')])

            #fig.update_layout(aspectmode='cube')
            fig.update_scenes(aspectmode='cube',xaxis_title=df_updated.columns[0],yaxis_title=df_updated.columns[1],)

            st.plotly_chart(fig)
            
        elif len(df_updated.columns) == 3:
            fig = go.Figure(data=[go.Scatter3d(
                x=df_updated[df_updated.keys()[0]], 
                y=df_updated[df_updated.keys()[1]], 
                z=df_updated[df_updated.keys()[2]], 
                mode='markers')])

            #fig.update_layout(aspectmode='cube')
            fig.update_scenes(aspectmode='cube',xaxis_title=df_updated.columns[0],yaxis_title=df_updated.columns[1],zaxis_title=df_updated.columns[2])

            st.plotly_chart(fig)

        button = st.download_button("Download", out_file, filename, "text/csv")

        if button:
            st.success("Download successful!")

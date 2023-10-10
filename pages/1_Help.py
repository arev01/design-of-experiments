import streamlit as st
import pandas as pd

st.title("Help")

st.markdown(
    """
    ### What is a Design of Experiments?  

    A Design of Experiments (DoE) represents a sequence of experiments to be performed, expressed in terms of factors (design variables) set at specified levels (design values). Properly designed experiments are essential for the effective utilization by computer systems. Several methods exist to proceed with this sequencing, each with their pros and cons.  

    **Full-factorial designs**  

    The most basic DoE is a full factorial design. The number of design points is simply the product of the number of levels for each factor. Since the size of a full factorial experiment increases exponentially with the number of factors, this can lead to an unmanageable number of experiments.  

    **Sparse-grid designs**  

    A fractional factorial design is a fraction of a full factorial design. These designs are frequently used to "screen" factors to identify those with the greatest effects.  

    **Composite designs**  

    It is often desirable to use the smallest number of factor levels in a DoE. The most common design configured to reduce the number of design points is a central composite design. It is formed by combining a full factorial design with center points and two "star" points positioned at the extremes for each factor.  

    **Random designs**  

    A completely random design is a type of DoE where the levels are randomly assigned to each factor. These designs typically work well for large systems with many variables.  

    **Space-filling designs**  

    A space-filling design attempts to improve on the random design by ensuring that the selected points are uncorrelated and more evenly distributed throughout the design space.  

    """
)

st.markdown(
    """
    ### How to use this tool?  
    
    There are three simple steps that must be taken to successfully create an experimental design:  
    1. Upload submission file  
    2. Select experiment design method
    3. Download DoE file
    """
)

st.markdown(
    """
    *Note: The variable names and their ranges (2-level i.e. min/max) are specified in an input file using a Python table data structure. This file can be edited with a normal spreadsheet program (e.g. Microsoft Excel) or using any text editor, but it must be saved in csv format. An example is shown below:*
    """
)

data = pd.read_csv("example/params.csv")

st.write(data)

with open("example/params.csv", "rb") as file:
    button = st.download_button(
        label="Download",
        data=file,
        file_name="params.csv"
    )
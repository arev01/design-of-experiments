import streamlit as st

from DOE_functions import *
from Read_Write_CSV import *

# ====================================================================
# Function to generate the DOE based on user's choice and input file
# ====================================================================

def generate_DOE(doe_choice, infile):
    """
    Generates the output design-of-experiment matrix by calling the appropriate function from the "DOE_function.py file".
    Returns the generated DataFrame (Pandas) and a filename (string) corresponding to the type of the DOE sought by the user. This filename string is used by the CSV writer function to write to the disk i.e. save the generated DataFrame in a CSV format.
    """
    
    dict_vars = read_variables_csv(infile)
    import pandas as pd
    df = pd.DataFrame(dict_vars)
    st.write(df)
    # if type(dict_vars)!=int:
    #     factor_count=len(dict_vars)
    # else:
    #     return (-1,-1)
    
    remove_key, remove_val, remove_index = ([] for i in range(3))
    
    i = 0
    # result from count matches with result from len()
    for key, val in dict_vars.items():
        if val.count(val[0]) == len(val):
            st.warning("All the elements of \'%s\' are equal. Dropping parameter..." % key)
            remove_key.append(key)
            remove_val.append(val[0])
            remove_index.append(i)
        i += 1
            
    for key in remove_key:
        dict_vars.pop(key, None)
        
    if doe_choice == "Full-Factorial":
        levels=st.text_input("Select the number of evenly spaced levels per factor", key="textbox_levels")
        try:
            levels=int(levels)
            df=build_full_fact(dict_vars, levels)
        except:
            st.error("Wrong input (not convertible to integer).")
            return (-1, -1)
        
    if doe_choice == "Box-Behnken":
        points=st.text_input("Select the number of center points to include", key="textbox_points")
        try:
            points=int(points)
            df=build_box_behnken(dict_vars, points)
        except:
            st.error("Wrong input (not convertible to integer).")
            return (-1, -1)
        
    if doe_choice == "Simple-Latin-Hypercube":
        samples=st.text_input("Select the overall number of samples for your experiment", key="textbox_samples")
        try:
            samples=int(samples)
            df=build_lhs(dict_vars, samples)
        except:
            st.error("Wrong input (not convertible to integer).")
            return (-1, -1)
        
    if doe_choice == "Uniform-Random":
        samples=st.text_input("Select the overall number of samples for your experiment", key="textbox_samples")
        try:
            samples=int(samples)
            df=build_uniform_random(dict_vars, samples)
        except:
            st.error("Wrong input (not convertible to integer).")
            return (-1, -1)
        
    if doe_choice == "Plackett-Burman":
        df=build_plackett_burman(dict_vars)

    if doe_choice == "Central-Composite":
        type=st.selectbox("Select the type of experimental design", options=["ccc", "cci", "ccf"], key="select_type")
        df=build_central_composite(dict_vars, face=type)

    if doe_choice == "Space-Filling-Latin-Hypercube":
        samples=st.text_input("Select the overall number of samples for your experiment", key="textbox_samples")
        try:
            samples=int(samples)
            df=build_space_filling_lhs(dict_vars, samples)
        except:
            st.error("Wrong input (not convertible to integer).")
            return (-1, -1)
        
    if doe_choice == "Halton-Sequence":
        samples=st.text_input("Select the overall number of samples for your experiment", key="textbox_samples")
        try:
            samples=int(samples)
            df=build_halton(dict_vars, samples)
        except:
            st.error("Wrong input (not convertible to integer).")
            return (-1, -1)
    
    nrow = len(df)
    for n, key in enumerate(remove_key):
        #df[key] = [remove_val[n]] * nrow
        df.insert(remove_index[n], key, [remove_val[n]] * nrow)

    filename = doe_choice+".csv"
    
    return (df,filename)

import streamlit as st

st.set_page_config(page_title="Multi-Page App", layout="wide")

st.title("Welcome to the Multi-Page App")
st.write("Choose an application to proceed:")

# Selection Menu
option = st.radio("Select an Option", ["Exercise Form Estimator", "Nutrition Recommendation Model"])

# Button to confirm selection
if st.button("Proceed"):
    if option == "Exercise Form Estimator":
        st.switch_page("pages/form_estimator.py")
    elif option == "Nutrition Recommendation Model":
        st.switch_page("pages/nutrition_model.py")

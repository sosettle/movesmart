# streamlit app
import streamlit as st
import numpy as np

# Feature names
features = ["Affordability", "Safety", "Job Opportunities", "Health"]

st.title("MoveSmart: City Recommendation")

# User input sliders
st.header("Set your preferences")

user_weights = []
for feature in features:
    w = st.slider(feature, 0.0, 1.0, 0.5)  # min=0, max=1, default=0.5
    user_weights.append(w)
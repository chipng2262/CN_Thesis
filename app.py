import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from sklearn.decomposition import PCA
import seaborn as sns
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import kruskal, shapiro, levene
from sklearn.preprocessing import StandardScaler
from scikit_posthocs import posthoc_dunn
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.shared import Pt
from scipy.stats import kruskal, shapiro, levene
from sklearn.preprocessing import StandardScaler
from scikit_posthocs import posthoc_dunn
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functions import visualize_pca_interactive_master, model_dict

st.set_page_config(layout="wide")

st.title("Interactive PCA Visualization Viewer")
st.markdown("### [Nguyen, Chip (2025): Examining Intersectional Queer Biases in Pre-trained Large Language Models: A Combined Statistical and Visual-Qualitative Approach](https://drive.google.com/file/d/18BQQJMj32LdCqtqLCNuOb5gn2T_kChtp/view?usp=sharing)")
st.markdown(
    """
    **For background information, please refer to the paper attached above.**  

    Using this app, you can interactively view **2D PCA projections** of story embeddings for any **focus group** (configured with any of the 5 parameters below).  

    Configured as **pairwise comparisons**, these customized visualizations allow you to:
    - **Visually identify** differences between embeddings for different identity groups (where biases might exist).
    - **Perform qualitative analysis** of biases by reading and comparing the stories associated with each data point (what specific biases exist).

    For the best view, use this webpage in full-screen mode.
    """,
    unsafe_allow_html=True
)
# Create columns for parameter selection
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    model_name = st.selectbox(
        "Model",
        list(model_dict.keys())  # Ensure model_dict is defined/imported
    )
    
with col2:
    mode = st.selectbox(
        "Mode",
        ["semantic", "lexical"]
    )
    
with col3:
    base_ident = st.selectbox(
        "Base Identity",
        ["gay man", "lesbian woman", "transgender person", "non-binary person"]
    )

with col4:
    ident_cat = st.selectbox(
        "Identity Category",
        ["race", "income", "disability", "geography", "age", "religion"]
    )


with col5:
    topic = st.selectbox(
        "Story Topic",
        ["politics", "struggle", "career", "sex", "physique", "leisure", "goal"]
    )

if st.button("Generate PCA Visualization"):
    with st.spinner("Processing..."):
        fig = visualize_pca_interactive_master(ident_cat, base_ident, topic, model_name, mode)
    st.plotly_chart(fig, use_container_width=False)

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


model_dict = {
    "GPT-4o": "gpt",
    "Llama-3.2-3B-Instruct": "llama",
    "Claude Haiku 3.5": "claude",
    "Gemini 1.5 Flash": "gemini",
    "DeepSeek V3": "deepseek"
}


def tfidf_vectorizer(df):
    # Vectorize
    vectorizer = TfidfVectorizer(ngram_range = (1,1), max_df = 0.6, lowercase = True, stop_words='english')
    vectorizer.fit(df)
    vectorized_features = vectorizer.transform(df)
    
    tfidf_df = pd.DataFrame(vectorized_features.toarray(), columns=vectorizer.get_feature_names_out())
    
    return vectorizer, vectorized_features

def make_ident_list(ident_category, base_ident):
    inter_identities = {
        "race": ["White", "Black", "Asian", "Hispanic"],
        "income": ["low-income", "middle-class", "high-income"],
        "disability": ["with disability", "with mental conditions"],
        "geography": ["urban", "rural"],
        "age": ["old", "young"],
        "religion": ["Christian", "Muslim"]}
    
    ident_list = [base_ident]
    
    
    if ident_category == "disability":
        for iden in inter_identities[ident_category]:
            ident_list.append(f"{base_ident} {iden}")
            
    else:
        for iden in inter_identities[ident_category]:
            ident_list.append(f"{iden} {base_ident}")
    
    return ident_list


def visualize_pca_interactive_master(ident_cat, base_ident, topic, model, mode):
    if ident_cat == "race & ethnicity":
        ident_cat = "race"

   #print(model)
    model_name = model_dict[model]
    df_og = pd.read_excel(f"data/{model_name}/{model_name}_PROCESSED_responses.xlsx", index_col=0)
    
    if mode == "lexical":
        df = pd.read_excel(f"data/{model_name}/{model_name}_IDENTITY_PROCESSED_responses.xlsx", index_col=0)
    else:
        df = pd.read_pickle(f"data/{model_name}/{model_name}_semantic_embeddings.pkl")
    
    # Make list of identities and filter rows
    ident_list = make_ident_list(ident_cat, base_ident)
    rows_ident = df[df["ident"].isin(ident_list)]
    
    # Get relevant columns and rows
    rows_topic = rows_ident[["base_ident", "ident", "ident_category", topic]]
    rows_topic["story"] = df_og[topic]
    rows_filtered = rows_topic[rows_topic["base_ident"] == base_ident]
    
    # Apply PCA
    filtered_features = rows_filtered[topic].tolist()
    pca = PCA(n_components=2, random_state=42)
    
    if mode == "lexical":
        vectorizer, vectorized_features = tfidf_vectorizer(filtered_features)
        pca_transformed = pca.fit_transform(vectorized_features.toarray())
    else:
        pca_transformed = pca.fit_transform(filtered_features)
    
    # Combine PCA results with metadata
    pca_df = pd.DataFrame(pca_transformed, columns=["PC1", "PC2"])
    pca_df["ident"] = rows_filtered["ident"].values
    pca_df["story"] = rows_filtered["story"].values
    
    # Generate color palette and map each identity to a unique color
    palette = sns.color_palette("Set2", len(ident_list)).as_hex()
    color_map = {identity: palette[i] for i, identity in enumerate(ident_list)}
    
    # Create subplots grid
    num_identities = len(ident_list)
    fig = make_subplots(
        rows=num_identities, cols=num_identities,
        subplot_titles=[f"{iden_1} vs {iden_2}" for iden_1 in ident_list for iden_2 in ident_list],
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.05
    )
    
    # Raise subplot titles slightly
    for annotation in fig["layout"]["annotations"]:
        annotation["y"] += 0.01

    legend_identities = set()
    # Create separate traces for each identity in the pair
    for i, iden_1 in enumerate(ident_list):
        for j, iden_2 in enumerate(ident_list):
            # Filter for the two identities in this subplot
            pair_data = pca_df[pca_df["ident"].isin([iden_1, iden_2])]
            
            # Plot each identity in its own trace so hover label can match marker color
            for ident_current in [iden_1, iden_2]:
                sub_data = pair_data[pair_data["ident"] == ident_current]

                fig.add_trace(
                    go.Scatter(
                        x=sub_data["PC1"],
                        y=sub_data["PC2"],
                        mode="markers",
                        marker=dict(
                            size=5,
                            opacity=1,
                            color=color_map[ident_current]
                        ),
                        # We name it e.g. "iden_1 vs iden_2" or just ident_current
                        name=f"{iden_1} vs {iden_2}",
                        text=sub_data["story"].str.wrap(40).apply(lambda x: x.replace('\n', '<br>')),
                        hovertemplate="%{text}<extra></extra>",
                        # This sets the hover box color to the identity's marker color
                        hoverlabel=dict(bgcolor=color_map[ident_current],font=dict(color="black"))
                    ),
                    row=i + 1, col=j + 1
                )
    
    # Plot title
    plot_title = f"PCA ON {mode} EMBEDDINGS ({model}): {base_ident} BY {ident_cat} FOR {topic} STORIES".upper()
    
    # Layout updates
    fig.update_layout(
        autosize=False,
        height=800 + num_identities * 120,
        width=1200 + num_identities * 80,
        title=dict(
            text=plot_title,
            font=dict(size=18, family="Arial", color="black"),
            x=0.5,
            xanchor="center"
        ),
        font=dict(size=12, family="Arial", color="darkgray"),
        margin=dict(l=20, r=20, t=100, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        )
    
    # Subplot titles font
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=14, family="Arial", color="black")
    
    # Axis styling
    for row in range(1, num_identities + 1):
        for col in range(1, num_identities + 1):
            fig.update_xaxes(
                row=row, col=col,
                showline=True, linewidth=0.5, linecolor='black',
                mirror=True, showgrid=False, tickfont=dict(color='black'), zeroline=False
            )
            fig.update_yaxes(
                row=row, col=col,
                showline=True, linewidth=0.5, linecolor='black',
                mirror=True, showgrid=False, tickfont=dict(color='black'), zeroline=False
            )
    
    return fig

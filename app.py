import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import (
    load_disney_data,
    train_sentiment_predictor,
    predict_single_review,
    BRANCH_COLORS,
    SENTIMENT_COLORS
)
from src.visualizer import (
    build_2d_branch_ratings,
    build_polarity_subjectivity_density,
    build_3d_sentiment_space,
    build_3d_visitor_globe,
    build_4d_yearly_sentiment_animation
)

st.set_page_config(
    page_title="Disneyland Reviews Sentiment NLP & 4D Explorer",
    page_icon="🏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Disney Magic CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    .disney-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_disney_data("data/disneyland_reviews.csv")

df = get_data()

@st.cache_resource
def get_model():
    return train_sentiment_predictor(df)

model_pipeline = get_model()

# Sidebar
st.sidebar.title("🏰 Magic Kingdom Telemetry")
st.sidebar.markdown("Visitor Sentiment Intelligence, 3D Morphospaces & 4D Spatio-Temporal Decade Dynamics.")

branch_filter = st.sidebar.multiselect("Park Branch", options=sorted(df["Branch"].unique()), default=sorted(df["Branch"].unique()))
year_filter = st.sidebar.slider("Observation Window", int(df["Year"].min()), int(df["Year"].max()), (int(df["Year"].min()), int(df["Year"].max())))
rating_filter = st.sidebar.multiselect("Star Rating", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

filtered_df = df[
    (df["Branch"].isin(branch_filter)) &
    (df["Year"] >= year_filter[0]) &
    (df["Year"] <= year_filter[1]) &
    (df["Rating"].isin(rating_filter))
]

# Header KPIs
col_title, col_kpis = st.columns([2, 3])
with col_title:
    st.title("Disneyland Sentiment NLP Engine")
    st.caption("Cross-branch visitor satisfaction benchmarking, 3D sentiment spaces, and 4D decade evolution.")

tot_rev = len(filtered_df)
avg_rating = filtered_df["Rating"].mean() if tot_rev > 0 else 0
pos_pct = ((filtered_df["Sentiment"] == "Positive").sum() / max(1, tot_rev)) * 100
avg_words = filtered_df["Word_Count"].mean() if tot_rev > 0 else 0

with col_kpis:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="disney-card">
            <span style="color: #94a3b8; font-size: 11px;">REVIEWS ANALYZED</span>
            <h3 style="margin: 0; color: #38bdf8;">{tot_rev:,}</h3>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="disney-card">
            <span style="color: #94a3b8; font-size: 11px;">SATISFACTION RATING</span>
            <h3 style="margin: 0; color: #f59e0b;">⭐ {avg_rating:.2f} / 5.0</h3>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="disney-card">
            <span style="color: #94a3b8; font-size: 11px;">POSITIVE SENTIMENT</span>
            <h3 style="margin: 0; color: #10b981;">{pos_pct:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="disney-card">
            <span style="color: #94a3b8; font-size: 11px;">AVG WORDS / REVIEW</span>
            <h3 style="margin: 0; color: #ec4899;">{avg_words:.0f} words</h3>
        </div>
        """, unsafe_allow_html=True)

# Tabs
tab_2d, tab_3d, tab_4d, tab_nlp = st.tabs([
    "📊 2D Branch Sentiment & Ratings",
    "🌐 3D Sentiment Space & Global Globe",
    "⏳ 4D Decade Sentiment Evolution",
    "✨ Live Review NLP Rating Predictor"
])

# TAB 1: 2D
with tab_2d:
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.plotly_chart(build_2d_branch_ratings(filtered_df), use_container_width=True)
    with col_b2:
        st.plotly_chart(build_polarity_subjectivity_density(filtered_df), use_container_width=True)

# TAB 2: 3D
with tab_3d:
    st.subheader("3D Sentiment Morphospace (Polarity × Subjectivity × Review Length)")
    st.markdown("Visualizes individual visitor reviews in 3D coordinate space colored by rating score.")
    sample_3d = filtered_df.sample(min(2500, len(filtered_df)), random_state=42)
    st.plotly_chart(build_3d_sentiment_space(sample_3d), use_container_width=True)
    
    st.divider()
    st.subheader("3D Global Visitor Origin & Park Rating Globe")
    st.plotly_chart(build_3d_visitor_globe(filtered_df), use_container_width=True)

# TAB 3: 4D
with tab_4d:
    st.subheader("4D Decade Experience Evolution (Temporal Shift from 2011 to 2019)")
    st.markdown("Press **Play** to observe how visitor sentiment, review detail, and park satisfaction shifted over the 10-year timeline.")
    st.plotly_chart(build_4d_yearly_sentiment_animation(filtered_df), use_container_width=True)

# TAB 4: NLP
with tab_nlp:
    st.subheader("✨ Real-Time Disneyland Review Sentiment & Rating AI")
    st.markdown("Type or paste any theme park review to evaluate instant sentiment polarity, subjectivity, and machine learning predicted star rating.")
    
    review_input = st.text_area(
        "Enter Visitor Review",
        value="The fireworks and castle projection show were utterly breathtaking! The staff was super friendly, though the queue for Space Mountain was nearly two hours long. Overall an unforgettable experience!",
        height=120
    )
    
    if review_input:
        res = predict_single_review(review_input, model_pipeline)
        
        st.markdown("---")
        st.markdown("### NLP Diagnostic Assessment")
        
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.markdown(f"<div class='disney-card'><span style='color: #94a3b8;'>PREDICTED RATING</span><h2 style='margin: 0; color: #f59e0b;'>⭐ {res['rating']} / 5</h2></div>", unsafe_allow_html=True)
        with p2:
            sent_col = SENTIMENT_COLORS.get(res['sentiment'], '#38bdf8')
            st.markdown(f"<div class='disney-card'><span style='color: #94a3b8;'>OVERALL SENTIMENT</span><h2 style='margin: 0; color: {sent_col};'>{res['sentiment']}</h2></div>", unsafe_allow_html=True)
        with p3:
            st.markdown(f"<div class='disney-card'><span style='color: #94a3b8;'>POLARITY SCORE</span><h2 style='margin: 0; color: #38bdf8;'>{res['polarity']:+.2f}</h2></div>", unsafe_allow_html=True)
        with p4:
            st.markdown(f"<div class='disney-card'><span style='color: #94a3b8;'>WORD COUNT</span><h2 style='margin: 0; color: #a855f7;'>{res['word_count']} words</h2></div>", unsafe_allow_html=True)

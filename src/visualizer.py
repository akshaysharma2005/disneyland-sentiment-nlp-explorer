import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from .data_loader import BRANCH_COLORS, SENTIMENT_COLORS

def build_2d_branch_ratings(df):
    """Comparative bar chart of star ratings across branches."""
    pivot = df.groupby(["Branch", "Rating"]).size().unstack(fill_value=0).reset_index()
    
    fig = go.Figure()
    rating_colors = ["#f43f5e", "#fb7185", "#f59e0b", "#38bdf8", "#10b981"]
    
    for r in range(1, 6):
        if r in pivot.columns:
            fig.add_trace(go.Bar(
                x=pivot["Branch"],
                y=pivot[r],
                name=f"{r} Star{'s' if r > 1 else ''}",
                marker_color=rating_colors[r-1]
            ))
            
    fig.update_layout(
        title="Visitor Star Rating Distribution by Theme Park Branch",
        barmode="stack",
        yaxis_title="Review Count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def build_polarity_subjectivity_density(df):
    """2D Scatter Density of Polarity vs Subjectivity."""
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample,
        x="Polarity",
        y="Subjectivity",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        opacity=0.6,
        labels={"Polarity": "Sentiment Polarity (-1 Negative, +1 Positive)", "Subjectivity": "Subjectivity Index (0-1)"}
    )
    fig.update_layout(
        title="Visitor Sentiment Polarity vs. Subjectivity Spectrum",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def build_3d_sentiment_space(df_sample):
    """
    3D Sentiment Morphospace:
    X: Polarity (-1 to 1)
    Y: Subjectivity (0 to 1)
    Z: Word Count
    Color: Star Rating (1 to 5)
    """
    fig = px.scatter_3d(
        df_sample,
        x="Polarity",
        y="Subjectivity",
        z="Word_Count",
        color="Rating",
        hover_name="Branch",
        hover_data={"Rating": True, "Reviewer_Location": True, "Sentiment": True},
        color_continuous_scale="Turbo",
        range_x=[-1.0, 1.0],
        range_y=[0.0, 1.0],
        opacity=0.75,
        labels={"Polarity": "Sentiment Polarity", "Subjectivity": "Subjectivity", "Word_Count": "Review Words"}
    )

    fig.update_layout(
        title="3D Sentiment Space: Polarity × Subjectivity × Review Length (Color: Rating)",
        scene=dict(
            xaxis=dict(title="Sentiment Polarity", backgroundcolor="#0f172a", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Subjectivity", backgroundcolor="#0f172a", gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(title="Review Word Count", backgroundcolor="#0f172a", gridcolor="rgba(255,255,255,0.1)"),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.2))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=620,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

def build_3d_visitor_globe(df):
    """3D Orthographic globe plotting international reviewer locations."""
    loc_agg = df.groupby("Reviewer_Location").agg(
        Review_Count=("Rating", "count"),
        Avg_Rating=("Rating", "mean")
    ).reset_index()
    loc_agg = loc_agg[loc_agg["Review_Count"] >= 5]

    fig = px.choropleth(
        loc_agg,
        locations="Reviewer_Location",
        locationmode="country names",
        color="Avg_Rating",
        hover_name="Reviewer_Location",
        hover_data={"Review_Count": True, "Avg_Rating": ":.2f"},
        projection="orthographic",
        color_continuous_scale="Plasma",
        title="3D Global Visitor Origin & Park Rating Geography"
    )

    fig.update_geos(
        showcoastlines=True, coastlinecolor="rgba(255,255,255,0.3)",
        showland=True, landcolor="#1e293b",
        showocean=True, oceancolor="#0b132b",
        bgcolor="rgba(0,0,0,0)"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=580,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

def build_4d_yearly_sentiment_animation(df):
    """
    4D Spatio-Temporal Animation:
    - Dim 1 (X): Polarity
    - Dim 2 (Y): Word Count
    - Dim 3 (Z): Rating
    - Dim 4 (Color): Branch
    - Dim 5 (Size): Subjectivity
    - Animation Frame: Year (2011 to 2019)
    """
    df_anim = df.sort_values(by="Year").copy()
    df_anim["Year_Str"] = df_anim["Year"].astype(str)

    fig = px.scatter_3d(
        df_anim,
        x="Polarity",
        y="Word_Count",
        z="Rating",
        color="Branch",
        size="Subjectivity",
        size_max=14,
        animation_frame="Year_Str",
        animation_group="Review_ID",
        hover_name="Branch",
        hover_data={"Rating": True, "Sentiment": True, "Reviewer_Location": True},
        color_discrete_map=BRANCH_COLORS,
        range_x=[-1.05, 1.05],
        range_y=[5, 600],
        range_z=[0.5, 5.5],
        labels={
            "Polarity": "Sentiment Polarity",
            "Word_Count": "Review Words",
            "Rating": "Star Rating (1-5)"
        }
    )

    fig.update_layout(
        title="4D Decade Sentiment Evolution (Watch Theme Park Shifts from 2011 to 2019)",
        scene=dict(
            xaxis=dict(title="Sentiment Polarity", backgroundcolor="#0b132b", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Review Words", backgroundcolor="#0b132b", gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(title="Star Rating", backgroundcolor="#0b132b", gridcolor="rgba(255,255,255,0.1)"),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.2))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=620,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

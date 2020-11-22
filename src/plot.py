import numpy as np
import pandas as pd
from plotly import express as px
import plotly.graph_objects as go

from src.objectives import saddle


def plot_optimization_history(path):
    points_df = pd.read_csv(path)
    fig = px.scatter(
        points_df,
        x="x",
        y="value",
        animation_frame="generation",
        animation_group="value",
        range_x=[-1, 3],
        range_y=[0, 5],
    )

    x = np.linspace(-1, 3, 1000)
    fig.add_trace(go.Scatter(x=x, y=np.vectorize(saddle)(x)))

    fig.show()

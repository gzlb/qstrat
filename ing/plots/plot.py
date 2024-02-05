import plotly.graph_objects as go
import plotly.express as px

import numpy as np


def plot_series_plotly(*args: np.ndarray, labels: list = None, title: str, xaxis_title: str, yaxis_title: str) -> None:
    """
    Plots NumPy arrays using Plotly.

    Parameters:
    - *args (np.ndarray): Variable number of NumPy arrays to be plotted.
    - labels (list): List of labels for the legend (default: None).
    - title (str): Title of the plot (default: 'Series Plot').
    - xaxis_title (str): Title of the x-axis (default: 'Time').
    - yaxis_title (str): Title of the y-axis (default: 'Values').

    Returns:
    - None
    """
    try:
        # Create Plotly figure
        fig = go.Figure()

        # If labels are not provided, generate default labels
        if labels is None:
            labels = [f'Series {i+1}' for i in range(len(args))]

        # Add traces for the provided series
        for series, label in zip(args, labels):
            fig.add_trace(go.Scatter(x=np.arange(len(series)), y=series, mode='lines', name=label))

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
        )

        # Show the plot
        fig.show()

    except Exception as e:
        print(f"Error plotting series using Plotly: {e}")

def plot_density(values, title, xlabel):
    fig = px.histogram(x=values, nbins=30, title=title, labels={'x': xlabel, 'y': 'Density'})
    fig.update_layout(showlegend=False)
    fig.show()

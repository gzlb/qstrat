import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def plot_series_plotly(
    *args: np.ndarray,
    labels: list = None,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    truncate: bool = False,
) -> None:
    """
    Plots NumPy arrays using Plotly.

    Parameters:
    - *args (np.ndarray): Variable number of NumPy arrays to be plotted.
    - labels (list): List of labels for the legend (default: None).
    - title (str): Title of the plot (default: 'Series Plot').
    - xaxis_title (str): Title of the x-axis (default: 'Time').
    - yaxis_title (str): Title of the y-axis (default: 'Values').
    - truncate (bool): Option to truncate data to the shortest array (default: False).

    Returns:
    - None
    """
    try:
        if truncate:
            min_length = min(len(series) for series in args)
            args = [series[:min_length] for series in args]
            if labels:
                labels = labels[: len(args)]

        fig = go.Figure()

        if labels is None:
            labels = [f"Series {i+1}" for i in range(len(args))]

        for series, label in zip(args, labels):
            fig.add_trace(
                go.Scatter(x=np.arange(len(series)), y=series, mode="lines", name=label)
            )

        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend=dict(x=0, y=1, traceorder="normal", orientation="h"),
        )

        fig.show()

    except Exception as e:
        print(f"Error plotting series using Plotly: {e}")


def plot_density(values, title, xlabel):
    fig = px.histogram(
        x=values, nbins=30, title=title, labels={"x": xlabel, "y": "Density"}
    )
    fig.update_layout(showlegend=False)
    fig.show()

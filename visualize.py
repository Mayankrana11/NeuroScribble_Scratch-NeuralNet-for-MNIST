import networkx as nx
from pyvis.network import Network
import numpy as np
from model import NeuralNet
from utils import load
import webbrowser
import os


# CONFIG

LAYER_SIZES = [784, 128, 64, 10]

# Limit edges so browser stays smooth
MAX_EDGES_PER_NEURON = 6

OUTPUT_HTML = "network.html"
X_SPACING = 800
Y_SPACING = 18



def build_graph(model):
    G = nx.DiGraph()

    # Add all neurons
    for layer, size in enumerate(LAYER_SIZES):
        for i in range(size):
            G.add_node(
                f"L{layer}_N{i}",
                layer=layer,
                label=str(i) if layer == len(LAYER_SIZES) - 1 else ""
            )

    # Collect weight matrices
    weights = [layer.W for layer in model.layers if hasattr(layer, "W")]

    # Add edges (top-k strongest per neuron)
    for l, W in enumerate(weights):
        for i in range(W.shape[0]):
            idx = np.argsort(np.abs(W[i]))[-MAX_EDGES_PER_NEURON:]
            for j in idx:
                w = W[i, j]
                G.add_edge(
                    f"L{l}_N{i}",
                    f"L{l+1}_N{j}",
                    color="green" if w > 0 else "red",
                    width=1 + min(4, abs(w))
                )

    return G


def visualize(G):
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#0e0e0e",
        font_color="white",
        directed=True
    )

    net.toggle_physics(False)

    # X positions per layer
    layer_x = {i: i * X_SPACING for i in range(len(LAYER_SIZES))}

    # Center layers vertically (hourglass)
    for layer, size in enumerate(LAYER_SIZES):
        y_start = -(size / 2) * Y_SPACING
        for i in range(size):
            net.add_node(
                f"L{layer}_N{i}",
                x=layer_x[layer],
                y=y_start + i * Y_SPACING,
                fixed=True,
                physics=False,
                shape="dot",
                size=6 if layer == 0 else (10 if layer < 3 else 14),
                color="#66ff66" if layer < 3 else "#66ccff"
            )

    # Add edges
    for u, v, data in G.edges(data=True):
        net.add_edge(
            u, v,
            color=data["color"],
            width=data["width"],
            smooth=True
        )

    # VALID JSON OPTIONS (IMPORTANT FIX)
    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "zoomView": true,
        "dragView": true,
        "navigationButtons": true,
        "keyboard": true
      },
      "physics": {
        "enabled": false
      },
      "edges": {
        "smooth": {
          "type": "dynamic"
        }
      }
    }
    """)

    net.write_html(OUTPUT_HTML)
    webbrowser.open("file://" + os.path.realpath(OUTPUT_HTML))


model = NeuralNet()
load(model)

G = build_graph(model)
visualize(G)

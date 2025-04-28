import torch
from torchview import draw_graph
from models.cnn_binary import Model

model = Model()

example_input = torch.randn(1, 50, 64, 64)

# Generate the visualization
graph = draw_graph(model, input_data=example_input, expand_nested=True)

# Render and display the graph
graph.visual_graph.attr(rankdir="TB")
graph.visual_graph.render("model_visualization_lr", format="svg")
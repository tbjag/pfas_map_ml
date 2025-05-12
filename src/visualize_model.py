import torch
from torchview import draw_graph
from models.CNNs.cnn_binary import Model
from models.CNNs.cnn3_binary import Model as Model3
from models.UNets.unet1 import UNet1
from models.ResNets.resnet import ResNetModel

model = ResNetModel()

example_input = torch.randn(1, 54, 64, 64)

# Generate the visualization
graph = draw_graph(model, input_data=example_input, expand_nested=True)

# Render and display the graph
graph.visual_graph.attr(rankdir="TB")
graph.visual_graph.render("visualizations/ResNet", format="png")
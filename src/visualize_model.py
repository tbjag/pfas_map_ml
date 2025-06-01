import torch
from torchview import draw_graph
from models.CNNs.test_lightning2 import Model as CNN2
from models.UNets.unet1 import UNet1
from models.ResNets.resnet2 import ResNetModel as ResNet2

model = CNN2()

example_input = torch.randn(1, 52, 64, 64)

# Generate the visualization
graph = draw_graph(model, input_data=example_input, expand_nested=True)

# Render and display the graph
graph.visual_graph.attr(rankdir="TB")
graph.visual_graph.render("visualizations/CNN2_LR", format="png")
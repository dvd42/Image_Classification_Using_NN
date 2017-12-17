import torch
from tensorboardX import graph_onnx
from tensorboardX import SummaryWriter


def generate_graph(model,input):
    with SummaryWriter() as w:
        model.train()
        torch.onnx.export(model, input, "Architecture/Model.onnx", verbose=True)
        w.add_graph_onnx("Architecture/Model.onnx")




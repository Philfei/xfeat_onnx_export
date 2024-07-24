import argparse
import os.path
from typing import List
import torch
import cv2
import numpy as np
# import tensor

from module_onnx import model
# from modules import model

def parse_input(x):
    if len(x.shape) == 3:
        x = x[None, ...]
        # print("x.shape:", x.shape)

    if isinstance(x, np.ndarray):
        # x = x / 255.0
        x = torch.tensor(x / 255.0, dtype=torch.float).permute(0,3,1,2)
    # print("x.shape:", x.shape)
    return x


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)[None]

def export_onnx(
    xfeat_path=None,
    img0_path="assets/rs_Color.png",
    img1_path="assets/rs_Infrared.png",
    dynamic=False,
    dense=False,
    end2end=False,
    top_k=None,
):
    # Sample images for tracing
    image0 = cv2.imread(img0_path)
    image1 = cv2.imread(img1_path)

    input_tensor = torch.randn(1, 3, 480, 640)

    # Models
    device = torch.device('cpu')
    xfeatModel = model.XFeatModel()
    xfeatModel.load_state_dict(torch.load(xfeat_path))
    # xfeatModel = torch.jit.script(xfeatModel)
    # xfeatModel = torch.jit.trace(xfeatModel, parse_input(image0), strict=False)
    # xfeatModel.eval()
    xfeatModel.to(device).eval()
    xfeatModel.forward = xfeatModel.detectAndComputeDense
    # torch.tensor.to(device)
    # -----------------
    # Export Extractor
    # -----------------
    dynamic_axes = {
        "images": {1: "channel", 2: "height", 3: "width"},
        "scales": {0: "num_keypoints"},
    }

    # xfeatmodel.forward = xfeat.detectAndComputeDense
    output_names = ["keypoints", "descriptors", "scales"]
    
    output_path = 'weights/xfeat_dense.onnx'
    # output_names = ["descriptors", "keypoints", "heatmap"]
    # Export model
    torch.onnx.export(
        xfeatModel,
        # input_tensor,
        # numpy_image_to_torch(image0),
        parse_input(image0),
        output_path,
        verbose=False,
        input_names=["images"],
        output_names=output_names,
        opset_version=17,
        dynamic_axes=dynamic_axes,
    )

if __name__ == "__main__":
    export_onnx(xfeat_path="weights/xfeat.pt", dynamic=True, dense=False, end2end=False, top_k=800)
import os
import simplecv as sc
import torch
from PIL import Image
import numpy as np
from module import register_model
config_path='isaid.factseg'
ckpt_path=os.path.join("log", "factseg50.pth")
patch_size = 896
temp_filepath = "temp/temp.jpg"
img = Image.open(temp_filepath)
image = np.array(img)
image = torch.Tensor(image)
model, global_step = sc.api.infer_tool.build_and_load_from_file(config_path, ckpt_path)
onnx_path = "my_model.onnx"
torch.onnx.export(model, image, onnx_path, input_names=["input"], output_names=["output"], opset_version=12)
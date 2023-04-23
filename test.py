import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

# model = SimpleModel()
# dummy_input = torch.randn(1, 3, 224, 224)
# onnx_model_path = "simple_model.onnx"

model = SimpleModel()
dummy_input = torch.randn(1, 3, 224, 224)
onnx_model_path = "simple_model_dynamic_input.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=12,
    dynamic_axes={"input": {0: "batch_size", 2: "height", 3: "width"},
                  "output": {0: "batch_size", 2: "height", 3: "width"}}
)

import onnxruntime as rt
import numpy as np
from PIL import Image

def load_image(image_path):
    img = Image.open(image_path) # W * H
    img = np.array(img).astype(np.float32) # H * W * C
    img = np.transpose(img, (2, 0, 1)) 
    img = np.expand_dims(img, axis=0)
    return img

def run_inference_onnx(onnx_model_path, input_image):
    sess = rt.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output = sess.run([output_name], {input_name: input_image})[0]
    return output

input_image_path = "temp/temp.jpg"
input_image = load_image(input_image_path)
output = run_inference_onnx(onnx_model_path, input_image)
import pdb
pdb.set_trace()
# Process and display the output as needed

import os
import simplecv as sc
from PIL import Image
import numpy as np
from module import register_model
from simplecv.api.preprocess import comm, segm
import torch
import copy
import torch.nn as nn
from segmslidingwininference import SegmSlidingWinInference

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        config_path='isaid.factseg'
        ckpt_path=os.path.join("log", "factseg50.pth")
        self.model, self.global_step = sc.api.infer_tool.build_and_load_from_file(config_path, ckpt_path)
        self.segm_helper = SegmSlidingWinInference()
        self.CLASSES_NEEDED = ['small_Vehicle', 'large_Vehicle', 'storage_tank', 'plane']
        self.NEW_CLASSES = [
            'background',
            'ship',
            'storage_tank',
            'baseball_diamond',
            'tennis_court',
            'basketball_court',
            'ground_Track_Field',
            'bridge',
            'large_Vehicle',
            'small_Vehicle',
            'helicopter',
            'swimming_pool',
            'roundabout',
            'soccer_ball_field',
            'plane',
            'harbor'
        ]
        self.patch_size = 896
        self.image_trans = comm.Compose([
            segm.ToTensor(True),
            comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)),
            comm.CustomOp(lambda x: x.unsqueeze(0))
        ])

    def forward(self, x):
        h, w = image.shape[:2]
        seg_helper = self.segm_helper.patch((h, w), patch_size=(self.patch_size, self.patch_size), stride=512, transforms=self.image_trans)
        out = seg_helper.forward(self.model, image, size_divisor=32)
        out_ori = out.argmax(dim=1).numpy()[0]
        out = np.zeros_like(out_ori)
        
        for class_needed in self.CLASSES_NEEDED:
            out[out_ori == self.NEW_CLASSES.index(class_needed)] = self.NEW_CLASSES.index(class_needed)
        out = torch.Tensor(out)
        return out

model = SimpleModel()
temp_filepath = "temp/temp.jpg"
img = Image.open(temp_filepath)
image = np.array(img)
image = torch.Tensor(image)
onnx_path = "my_model.onnx"
torch.onnx.export(
    model,
    image,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=12,
    dynamic_axes={"input": {0: "height", 1: "width", 2: "channel"},
                  "output": {0: "height", 1: "width", 2: "channel"}}
)


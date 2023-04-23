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
    output = sess.run(None, {"input": input_image})
    
    return output

input_image_path = "temp/temp.jpg"
input_image = load_image(input_image_path)
onnx_model_path = "my_model.onnx"
output = run_inference_onnx(onnx_model_path, input_image)
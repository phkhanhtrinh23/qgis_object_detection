# from ossaudiodev import control_labels
# from tqdm import tqdm
# from scipy.io import savemat
# from ade import CLASSES
# import glob
# import os
# from io import BytesIO
# import tarfile
# import tempfile
# from six.moves import urllib

# from matplotlib import gridspec
# from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = "ImageTensor:0"
    OUTPUT_TENSOR_NAME = "SemanticPredictions:0"
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = "frozen_inference_graph"

    def __init__(self, frozen_graph_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        f = open(frozen_graph_path, "rb")
        graph_def = tf.compat.v1.GraphDef.FromString(
            f.read()
        )
        f.close()
        
        if graph_def is None:
            raise RuntimeError("Cannot find inference graph in tar archive.")

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="")

        self.sess = tf.compat.v1.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]},
        )
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


# def create_pascal_label_colormap():
#     """Creates a label colormap used in PASCAL VOC segmentation benchmark.

#     Returns:
#       A Colormap for visualizing segmentation results.
#     """
#     colormap = np.zeros((256, 3), dtype=int)
#     ind = np.arange(256, dtype=int)

#     for shift in reversed(range(8)):
#         for channel in range(3):
#             colormap[:, channel] |= ((ind >> channel) & 1) << shift
#         ind >>= 3

#     return colormap


# def label_to_color_image(label):
#     """Adds color defined by the dataset colormap to the label.

#     Args:
#       label: A 2D array with integer type, storing the segmentation label.

#     Returns:
#       result: A 2D array with floating type. The element of the array
#         is the color indexed by the corresponding element in the input label
#         to the PASCAL color map.

#     Raises:
#       ValueError: If label is not of rank 2 or its value is larger than color
#         map maximum entry.
#     """
#     if label.ndim != 2:
#         raise ValueError("Expect 2-D input label")

#     colormap = create_pascal_label_colormap()

#     if np.max(label) >= len(colormap):
#         raise ValueError("label value too large.")

#     return colormap[label]


def get_largest_object_polygon(segment_img, origin_x, origin_y, im_width, im_height, IDX2CLASS, max_n_objects=2):
    vals = list(set(segment_img.flatten()))
    vals = [x for x in vals if x != 0]
    if len(vals) == 0:
        return [{
            "label": "",
            "shape_type": "",
            "points": [],
        }]

    # find largest object

    all_labels = []
    all_contours = []
    all_contours_areas = []
    for val in vals:
        mask = np.zeros_like(segment_img, dtype=np.uint8)
        # smooth image
        mask[segment_img == val] = 255
        kernel = np.ones((7, 7), np.float32) / 49
        mask = cv2.filter2D(mask, -1, kernel)
        mask[mask >= 127] = 255
        mask[mask < 127] = 0
        mask[mask == 255] = 1
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # end smooth image

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = [x for x in contours if cv2.contourArea(x) > 20]
        contours_areas = [cv2.contourArea(x) for x in contours]
        all_contours.extend(contours)
        all_contours_areas.extend(contours_areas)
        all_labels.extend([val] * len(contours))

    boxes = []
    # all_contours_areas.sort(reverse=True)
    for i in range(max_n_objects):
        largest_ind = np.argmax(all_contours_areas)
        contour = all_contours[largest_ind]

        # smooth contour
        contour = contour.reshape(-1, 2)
        new_contour = [contour[0]]
        for i in range(1, len(contour)):
            prev_pt = new_contour[-1]
            dist = np.sqrt(np.sum((contour[i] - prev_pt) ** 2))
            if dist > 10:
                new_contour.append(contour[i])
        if len(new_contour) >= 3:
            contour = new_contour
        box = {
            "label": IDX2CLASS[all_labels[largest_ind]],
            "shape_type": "polygon",
            "points": [
                [
                    int(origin_x + int(x * im_width / mask.shape[1])),
                    int(origin_y + int(y * im_height / mask.shape[0])),
                ]
                for x, y in contour
            ],
        }

        boxes.append(box)

        # remove the previous largest contour_areas
        all_contours_areas[largest_ind] = -1

    return boxes


# if __name__ == "__main__":
#     # MODEL_NAME = 'xception65_ade20k_train'

#     # _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
#     # _MODEL_URLS = {
#     #     'xception65_ade20k_train': 'deeplabv3_xception_ade20k_train_2018_05_29.tar.gz'
#     # }
#     # _TARBALL_NAME = 'deeplab_model.tar.gz'
#     # model_dir = 'model/'
#     # tf.gfile.MakeDirs(model_dir)

#     # download_path = os.path.join(model_dir, _TARBALL_NAME)
#     # print('downloading model, this might take a while...')
#     # urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)
#     # print('download completed! loading DeepLab model...')

#     MODEL = DeepLabModel(
#         "deeplabv3_mnv2_ade20k_train_2018_12_03/frozen_inference_graph.pb"
#     )
#     print("model loaded successfully!")

#     files = glob.glob("data/images/*")
#     CONSIDER_CLASSES = {
#         "person, individual, someone, somebody, mortal, soul": 1,
#         "grass": 2,
#         "tree": 3,
#         "mountain, mount": 4,
#         "plant, flora, plant life": 3,
#         "water": 5,
#         "sea": 6,
#         "rock, stone": 7,
#         "sand": 8,
#         "river": 5,
#         "flower": 3,
#         "palm, palm tree": 9,
#         "land, ground, soil": 10,
#         "waterfall, falls": 11,
#         "building, edifice": 12,
#         "house": 12,
#         "skyscraper": 13,
#         "car, auto, automobile, machine, motorcar": 14,
#         "railing, rail": 15,
#         "boat": 16,
#         "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle": 17,
#         "truck, motortruck": 14,
#         "ship": 16,
#         "minibike, motorbike": 18,
#         "bicycle, bike, wheel, cycle": 19,
#     }  # class to our new label indices
#     CLASS2IDX = {k: i for i, k in enumerate(CLASSES)}
#     OUTPUT_DIR = "output/segment_img"
#     if not os.path.isdir(OUTPUT_DIR):
#         os.makedirs(OUTPUT_DIR)

#     consider_indices = [CLASS2IDX[x] for x in CONSIDER_CLASSES.keys()]
#     print(CLASS2IDX)
#     for file in tqdm(sorted(files)):
#         try:
#             img = Image.open(file)
#             resized_im, seg_map = MODEL.run(img)
#             filter_seg_map = np.zeros_like(seg_map, dtype=np.int32)
#             for label in CONSIDER_CLASSES.keys():
#                 filter_seg_map[seg_map == CLASS2IDX[label]] = CONSIDER_CLASSES[label]
#             filename = file.split("/")[-1].split(".")[0]
#             savemat(
#                 OUTPUT_DIR + "/" + filename + ".mat",
#                 {"segment": filter_seg_map},
#                 do_compression=True,
#             )
#         except:
#             pass

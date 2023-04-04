import numpy as np
from qgis.core import QgsMapLayerProxyModel, QgsRectangle, QgsWkbTypes, QgsPointXY, QgsGeometry, QgsFeature, \
    QgsProject, QgsMapLayerType, QgsPoint, QgsLayerTreeModel, QgsPalLayerSettings, QgsVectorLayerSimpleLabeling, \
    QgsTextFormat, QgsExpression
from qgis.gui import QgsMapLayerComboBox, QgsMapToolEmitPoint, QgsRubberBand, QgsMapTool
from qgis.PyQt.QtWidgets import QAction, QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QPushButton,\
    QSlider, QMessageBox
from qgis.PyQt.QtGui import QIcon, QFont, QColor
from qgis.PyQt.QtCore import Qt

import os
import math
import random
import sys
random.seed(2023)

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "module"))

import importlib
import subprocess

class RectangleMapTool(QgsMapToolEmitPoint):
    def __init__(self, canvas):
        self.canvas = canvas
        QgsMapToolEmitPoint.__init__(self, self.canvas)
        self.rubberBand = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
        self.rubberBand.setColor(Qt.red)
        self.rubberBand.setWidth(1)
        self.reset()

    def reset(self):
        self.startPoint = self.endPoint = None
        self.isEmittingPoint = False
        self.rubberBand.reset(QgsWkbTypes.PolygonGeometry)

    def canvasPressEvent(self, e):
        self.startPoint = self.toMapCoordinates(e.pos())
        self.endPoint = self.startPoint
        self.isEmittingPoint = True
        self.showRect(self.startPoint, self.endPoint)

    def canvasReleaseEvent(self, e):
        self.isEmittingPoint = False
        r = self.rectangle()
        if r is not None:
            min_x = r.xMinimum()
            max_x = r.xMaximum()
            min_y = r.yMinimum()
            max_y = r.yMaximum()
            print("Rectangle:", min_x, min_y, max_x, max_y)

    def canvasMoveEvent(self, e):
        if not self.isEmittingPoint:
            return

        self.endPoint = self.toMapCoordinates(e.pos())
        self.showRect(self.startPoint, self.endPoint)

    def hideRect(self):
        self.rubberBand.setVisible(False)

    def showRect(self, startPoint, endPoint):
        self.rubberBand.reset(QgsWkbTypes.PolygonGeometry)
        if startPoint.x() == endPoint.x() or startPoint.y() == endPoint.y():
            return

        point1 = QgsPointXY(startPoint.x(), startPoint.y())
        point2 = QgsPointXY(startPoint.x(), endPoint.y())
        point3 = QgsPointXY(endPoint.x(), endPoint.y())
        point4 = QgsPointXY(endPoint.x(), startPoint.y())

        self.rubberBand.addPoint(point1, False)
        self.rubberBand.addPoint(point2, False)
        self.rubberBand.addPoint(point3, False)
        self.rubberBand.addPoint(point4, True)    # true to update canvas
        self.rubberBand.show()

    def rectangle(self):
        if self.startPoint is None or self.endPoint is None:
            return None
        elif (self.startPoint.x() == self.endPoint.x() or \
            self.startPoint.y() == self.endPoint.y()):
            return None

        return QgsRectangle(self.startPoint, self.endPoint)

    def get_coordinates(self):
        r = self.rectangle()
        return (r.xMinimum(), r.yMinimum(), r.xMaximum(), r.yMaximum())

    def deactivate(self):
        QgsMapTool.deactivate(self)
        self.deactivated.emit()

class ObjectDetection:
    def __init__(self, iface):
        self.iface = iface
        self.canvas = self.iface.mapCanvas()
        self.layer = None
        self.num_points = 20
        self.rectangle_tool = None
        self.toolbar = self.iface.addToolBar('My Plugin Toolbar')
        try:
            importlib.import_module('cv2')
            importlib.import_module('PIL')
            importlib.import_module('numpy')
            importlib.import_module('tensorflow')
            importlib.import_module('simplecv')
        except ImportError:
            # Prompt user to install packages
            message = 'The plugin requires the some packages to be installed. Do you want to install it now?'
            result = QMessageBox.question(self.iface.mainWindow(), 'Install some packages', \
                message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if result == QMessageBox.Yes:
                # Install using pip
                subprocess.call(['pip', 'install', 'opencv-python==4.7.0'])
                subprocess.call(['pip', 'install', 'Pillow==9.2.0'])
                subprocess.call(['pip', 'install', 'numpy==1.20.2'])
                subprocess.call(['pip', 'install', 'tensorflow==2.11.1'])
            else:
                # User chose not to install
                pass
        
        try:
            importlib.import_module('simplecv')
        except ImportError:
            message = 'The plugin requires the simplecv package to be installed. Do you want to install it now?'
            result = QMessageBox.question(self.iface.mainWindow(), 'Install one package', \
                message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if result == QMessageBox.Yes:
                # # * Install requirements.txt
                requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
                subprocess.call(['python', '-m', 'pip', 'install', '-r', requirements_path])
                subprocess.call(['pip', 'install', '-i', 'https://test.pypi.org/simple/', 'simplecv==0.0.2'])
            else:
                pass
            
        # ! Old model
        from .segmentation import DeepLabModel
        from .ade import CLASSES
        
        self.DEEPLABMODEL = DeepLabModel(
            os.path.join(os.path.dirname(__file__), "deeplabv3_xception_ade20k_train/frozen_inference_graph.pb")
        )
        print("model front view loaded successfully!")

        self.TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(self.TEMP_DIR, exist_ok=True)

        self.ORI_CLASS2IDX = {k: i for i, k in enumerate(CLASSES)}

        self.CONSIDER_CLASSES = {
            "building, edifice": 1,
            "house": 1,
            "skyscraper": 1,
            "car, auto, automobile, machine, motorcar": 2,
            "truck, motortruck": 2,
            "airplane, aeroplane, plane": 3
        }  # class to our new label indices

        self.IDX2CONSIDER_CLASS = {1: "building", 2: "car+truck", 3: "plane"}
        
        # ! New Model
        import simplecv as sc
        from .data.isaid import COLOR_MAP
        from simplecv.api.preprocess import comm
        from simplecv.api.preprocess import segm
        from .module import register_model
        from .segmslidingwininference import SegmSlidingWinInference
        
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
        self.IDX2CONSIDER_CLASS_NEW = {i: c for i, c in enumerate(self.NEW_CLASSES)}

        self.CLASSES_NEEDED = ['small_Vehicle', 'large_Vehicle', 'storage_tank', 'plane']

        self.config_path='isaid.factseg'
        self.model_dir=os.path.join(os.path.dirname(__file__), "log")
        self.log_dir = self.model_dir
        self.ckpt_path=os.path.join(os.path.dirname(__file__), "log", "factseg50.pth")
        self.patch_size = 896

        self.model, self.global_step = sc.api.infer_tool.build_and_load_from_file(self.config_path, self.ckpt_path)
        self.segm_helper = SegmSlidingWinInference()
        
        print("model top view loaded successfully!")

        self.palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
        self.miou_op = sc.api.metric.NPmIoU(num_classes=len(self.NEW_CLASSES), logdir=self.log_dir)

        self.image_trans = comm.Compose([
            segm.ToTensor(True),
            comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)),
            comm.CustomOp(lambda x: x.unsqueeze(0))
        ])

        # Create a button and add it to the toolbar
        self.my_button_top = QPushButton('Detect Object Top View')
        self.my_button_top.clicked.connect(self.on_detect_button_click_satellite)
        self.toolbar.addWidget(self.my_button_top)

        # Create a button and add it to the toolbar
        self.my_button_front = QPushButton('Detect Object Front View')
        self.my_button_front.clicked.connect(self.on_detect_button_click)
        self.toolbar.addWidget(self.my_button_front)

        # Create a slider and add it to the toolbar
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(20, 1000)
        self.slider.setValue(20)
        self.slider.setTickInterval(20)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setFixedWidth(500)
        
        # Create a label widget to display the slider value
        self.label = QLabel(str(self.slider.value()))

        self.slider.valueChanged.connect(self.on_slider_value_changed)
        self.toolbar.addWidget(self.slider)
        self.toolbar.addWidget(self.label)

    def initGui(self):
        self.action = QAction(QIcon("icon.png"), "Draw Rectangle", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)

    def run(self):
        # Create a QDialog to show the layer selection dialog
        if self.rectangle_tool:
            self.rectangle_tool.hideRect()
        layer_dialog = QDialog(self.iface.mainWindow())
        layer_dialog.setWindowTitle("Select a layer")

        # Create a layout for the dialog
        layout = QVBoxLayout(layer_dialog)

        # Add a label to the dialog
        label = QLabel("Select a layer:")
        layout.addWidget(label)

        # Add a QgsMapLayerComboBox to the dialog
        layer_combo_box = QgsMapLayerComboBox()
        layer_combo_box.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        layout.addWidget(layer_combo_box)

        # Add buttons to the dialog
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(layer_dialog.accept)
        buttons.rejected.connect(layer_dialog.reject)
        layout.addWidget(buttons)

        # Show the dialog and wait for the user to make a selection
        if layer_dialog.exec() == QDialog.Accepted:
            self.layer = layer_combo_box.currentLayer()
            layers = QgsProject.instance().mapLayers()

            raster_layers = [layer for layer in layers.values() if layer.type() == QgsMapLayerType.RasterLayer]
            self.iface.mapCanvas().zoomToFeatureExtent(raster_layers[0].extent())
            self.iface.mapCanvas().refresh()

            self.rectangle_tool = RectangleMapTool(self.iface.mapCanvas())
            self.iface.mapCanvas().setMapTool(self.rectangle_tool)

    def on_detect_button_click_satellite(self):
        # * If Detect button is clicked with a drawn Rectangle
        # * -> Call Detect function
        # * Else
        # * -> Adjust the polygons' points according to the Slider
        
        if self.rectangle_tool and self.rectangle_tool.rectangle():
            rec = self.rectangle_tool.get_coordinates()
            self.detect_objects_satellite(rec)
            self.rectangle_tool.reset()
        else:            
            # Get all layers in the project
            layers = QgsProject.instance().mapLayers()

            # Filter the layers to only include shapefile layers
            shapefile_layers = [layer for layer in layers.values() if layer.type() == QgsMapLayerType.VectorLayer \
                and layer.dataProvider().name() == 'ogr' \
                and layer.dataProvider().dataSourceUri().lower().endswith('.shp')]

            if len(shapefile_layers) > 0:
                for feature in shapefile_layers[0].dataProvider().getFeatures():
                    multi_polygon = feature.geometry()
                    
                    # Convert the MultiPolygon to a list of QgsPolygon objects
                    multi_polygon_geometry = multi_polygon.asMultiPolygon()
                    
                    # Get each point [x, y] in a Polygon
                    point_list = []
                    flag_minus = False
                    
                    # ! Get all points except the last point because it is the first point
                    for point in multi_polygon_geometry[0][0][:-1]:
                        if point.y() < 0:
                            flag_minus = True
                        point_list.append([point.x(), abs(point.y())])
                    
                    # Adjust points to the slider's value
                    point_list = self.adjust_points(point_list)
                    
                    # Save to The Layer's Polygon
                    new_polygon = self.convert_to_polygon(point_list, flag_minus, edit=False)
                    
                    shapefile_layers[0].startEditing()
                    
                    # ! Update new feature to the shapefile_layers
                    if shapefile_layers[0].getFeature(feature.id()):
                        shapefile_layers[0].dataProvider().changeGeometryValues({feature.id(): new_polygon})
                        
                        # Commit the changes to the layer
                        shapefile_layers[0].commitChanges()
                    else:
                        error_message = f'Feature ID {feature.id()} not found in layer'
                        QMessageBox.critical(None, "Error", error_message)
    
    def on_detect_button_click(self):
        # * If Detect button is clicked with a drawn Rectangle
        # * -> Call Detect function
        # * Else
        # * -> Adjust the polygons' points according to the Slider
        
        if self.rectangle_tool and self.rectangle_tool.rectangle():
            rec = self.rectangle_tool.get_coordinates()
            self.detect_objects(rec)
            self.rectangle_tool.reset()
        else:            
            # Get all layers in the project
            layers = QgsProject.instance().mapLayers()

            # Filter the layers to only include shapefile layers
            shapefile_layers = [layer for layer in layers.values() if layer.type() == QgsMapLayerType.VectorLayer \
                and layer.dataProvider().name() == 'ogr' \
                and layer.dataProvider().dataSourceUri().lower().endswith('.shp')]

            if len(shapefile_layers) > 0:
                for feature in shapefile_layers[0].dataProvider().getFeatures():
                    multi_polygon = feature.geometry()
                    
                    # Convert the MultiPolygon to a list of QgsPolygon objects
                    multi_polygon_geometry = multi_polygon.asMultiPolygon()
                    
                    # Get each point [x, y] in a Polygon
                    point_list = []
                    flag_minus = False
                    
                    # ! Get all points except the last point because it is the first point
                    for point in multi_polygon_geometry[0][0][:-1]:
                        if point.y() < 0:
                            flag_minus = True
                        point_list.append([point.x(), abs(point.y())])
                    
                    # Adjust points to the slider's value
                    point_list = self.adjust_points(point_list)
                    
                    # Save to The Layer's Polygon
                    new_polygon = self.convert_to_polygon(point_list, flag_minus, edit=False)
                    
                    shapefile_layers[0].startEditing()
                    
                    # ! Update new feature to the shapefile_layers
                    if shapefile_layers[0].getFeature(feature.id()):
                        shapefile_layers[0].dataProvider().changeGeometryValues({feature.id(): new_polygon})
                        
                        # Commit the changes to the layer
                        shapefile_layers[0].commitChanges()
                    else:
                        error_message = f'Feature ID {feature.id()} not found in layer'
                        QMessageBox.critical(None, "Error", error_message)

    def on_slider_value_changed(self, value):
        # Code to handle slider value changes
        self.num_points = value
        self.label.setText(str(value))
        
    def angle_between_points(self, point_1, point_2):
        vector_1 = [point_1[0], point_1[1]]
        vector_2 = [point_2[0], point_2[1]]

        unit_vector_1 = vector_1 / (np.linalg.norm(vector_1) + 0.001)
        unit_vector_2 = vector_2 / (np.linalg.norm(vector_2) + 0.001)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

        return angle * 180 / math.pi

    def adjust_points(self, obj_points):
        if self.num_points == 0:
            return obj_points
        
        # * There are num_points around a closed polygon 360 degrees => angle 
        # * between points should be 360 / num_points.
        # * If the size of obj_points is already less than self.num_points => proceed
        # * to add more points, no need to use angle sorting as above => save more time.
        
        if len(obj_points) == self.num_points:
            return obj_points
        
        elif len(obj_points) < self.num_points:
            new_obj_points = obj_points
            
        else: # CASE: len(obj_points) > self.num_points
            new_obj_points = []
            
            # * Calculate coordinates of the center point
            temp_x, temp_y = 0, 0
            for point in obj_points:
                temp_x += point[0]
                temp_y += point[1]
            
            temp_x, temp_y = temp_x/len(obj_points), temp_y/len(obj_points)
            center = [temp_x, temp_y]

            # * Angle sorting
            chosen_angle = 360 / self.num_points
            temp = obj_points[0]
            new_obj_points += [temp]
            
            for point in obj_points:
                diff_1 = [temp[0] - center[0], temp[1] - center[1]]
                diff_2 = [point[0] - center[0], point[1] - center[1]]
                if self.angle_between_points(diff_1, diff_2) >= chosen_angle:
                    new_obj_points += [point]
                    temp = point
                else:
                    pass
            
        # ! If the size of new_obj_points is less than the num_points required
        # ! => add more points to the midpoint of random pairs of points
        size_n = len(new_obj_points)
        
        if size_n < self.num_points and size_n >= 3:
            # * pair = [current index, current_index + 1] => cannot count the last index
            # * => range(size_n - 1)
            random_indices = random.choices(range(size_n - 1), k = self.num_points - size_n)
            for index in random_indices:
                # Generate midpoint of a pair
                midpoint_x = (new_obj_points[index][0] + new_obj_points[index + 1][0]) / 2
                midpoint_y = (new_obj_points[index][1] + new_obj_points[index + 1][1]) / 2
                new_obj_points.insert(index + 1, [midpoint_x, midpoint_y])
        
        return new_obj_points

    def get_image(self, layer):
        extent = layer.extent()
        provider = layer.dataProvider()

        # Get the block size of the raster data provider
        bands = provider.bandCount()
        image = np.empty((layer.height(), layer.width(), bands), dtype=np.uint8)
        for band in range(bands):
            block = provider.block(band+1, extent, layer.width(), layer.height())
            data = bytes(block.data())
            image[:, :, band] = np.frombuffer(data, dtype=np.uint8).reshape(layer.height(), layer.width())
        return image

    def convert_to_polygon(self, obj_points, flag_minus, label_name=None, edit=True):
        # Convert the polygon coordinates to QGIS format
        if flag_minus:
            polygon_coords = [QgsPointXY(p[0], -p[1]) for p in obj_points]
        else:
            polygon_coords = [QgsPointXY(p[0], p[1]) for p in obj_points]
            
        polygon_coords.append(polygon_coords[0])
        polygon = QgsGeometry.fromPolygonXY([polygon_coords])
        
        if edit:
            # Create a feature for the polygon and add it to the list
            feature = QgsFeature()
            feature.setGeometry(polygon)
            self.layer.startEditing()
            self.layer.dataProvider().addFeatures([feature])
            
            # ! ==================================================
            # ! Add feature label name here                      
            # ! ==================================================
            
            
            # ! ==================================================
            # ! End
            # ! ==================================================
            
            # * Update the current polygon on the iface
            self.iface.mapCanvas().refresh()
        else:
            return polygon
        
    def detect_objects_satellite(self, rect_coords):
        import cv2
        from PIL import Image
        from .segmentation_new import get_top_k_objects_polygon
        
        self.rectangle_tool.hideRect()
        layers = QgsProject.instance().mapLayers()
        raster_layers = [layer for layer in layers.values() if layer.type() == QgsMapLayerType.RasterLayer]
        for layer in raster_layers:
            image = self.get_image(layer)
            
            # Set the rectangle
            x1 = rect_coords[0]
            x2 = rect_coords[2]
            y1 = rect_coords[1]
            y2 = rect_coords[3]
            flag_minus = False
            if y1 < 0:
                flag_minus = True
                y1 = abs(rect_coords[3])
                y2 = abs(rect_coords[1])

            max_n_objects = 999
            
            temp_filepath = f"{self.TEMP_DIR}/temp.jpg"
            cv2.imwrite(temp_filepath, image)
            
            # ! New model
            img = Image.open(temp_filepath)
            img = img.crop((x1, y1, x2, y2))

            # inference
            image = np.array(img)
            h, w = image.shape[:2]
            seg_helper = self.segm_helper.patch((h, w), patch_size=(self.patch_size, self.patch_size), stride=512,
                                            transforms=self.image_trans)
            out = seg_helper.forward(self.model, image, size_divisor=32)
            out_ori = out.argmax(dim=1).numpy()[0]
            out = np.zeros_like(out_ori)
            
            for class_needed in self.CLASSES_NEEDED:
                out[out_ori == self.NEW_CLASSES.index(class_needed)] = self.NEW_CLASSES.index(class_needed)

            boxes = get_top_k_objects_polygon(
                out,
                x1,
                y1,
                img.width,
                img.height,
                self.IDX2CONSIDER_CLASS_NEW,
                max_n_objects=max_n_objects,
            )
            
            if len(boxes) > 0:
                for obj in boxes:
                    if len(obj["points"]) > 0:
                        # Adjust the number of points to the slider's value
                        obj_points = self.adjust_points(obj["points"])
                        self.convert_to_polygon(obj_points, flag_minus, label_name=obj["label"], edit=True)
    
    def detect_objects(self, rect_coords):
        import cv2
        from PIL import Image
        from .segmentation import get_largest_object_polygon
        
        self.rectangle_tool.hideRect()
        layers = QgsProject.instance().mapLayers()
        raster_layers = [layer for layer in layers.values() if layer.type() == QgsMapLayerType.RasterLayer]
        for layer in raster_layers:
            image = self.get_image(layer)
            
            # Set the rectangle
            x1 = rect_coords[0]
            x2 = rect_coords[2]
            y1 = rect_coords[1]
            y2 = rect_coords[3]
            flag_minus = False
            if y1 < 0:
                flag_minus = True
                y1 = abs(rect_coords[3])
                y2 = abs(rect_coords[1])

            max_n_objects = 2
            
            temp_filepath = f"{self.TEMP_DIR}/temp.jpg"
            cv2.imwrite(temp_filepath, image)

            # ! Old model
            img = Image.open(temp_filepath)
            img = img.crop((x1, y1, x2, y2))
            
            resized_im, seg_map = self.DEEPLABMODEL.run(img)
            filter_seg_map = np.zeros_like(seg_map, dtype=np.int32)
            for label in self.CONSIDER_CLASSES.keys():
                filter_seg_map[seg_map == self.ORI_CLASS2IDX[label]] = self.CONSIDER_CLASSES[label]
            boxes = get_largest_object_polygon(filter_seg_map, x1, y1, img.width, img.height, self.IDX2CONSIDER_CLASS, max_n_objects)
            
            if len(boxes) > 0:
                for obj in boxes:
                    if len(obj["points"]) > 0:
                        # Adjust the number of points to the slider's value
                        obj_points = self.adjust_points(obj["points"])
                        self.convert_to_polygon(obj_points, flag_minus, label_name=obj["label"], edit=True)

    def unload(self):
        # Remove plugin button from toolbar
        self.iface.removeToolBarIcon(self.action)
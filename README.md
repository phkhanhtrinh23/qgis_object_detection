# QGIS Object Detection

## How to place the repository?
Your QGIS plugin directory should be located at in order for QGIS to plug it in:
`C:/Users/<your_username>/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/qgis_object_detection`

## What's Next?
- Customize it by editing the implementation file: ``object_detection.py``
- Create your own custom icon, replacing the default icon.png
- Modify your user interface by opening `object_detection_dockwidget_base.ui` in Qt Designer
- You can use the Makefile to compile your Ui and resource files when you make changes. This requires GNU make (gmake)

## Weights
- You can download 2 folders containing pretrained weights from this [link](https://drive.google.com/drive/folders/1Lr_cA6nJc3kMoOzineWcSJ-rXLT5jvv_?usp=sharing):
  - `deeplabv3_xception_ade20k_train`
  - `log`

and place them in this repository: `./qgis_object_detection/`

## Additional information
For more information, see the PyQGIS Developer Cookbook at:

http://www.qgis.org/pyqgis-cookbook/index.html

(C) 2011-2018 GeoApt LLC - geoapt.com

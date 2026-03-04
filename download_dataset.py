from roboflow import Roboflow

rf = Roboflow(api_key="WlhlOTiRl4eMdec4IL3y")
project = rf.workspace("drones-lfobz").project("drone-detection-vg2iu")
dataset = project.version(1).download("yolov8")
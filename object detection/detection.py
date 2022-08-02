from imageai.Detection import ObjectDetection
import tensorflow
import keras
import os
import cv2

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path,
                                                                      "execution_path/image.jpg"),
                                             output_image_path=os.path.join(execution_path, "image_new/imagenew.jpg"),
                                             display_percentage_probability=True, display_object_name=True)

for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"])

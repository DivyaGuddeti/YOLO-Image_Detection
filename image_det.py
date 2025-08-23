# Om Shri Maathre Namaha
# Om Shri Maathre Namaha
# Om Shri Maathre Namaha

from ultralytics import YOLO
import numpy

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")

# Predict an Image
detection_output = model.predict(source=r"D:\SriKrishna\Sai - FSDS - Global AI Engineer\Sai - Programs\Data-Source\XUV.jpg", conf=0.25, save=True)

# Display tensor array
print(detection_output)

# Display numpy array
print(detection_output[0].numpy())
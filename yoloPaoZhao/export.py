from ultralytics import YOLO

# Load a model
model = YOLO(r"F:\AI\AI\yoloPaoZhao\runs\train\yolo11n\weights\best.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom trained model

# Export the model

#model.export(format="openvino" , opset=11 ,simplify=True)

model.export(format="onnx",
             simplify=True)

#yolo export model=yolov10m.pt format=onnx opset=11 simplify
#ovc yolov10m.onnx
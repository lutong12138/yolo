import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=32,
                close_mosaic=0,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
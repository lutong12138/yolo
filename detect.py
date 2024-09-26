import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('best.pt') # select your model.pt path
    model.predict(source='frame_1.jpg',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # visualize=True # visualize model features maps
                )
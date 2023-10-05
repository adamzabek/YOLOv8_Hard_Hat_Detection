from ultralytics import YOLO


model = YOLO('./model/best.pt')


results = model.track(source = 0, show = True, persist = True, tracker="bytetrack.yaml")
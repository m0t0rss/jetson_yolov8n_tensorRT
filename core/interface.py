import cv2
def draw_detections(frame, detections, target_classes):
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = map(float, det)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_id = int(cls_id)
        
        # Безпечне отримання імені класу
        if cls_id < len(target_classes):
            label = f"{target_classes[cls_id]} {conf:.2f}"
        else:
            label = f"unknown({cls_id}) {conf:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame
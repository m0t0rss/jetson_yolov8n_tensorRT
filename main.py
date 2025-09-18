import multiprocessing
import logging
import gc
import cv2


from core.camera import video_cam  # повертає CameraStream
from core.detector import TRTDetector
from core.interface import draw_detections
from core.controll import dummy_controller  # контролер керування (твій) motor_controller = none 
from utils.config import TARGET_CLASSES  # наприклад, ["person", "cell phone"]

logging.basicConfig(level=logging.INFO)
def main():
    # Ініціалізація
    model = TRTDetector("model/yolov8n.engine")
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        cam = video_cam()
        
        # Процес керування
        control_queue = multiprocessing.Queue()
        controller = multiprocessing.Process(
            target=dummy_controller,
            args=(control_queue,),
            daemon=True
        )
        controller.start()

        while True:
            ret, frame = cam.read()
            if not ret:
                logger.warning("Failed to read frame")
                break

            # Детекція об'єктів 
            results = model(frame, conf=0.3) 
            

            # Перевірка на наявність детекцій
            if results and results[0].boxes is not None:
                classes = [int(c) for c in results[0].boxes.cls]
                # print("BBoxes:", results[0].boxes.xyxy)
                # print("Confidences:", results[0].boxes.conf)
                print("Class names:", [results[0].names[c] for c in classes])
                
                # Візуалізація
                vis_frame = results[0].plot() # plot() алює bounding boxes
                cv2.imshow("Detection", vis_frame)
            else:
                print("No detections")
                cv2.imshow("Detection", frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    finally:
        control_queue.put("STOP")
        controller.join(timeout=2)
        if controller.is_alive():
            controller.terminate()
        cam.stop()
        cv2.destroyAllWindows()
        model.release()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # важливо для Jetson!
    main()
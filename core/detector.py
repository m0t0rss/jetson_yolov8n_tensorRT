import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import cv2
from typing import List, Optional, Union, Tuple

class TRTResults:
    """Detection results compatible with Ultralytics YOLO format."""
    def __init__(self, detections: np.ndarray, orig_img: np.ndarray, names: List[str]):
        """
        Args:
            detections: Array of detections [N, 6] in format [x1, y1, x2, y2, conf, cls_id]
            orig_img: Original image (BGR format)
            names: List of class names
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.names = names
        self.boxes = self.Boxes(detections) if detections.size > 0 else None

    class Boxes:
        """Container for bounding boxes."""
        def __init__(self, detections: np.ndarray):
            self.data = detections                   # [N, 6]
            self.xyxy = detections[:, :4]            # [N, 4]
            self.conf = detections[:, 4]             # [N]
            self.cls = detections[:, 5].astype(int)  # [N]
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return {
                'xyxy': self.xyxy[idx],
                'conf': self.conf[idx],
                'cls': self.cls[idx]
            }

    def plot(self, show_conf: bool = True, line_thickness: int = 2) -> np.ndarray:
        """Visualize results on image."""
        img = self.orig_img.copy()
        if self.boxes is None:
            return img
            
        for box in self.boxes:
            x1, y1, x2, y2 = map(int, box['xyxy'])
            conf, cls_id = box['conf'], box['cls']
            
            # Draw rectangle
            cv2.rectangle(
                img, (x1, y1), (x2, y2),
                (0, 255, 0), line_thickness
            )
            
            # Label with class and confidence
            label = f"{self.names[cls_id]}: {conf:.2f}" if show_conf else self.names[cls_id]
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                img, (x1, y1 - h - 10), (x1 + w, y1),
                (0, 255, 0), -1
            )
            cv2.putText(
                img, label, (x1, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        return img

class TRTDetector:
    """Optimized TensorRT object detector with YOLOv8 interface."""
    def __init__(self, engine_path: str, classes: Optional[List[str]] = None):
        """
        Args:
            engine_path: Path to .engine file
            classes: List of class names (None for default COCO classes)
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        
        # Input/output configuration
        self.input_shape = (1, 3, 640, 640)  # batch, channels, height, width
        self.output_shape = (1, 84, 8400)    # For YOLOv8
        
        # Allocate memory
        self._setup_bindings()
        
        # Class names
        self.names = classes or self._get_default_classes()
        print(f"🟢 TRTDetector ready. Classes: {len(self.names)}")

    def _load_engine(self, engine_path: str) -> trt.ICudaEngine:
        """Load TensorRT engine."""
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _setup_bindings(self):
        """Setup CUDA bindings."""
        self.inputs = cuda.mem_alloc(int(np.prod(self.input_shape) * np.float32().nbytes))
        self.outputs = cuda.mem_alloc(int(np.prod(self.output_shape) * np.float32().nbytes))
        self.bindings = [int(self.inputs), int(self.outputs)]
        self.stream = cuda.Stream()

    def _get_default_classes(self) -> List[str]:
        """Return default COCO class names."""
        return[
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]


    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess image for TensorRT inference."""
        if frame is None or frame.size == 0:
            raise ValueError("Empty input image")
        
        # Convert BGR to RGB, resize, normalize
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(np.expand_dims(img, axis=0))

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Custom NMS implementation."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep, dtype=np.int32)
    
           # переводить число в діапазон 0.1 -1 для йоло коф впевності 
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  

    def postprocess(self, preds: np.ndarray, orig_shape: Tuple[int,int], conf_thres: float = 0.25, iou_thres: float = 0.45) -> np.ndarray:
        try:
            # preds повертає в нормальному вігляді (0 -1)
            preds = preds[0].T  # (8400, 84)

            # Застосовуємо sigmoid до confidence (objectness) і класів
            preds[:, 4] = self.sigmoid(preds[:, 4])       # objectness score
            preds[:, 5:] = self.sigmoid(preds[:, 5:])     # класові ймовірності

            obj_conf = preds[:, 4]
            class_confs = preds[:, 5:]
            cls_ids = np.argmax(class_confs, axis=1)
            cls_conf = class_confs[np.arange(len(cls_ids)), cls_ids]

            scores = obj_conf * cls_conf

            mask = scores > conf_thres
            preds = preds[mask]
            scores = scores[mask]
            cls_ids = cls_ids[mask]


            if len(preds) == 0:
                print("No detections")
                return np.empty((0, 6), dtype=np.float32)

            boxes = np.zeros((len(preds), 4), dtype=np.float32)
            boxes[:, 0] = preds[:, 0] - preds[:, 2] / 2  # x1
            boxes[:, 1] = preds[:, 1] - preds[:, 3] / 2  # y1
            boxes[:, 2] = preds[:, 0] + preds[:, 2] / 2  # x2
            boxes[:, 3] = preds[:, 1] + preds[:, 3] / 2  # y2

            keep = self._nms(boxes, scores, iou_thres)

            if len(keep) == 0:
                return np.empty((0, 6), dtype=np.float32)

            return np.concatenate([
                boxes[keep],
                scores[keep, None],
                cls_ids[keep, None].astype(np.float32)
            ], axis=1)

        except Exception as e:
            print(f"Postprocessing error: {e}")
            return np.empty((0, 6), dtype=np.float32)

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Run inference on a single frame."""
        try:
            input_tensor = self.preprocess(frame)
            
            # Async inference
            cuda.memcpy_htod_async(self.inputs, input_tensor.ravel(), self.stream)
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )
            
            # Get results
            output_tensor = np.empty(self.output_shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(output_tensor, self.outputs, self.stream)
            self.stream.synchronize()
            
            # print("Input tensor stats:", input_tensor.min(), input_tensor.max(), input_tensor.mean())
            # print("Output tensor shape:", output_tensor.shape)
            # print("Output tensor sample:", output_tensor.flatten()[:10])

            return output_tensor
            
        except Exception as e:
            print(f"Inference error: {e}")
            return np.zeros(self.output_shape, dtype=np.float32)

    def predict(self, 
               source: np.ndarray, 
               conf: float = 0.25,
               iou: float = 0.45,
               show: bool = False) -> List['TRTResults']:
        """Main detection interface compatible with Ultralytics YOLO."""
        try:
            preds = self.infer(source)
            if preds is None or preds.size == 0:
                return [TRTResults(np.empty((0, 6), dtype=np.float32), source, self.names)]
                
            detections = self.postprocess(preds, orig_shape=source.shape[:2], conf_thres=conf, iou_thres=iou)

            results = TRTResults(detections, source, self.names)
            
            if show:
                cv2.imshow("Detection", results.plot())
                cv2.waitKey(1)
            
            return [results]
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return [TRTResults(np.empty((0, 6), dtype=np.float32), source, self.names)]

    def __call__(self, *args, **kwargs):
        """Enable calling detector like a function: model(frame)."""
        return self.predict(*args, **kwargs)

    def release(self):
        """Properly release resources."""
        if hasattr(self, 'engine'):
            del self.engine
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'inputs'):
            self.inputs.free()
        if hasattr(self, 'outputs'):
            self.outputs.free()
        print("🔴 TRTDetector resources released")
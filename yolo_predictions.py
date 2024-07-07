import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader


class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        # Load YAML
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']

        # Load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Counters for vehicle types and total vehicles
        self.vehicle_count = 0
        self.car_count = 0
        self.bike_count = 0
        self.scooter_count = 0
        self.auto_count = 0
        self.bus_count = 0
        self.truck_count = 0
        
        # Line position to count vehicles
        self.line_position = None
        # Dictionary to store tracked vehicles
        self.tracker = {}
        # Counter for vehicle IDs
        self.vehicle_id = 0
        # Distance threshold for matching objects
        self.distance_threshold = 50

    def predictions(self, image):
        row, col, d = image.shape
        if self.line_position is None:
            self.line_position = int(row * 0.9)  # Define line position at 90% of the height

        # Convert image to square array
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image
        
        # Get YOLO prediction from the image
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()  # Detection from YOLO

        # Non-Maximum Suppression
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        # Image dimensions
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]  # Confidence of detection
            if confidence > 0.4:
                class_score = row[5:].max()  # Maximum probability from 20 objects
                class_id = row[5:].argmax()  # Index of max probability

                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    # Bounding box coordinates
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])

                    # Append values to lists
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # NMS
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45).flatten()

        current_tracker = {}
        for ind in index:
            # Extract bounding box
            x, y, w, h = boxes_np[ind]
            class_id = classes[ind]
            class_name = self.labels[class_id]
            colors = self.generate_colors(class_id)

            # Draw bounding box and text
            text = f'{class_name}'
            cv2.rectangle(image, (x, y), (x + w, y + h), colors, 2)
            cv2.rectangle(image, (x, y - 30), (x + w, y), colors, -1)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

            # Track vehicles
            if class_name in ["car", "bike", "scooter", "auto", "bus", "truck"]:
                matched_id = None
                for vid, (vx, vy, vw, vh) in self.tracker.items():
                    if self.is_same_vehicle((x, y, w, h), (vx, vy, vw, vh)):
                        matched_id = vid
                        current_tracker[vid] = (x, y, w, h)
                        break

                if matched_id is None:
                    current_tracker[self.vehicle_id] = (x, y, w, h)
                    self.vehicle_id += 1

        # Check if vehicles have crossed the line
        for vehicle_id, (x, y, w, h) in current_tracker.items():
            if y + h > self.line_position and self.tracker.get(vehicle_id) is None:
                class_id = classes[ind]
                class_name = self.labels[class_id]
                if vehicle_id in self.tracker:  # Ensure it's not counted already
                    continue
                if class_name == "car":
                    self.car_count += 1
                elif class_name == "bike":
                    self.bike_count += 1
                elif class_name == "scooter":
                    self.scooter_count += 1
                elif class_name == "auto":
                    self.auto_count += 1
                elif class_name == "bus":
                    self.bus_count += 1
                elif class_name == "truck":
                    self.truck_count += 1
                self.vehicle_count += 1  # Increment total vehicle count
                self.tracker[vehicle_id] = (x, y, w, h)  # Update tracker

        # Remove old vehicle IDs from the tracker
        self.tracker = {k: v for k, v in self.tracker.items() if k in current_tracker}

        # Draw the counting line
        cv2.line(image, (0, self.line_position), (col, self.line_position), (0, 255, 255), 2)

        # Display vehicle counts on the image
        cv2.putText(image, f'Car: {self.car_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, f'Bike: {self.bike_count}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, f'Scooter: {self.scooter_count}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, f'Auto: {self.auto_count}', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, f'Bus: {self.bus_count}', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, f'Truck: {self.truck_count}', (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, f'Total Vehicles: {self.vehicle_count}', (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return image

    def generate_colors(self, ID):
        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()
        return tuple(colors[ID])
    
    def is_same_vehicle(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        center1 = (x1 + w1 / 2, y1 + h1 / 2)
        center2 = (x2 + w2 / 2, y2 + h2 / 2)
        distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        return distance < self.distance_threshold

# Example usage:
# model = YOLO_Pred('model.onnx', 'data.yaml')
# cap = cv2.VideoCapture('video.mp4')
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     result_frame = model.predictions(frame)
#     cv2.imshow('Vehicle Detection', result_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

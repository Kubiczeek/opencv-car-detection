from ultralytics import YOLO
import cv2
import numpy as np


def detect_vehicles(image_path, conf_threshold=0.3):
    # Load YOLOv8 model
    model = YOLO('yolo11n.pt')

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    # List of vehicle classes in COCO dataset
    vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']

    # Run detection
    results = model(image)

    # Counter for vehicles
    vehicle_count = 0

    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class name
            cls = result.names[int(box.cls[0])]

            # Check if detected object is a vehicle
            if cls in vehicle_classes:
                # Get confidence score
                conf = float(box.conf[0])

                # Continue only if confidence is above threshold
                if conf >= conf_threshold:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add label with class name and confidence
                    label = f"{cls} {conf * 100:.1f}%"
                    cv2.putText(image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    vehicle_count += 1

    # Add total vehicle count
    cv2.putText(image, f"Vehicles: {vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image


def main():
    # Replace with your image path
    image_path = "Cars_image.jpg"

    try:
        # Process image
        result_image = detect_vehicles(image_path)

        # Save result
        output_path = "detected_vehicles.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"Detection completed. Result saved to {output_path}")

        # Display result (optional)
        cv2.imshow("Vehicle Detection", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
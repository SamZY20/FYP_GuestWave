import cv2 as cv
import pytesseract
from ultralytics import YOLO
from sort.sort import *
from util import get_car

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'


def capture_license_plate():
    mot_tracker = Sort()
    vehicles = [2, 3, 5, 6, 7]

    # Create a VideoCapture object for device 0 (default camera)
    cap = cv.VideoCapture(0)

    # Load Models
    preTrainedModel = YOLO('yolov8n.pt')
    trainedModel = YOLO('runs\\detect\\train\\weights\\best.pt')

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

        # Detect Vehicle
        detections = preTrainedModel(frame)[0]
        detections_ = []

        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection

            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect License Plate
        license_plates = trainedModel(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            xcar1, ycar1, xcar2, ycar2, car_id = get_car(
                license_plate, track_ids)  # Assign license plate to car

            license_plate_crop = frame[int(y1): int(y2), int(
                x1): int(x2), :]  # Crop License Plate

            gray = cv.cvtColor(license_plate_crop, cv.COLOR_BGR2GRAY)
            _, thresh = cv.threshold(gray, 64, 255, cv.THRESH_BINARY_INV)

            license_plate_text = pytesseract.image_to_string(
                thresh, config='--psm 7')

            if license_plate_text is not None:
                # Draw a bounding box around the car
                cv.rectangle(frame, (xcar1, ycar1),
                             (xcar2, ycar2), (0, 255, 0), 2)

                # Draw a bounding box around the license plate
                cv.rectangle(frame, (int(x1), int(y1)),
                             (int(x2), int(y2)), (0, 0, 255), 2)

                # Print the license plate number
                print("Car ID:", car_id)
                print("Detected license plate:", license_plate_text)

        # Display the frame with bounding boxes
        cv.imshow("License Plate Detection", frame)

        # Break the loop when the 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    capture_license_plate()

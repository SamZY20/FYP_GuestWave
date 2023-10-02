import cv2 as cv
import pytesseract

from ultralytics import YOLO
from sort.sort import *
from util import get_car

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'


def capture_license_plate(image_path):
    mot_tracker = Sort()
    vehicles = [2, 3, 5, 6, 7]

    # Load Image
    image = cv.imread(image_path)

    # Load Models
    preTrainedModel = YOLO('yolov8n.pt')
    trainedModel = YOLO('runs\\detect\\train\\weights\\best.pt')

    # Detect Vehicle
    detections = preTrainedModel(image)[0]
    detections_ = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection

        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect License Plate
    license_plates = trainedModel(image)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        xcar1, ycar1, xcar2, ycar2, car_id = get_car(
            license_plate, track_ids)  # Assign license plate to car

        license_plate_crop = image[int(y1): int(y2), int(
            x1): int(x2), :]  # Crop License Plate

        gray = cv.cvtColor(license_plate_crop, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 64, 255, cv.THRESH_BINARY_INV)

        license_plate_text = pytesseract.image_to_string(
            thresh, config='--psm 7')

        if license_plate_text is not None:
            # Draw a bounding box around the license plate
            cv.rectangle(image, (int(x1), int(y1)),
                         (int(x2), int(y2)), (0, 0, 255), 2)

            # Print the license plate number
            print("Car ID:", car_id)
            print("Detected license plate:", license_plate_text)

    cv.imshow("License Plate Detection", image)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":

    image_path = "assets\pictures\proton_iriz.webp"
    capture_license_plate(image_path)

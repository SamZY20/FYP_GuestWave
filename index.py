import base64
import cv2 as cv
import pytesseract

from ultralytics import YOLO
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from sort.sort import *
from util import get_car

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'

# Initialize Flask
app = Flask(__name__)
# Initialize Flask-SocketIO
app.config['SECRET_KEY'] = 'sam-secret'
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

vehicles = [2, 3, 5, 6, 7]

# Load Models
preTrainedModel = YOLO('yolov8n.pt')
trainedModel = YOLO('runs\\detect\\train\\weights\\best.pt')


def capture_license_plate(image):
    mot_tracker = Sort()

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
        gray_blurred = cv.GaussianBlur(gray, (5, 5), 0)
        thresh = cv.adaptiveThreshold(
            gray_blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)
        # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        # thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        # _, thresh = cv.threshold(gray, 64, 255, cv.THRESH_BINARY_INV)

        license_plate_text = pytesseract.image_to_string(
            thresh, config='--psm 7')

        if license_plate_text is not None:
            # Draw a bounding box around the license plate
            cv.rectangle(image, (int(x1), int(y1)),
                         (int(x2), int(y2)), (0, 0, 255), 2)

            return license_plate_text, image

    return None


@socketio.on('laravel_event')
def handle_laravel_event(data):
    print('Received event from Laravel:', data)

    # Example: Send response back to Laravel
    emit('flask_response', {'response': 'Hello from Flask-SocketIO!'})


@socketio.on('video_frame')
def handle_video_frame(data):
    # Decode base64 string to bytes
    image_data = base64.b64decode(data)
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    # Decode numpy array to OpenCV BGR image
    image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    try:
        license_plate_text, captured_image = capture_license_plate(image)
        if license_plate_text:
            retval, buffer = cv.imencode('.jpg', captured_image)
            image_data_with_boxes = base64.b64encode(buffer).decode('utf-8')

            emit('license_plate', {
                 'text': license_plate_text, 'image': image_data_with_boxes})

    except Exception as e:
        print(f"Error processing video frame: {e}")


if __name__ == "__main__":
    socketio.run(app, debug=False)

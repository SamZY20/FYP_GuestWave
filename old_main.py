import cv2 as cv
import pytesseract
from ultralytics import YOLO
from flask import Flask, Response, render_template
from flask_socketio import SocketIO

from sort.sort import *
from util import get_car

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'

# Initialize Flask
app = Flask(__name__)
# Initialize Flask-SocketIO
app.config['SECRET_KEY'] = 'sam-secret'
socketio = SocketIO(app)


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
        detections = preTrainedModel(frame, stream=True)
        detections_ = []

        for detection in detections:
            for result in detection.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result

                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect License Plate
        license_plates = trainedModel(frame, stream=True)
        for license_plate in license_plates:
            for license_plate_data in license_plate.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate_data

                xcar1, ycar1, xcar2, ycar2, car_id = get_car(
                    license_plate_data, track_ids)  # Assign license plate to car

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

        _, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        socketio.emit('video', frame)

        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # # Display the frame with bounding boxes
        # cv.imshow("License Plate Detection", frame)

        # # Break the loop when the 'q' key is pressed
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release the camera and close all OpenCV windows
    # cap.release()
    # cv.destroyAllWindows()


@app.route('/video_feed')
def video_feed():
    return Response(capture_license_plate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on("connect")
def handle_connect():
    print("Client Connected!")


@socketio.on("capture_device")
def capture_device():
    capture_license_plate()


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    socketio.run(app, debug=False)

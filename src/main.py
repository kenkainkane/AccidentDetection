import cv2
import numpy as np
from ultralytics import YOLO

import os


def get_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    video_path = os.path.join(base_dir, "..", "videos", "PedestrianAccident.mp4")
    model_path = os.path.join(base_dir, "..", "models", "yolo11l.pt")

    return video_path, model_path


def load_model(model_path):
    return YOLO(model_path)


def get_selected_class_ids(model, selected_classes):
    return [i for i, name in model.names.items() if name in selected_classes]


def setup_video_capture(video_path, skip_frames=0):
    cap = cv2.VideoCapture(video_path)

    # Skip initial frames if specified
    for _ in range(skip_frames):
        cap.read()

    return cap


def process_frame(frame, crop_video=True):
    if crop_video and frame.shape[1] > 500:
        frame = frame[:, 250:-250]

    frame_height, frame_width = frame.shape[:2]
    return frame, frame_height, frame_width


def detect_objects(model, frame, selected_class_ids):
    detected_objects = []
    results = model.track(frame, persist=True, iou=0.3)

    if results and len(results[0].boxes) > 0:
        for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            cls = int(cls)
            if cls in selected_class_ids:
                x1, y1, x2, y2 = map(int, box)
                label = model.names[cls]

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                axes = ((x2 - x1) // 2, (y2 - y1) // 2)

                # For car objects, check for red color
                if label == "car":
                    car_roi = frame[y1:y2, x1:x2]
                    if car_roi.size > 0:
                        mean_r = np.mean(car_roi[:, :, 2])
                        if mean_r < 95:
                            continue

                detected_objects.append(
                    {
                        "label": label,
                        "box": (x1, y1, x2, y2),
                        "center": center,
                        "axes": axes,
                    }
                )

    return detected_objects


def find_object_by_label(detected_objects, label):
    for obj in detected_objects:
        if obj["label"] == label:
            return obj
    return None


def get_object_info(detected_object, last_pose=None):
    if detected_object:
        box = detected_object["box"]
        center = detected_object["center"]
        axes = detected_object["axes"]
        return (box, center, axes), box

    elif last_pose:
        x1, y1, x2, y2 = last_pose
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        return ((x1, y1, x2, y2), center, axes), last_pose

    return None, None


def calculate_distance(point1, point2):
    return int(np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2))


def draw_object(frame, box, center, axes, label, color):
    x1, y1, x2, y2 = box
    cv2.ellipse(frame, center, axes, 0, 0, 360, color, 2)
    cv2.circle(frame, center, 3, color, -1)
    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )


def draw_distance_line(frame, point1, point2, distance, threshold):
    alert_color = (0, 0, 255) if distance < threshold else (0, 255, 0)

    cv2.line(frame, point1, point2, alert_color, 2)
    line_x, line_y = (point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2
    cv2.putText(
        frame,
        str(distance),
        (line_x, line_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        alert_color,
        2,
    )

    return alert_color


def draw_alert_border(
    frame, frame_width, frame_height, alert_color, border_thickness=10
):
    cv2.rectangle(
        frame,
        (border_thickness, border_thickness),
        (frame_width - border_thickness, frame_height - border_thickness),
        alert_color,
        border_thickness // 2,
    )


def display_frame(frame):
    cv2.imshow("frame", frame)
    return not (cv2.waitKey(25) & 0xFF == ord("q"))


def detect_pedestrian_accidents(
    video_path,
    model_path="yolo11l.pt",
    selected_classes=None,
    skip_frames=100,
    distance_threshold=150,
    crop_video=True,
    display_output=True,
):
    # Set default selected classes if none provided
    if selected_classes is None:
        selected_classes = ["person", "car"]

    # Initialize
    model = load_model(model_path)
    selected_class_ids = get_selected_class_ids(model, selected_classes)
    cap = setup_video_capture(video_path, skip_frames)

    last_person_pose = None
    last_car_pose = None
    border_thickness = 10

    # Main processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pre-process frame
        frame, frame_height, frame_width = process_frame(frame, crop_video)

        # Detect objects
        detected_objects = detect_objects(model, frame, selected_class_ids)

        # Find person and car objects
        person_obj = find_object_by_label(detected_objects, "person")
        car_obj = find_object_by_label(detected_objects, "car")

        # Get person information
        person_info, last_person_pose = get_object_info(person_obj, last_person_pose)

        # Get car information
        car_info, last_car_pose = get_object_info(car_obj, last_car_pose)

        # Draw person if detected
        if person_info:
            box, center, axes = person_info
            draw_object(frame, box, center, axes, "person", (255, 255, 0))
            person_center = center
        else:
            person_center = None

        # Draw car if detected
        if car_info:
            box, center, axes = car_info
            draw_object(frame, box, center, axes, "car", (255, 0, 0))
            car_center = center
        else:
            car_center = None

        # Calculate distance and draw line if both objects are detected
        if person_center and car_center:
            distance = calculate_distance(person_center, car_center)
            alert_color = draw_distance_line(
                frame, person_center, car_center, distance, distance_threshold
            )
            draw_alert_border(
                frame, frame_width, frame_height, alert_color, border_thickness
            )

        # Display output
        if display_output:
            should_continue = display_frame(frame)
            if not should_continue:
                break

    cap.release()
    if display_output:
        cv2.destroyAllWindows()


if __name__ == "__main__":

    video_path, model_path = get_paths()

    detect_pedestrian_accidents(
        video_path=video_path,
        model_path=model_path,
        selected_classes=["person", "car"],
        skip_frames=100,
        distance_threshold=150,
    )

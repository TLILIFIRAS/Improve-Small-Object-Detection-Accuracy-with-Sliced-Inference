import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics.utils.torch_utils import select_device

# Initialize your YOLOv8 model wrapped by SAHI AutoDetectionModel
device = select_device('0' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolov8s.pt",
    device=device
)

# Open video file
cap = cv2.VideoCapture("crowd.mp4")

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create VideoWriter for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("crowd_output_sahi.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # SAHI expects RGB images for prediction
    rgb_frame = frame[..., ::-1]

    # Run sliced prediction on frame
    results = get_sliced_prediction(
        image=rgb_frame,
        detection_model=model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # Draw predictions on the original BGR frame
    for det in results.object_prediction_list:
        bbox = det.bbox.to_voc_bbox()  # x1,y1,x2,y2 in VOC format
        score = det.score.value
        label = det.category.name

        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box and label with confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Write the annotated frame to the output video
    out.write(frame)

    # Optional: display the frame in a window
    # cv2.imshow("Crowd Detection (SAHI + YOLOv8)", frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8s.pt")

# Open the video file
cap = cv2.VideoCapture("crowd.mp4")

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID'
out = cv2.VideoWriter("crowd_output.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the current frame
    results = model(frame, verbose=False)[0]

    # Extract bounding boxes, class IDs, and confidence scores
    boxes = results.boxes

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().item()
        class_id = int(box.cls[0].cpu().item())
        label = model.names[class_id]

        # Draw bounding box and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Write the frame to the output video
    out.write(frame)

    # Show the frame (optional)
    #cv2.imshow("Crowd Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

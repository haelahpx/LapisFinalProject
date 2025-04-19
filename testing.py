from ultralytics import YOLO
import cv2
import easyocr
import os

# === CONFIG ===
MODEL_PATH = "vec-YOLO8/vec-YOLO8-RGB/weights/best.pt"
VIDEO_PATH = os.path.join("assets/videos", "JakartaSarinahCrossroadTrafficView.mp4")  # path to your video
OUTPUT_PATH = os.path.join("assets/predicts", "vec-YOLO8-RGB(JakartaSarinahCrossroadTrafficView).mp4")
CONFIDENCE_THRESHOLD = 0.25

# === LOAD MODEL & OCR ===
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'])

# === LOAD VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

frame_count = 0
total_cars = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_car_count = 0

    # === RUN PREDICTION ===
    results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, save=False, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if label.lower() == "car":
                current_car_count += 1
                total_cars += 1

                # Draw car bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # === OCR on the detected region ===
                plate_img = frame[y1:y2, x1:x2]
                ocr_result = reader.readtext(plate_img)

                plate_number = ""
                even_odd_result = "Unknown"

                if ocr_result:
                    plate_number = ocr_result[0][1].strip()

                    if len(plate_number) > 0:
                        last_char = plate_number[-1]
                        if last_char.isdigit():
                            last_digit = int(last_char)
                            even_odd_result = "Even" if last_digit % 2 == 0 else "Odd"

                    # Draw plate text and even/odd
                    display_text = f"{plate_number} ({even_odd_result})"
                    cv2.putText(frame, display_text, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # === Write frame to output ===
    out.write(frame)

    # Optional: Show live window (comment out if not needed)
    # cv2.imshow("Video Output", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Finished processing video: {VIDEO_PATH}")
print(f"🚗 Total cars detected in video: {total_cars}")

import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Can't receive frame")
        break

    cv2.imshow("Webcam Test", frame)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

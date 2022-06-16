from object_tracker import ObjectTracker
import cv2

OT = ObjectTracker()

cap = cv2.VideoCapture(0)
prev_frames = None
while cap.isOpened():
	
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue
    
    _, res, prev_frames = OT.process(frame, prev_frames)
    cv2.imshow('test', res)

    if cv2.waitKey(5) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
import cv2
import numpy as np
from ultralytics import YOLO
from dqn import DQNAgent

# Load YOLO model
model = YOLO("yolov8n.pt")   # tiny model for fast inference

# Load RL agent
agent = DQNAgent()
agent.load("results/dqn_final.pth")

# Open video
cap = cv2.VideoCapture("traffic.mp4")

# Define lane ROIs (you adjust based on your video)
lanes = {
    "north":   (100, 0, 300, 200),
    "south":   (100, 300, 300, 500),
    "east":    (300, 100, 500, 300),
    "west":    (0,   100, 200, 300)
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, verbose=False)[0]

    # Count cars in each lane
    lane_counts = {"north":0, "south":0, "east":0, "west":0}

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls != 2 and cls != 3 and cls != 5:  # classes: car, truck, bus
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        for lane, (xA,yA,xB,yB) in lanes.items():
            if xA <= cx <= xB and yA <= cy <= yB:
                lane_counts[lane] += 1

    # Prepare RL agent input
    state = np.array([
        lane_counts["north"],
        lane_counts["south"],
        lane_counts["east"],
        lane_counts["west"],
        0,   # phase dummy value
        0    # time_since_change dummy
    ], dtype=np.float32)

    state = state / 20  # normalize

    # RL decides action
    action = agent.act(state)

    # Display action
    if action == 0:
        decision = "KEEP CURRENT SIGNAL"
    else:
        decision = "SWITCH SIGNAL NOW"

    # Show video feed
    cv2.putText(frame, f"RL Decision: {decision}", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    # Draw lanes
    for lane, (xA,yA,xB,yB) in lanes.items():
        cv2.rectangle(frame, (xA,yA), (xB,yB), (255,0,0), 2)
        cv2.putText(frame, f"{lane}:{lane_counts[lane]}", (xA,yA-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow("Real-Time Traffic RL Demo", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


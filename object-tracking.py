import cv2
import numpy as np

cap = cv2.VideoCapture(0)

_, f = cap.read()
old_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

def select_point(event, x, y, flags, param):
    global point, point_selected, p0
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x,y)
        point_selected = True
        p0 = np.array([[x, y]], dtype=np.float32)

point = ()
point_selected = False
p0 = np.array([[]])

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", select_point)

trajectory_points = []

while(True):
    _, frame = cap.read()

    frame = cv2.flip(frame, 1)

    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if point_selected == True:
        cv2.circle(frame, point, 5, (0,0,255), -1)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.01))
        old_gray = new_gray.copy()
        p0 = new_points

        x, y = new_points.ravel()
        a, b = p0.ravel()
        x, y, a, b = int(x), int(y), int(a), int(b)

        trajectory_points.append((x,y))

        for point_ in trajectory_points:
            cv2.circle(frame, point_, 5, (0,255,0), -1)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

_, old = cap.read()

old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)

def select_point(event, x, y, flags, parm):
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)

point = ()
point_selected = False
old_points = np.array([[]])

lk_params = dict(winSize = (15, 15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01))

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", select_point)

while(True):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if point_selected == True:
        cv2.circle(frame, point, 5, (0,0,255), -1)
    
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None, **lk_params)
        old_gray = new_gray.copy()
        a,b = old_points.ravel().astype(np.uint8)
        old_points = new_points

        x, y = new_points.ravel().astype(np.uint8)
        cv2.circle(frame, (x,y), 5, (0,255,0), -1)
        cv2.arrowedLine(frame, (x,y), (a,b), (255,0,0), 2)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

_, f = cap.read()
prev_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

hsv_mask = np.zeros_like(f)

while(True):
    _, frame = cap.read()

    frame = cv2.flip(frame, 1)

    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    prev_gray = new_gray.copy()

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv_mask[..., 0] = ang * 180/np.pi / 2
    hsv_mask[..., 1] = 255
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    flow_bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    mask = mag > 10

    flow_bgr[~mask] = 0
    
    cv2.imshow("motion", flow_bgr)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
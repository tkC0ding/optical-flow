import cv2
import numpy as np
import pyautogui

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 64)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 64)

frame_count = 0
interval = 5
magnitude_clip = 2
scale_factor = 10

lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.01))
trajectories = []

while(True):
    _, frame = cap.read()
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if(len(trajectories) > 0):
        old_points = np.array(trajectories).reshape(-1,2)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None, **lk_params)
        old_points_pred, status, error = cv2.calcOpticalFlowPyrLK(new_gray, old_gray, new_points, None, **lk_params)
        d = abs(old_points - old_points_pred).max(-1)
        good = d < 1

        old_good_points = []
        new_good_points = []
        for trajectory, point, flag in zip(trajectories, points, good):
            if not flag:
                continue
            
            old_good_points.append(trajectory)
            new_good_points.append(point)
        
        old_good_points = np.array(old_good_points)
        new_good_points = np.array(new_good_points)
        
        motion_vector = np.mean(new_good_points-old_good_points, axis=0)
        magnitude = np.linalg.norm(motion_vector)
        if(magnitude > magnitude_clip):
            dx, dy = (motion_vector * scale_factor).astype(int)

            screen_w, screen_h = pyautogui.size()
            current_position_x, current_position_y = pyautogui.position()
            new_position_x = np.clip(current_position_x + dx, 0, screen_w - 1)
            new_position_y = np.clip(current_position_y + dy, 0, screen_h - 1)
            pyautogui.moveTo(new_position_x, new_position_y)
        
        trajectories = new_points.copy()

    if(frame_count % interval == 0):
        trajectories = []
        points = cv2.goodFeaturesToTrack(new_gray, 15, 0.01, 10, None, None, 7, True, 0.04)
        points = points.reshape(-1, 2)
        for (x,y) in points:
            trajectories.append([x,y])
        
    frame_count += 1
    old_gray = new_gray.copy()

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
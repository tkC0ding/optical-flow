import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lkparams = dict(winSize=(15,15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01))

trajectory_len = 30
frame_count = 0
interval = 5

trajectories = []

while(True):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if(len(trajectories) > 0):
        points_old = np.float32([[trajectory[-1][0], trajectory[-1][1]] for trajectory in trajectories])
        points_new, status, error = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, points_old, None, **lkparams)
        points_old_pred, status, error = cv2.calcOpticalFlowPyrLK(new_gray, old_gray, points_new, None, **lkparams)
        d = (points_old - points_old_pred).max(-1)
        good = d < 1

        new_trajectories = []
        for trajectory, (x,y), good_flag in zip(trajectories, points_new, good):
            if(good_flag == True):
                if(len(trajectory) <= trajectory_len):
                    trajectory.append((x,y))
                else:
                    del trajectory[0]
                new_trajectories.append(trajectory)
            else:
                continue

        trajectories = new_trajectories
        cv2.polylines(frame, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))


    if (frame_count % interval == 0):
        points = cv2.goodFeaturesToTrack(new_gray, 100, 0.01, blockSize=7, minDistance=10, k=0.04, useHarrisDetector=True)
        points = points.reshape(-1,2)

        for (x,y) in points:
            trajectories.append([(x,y)])
    
    old_gray = new_gray.copy()
    frame_count += 1

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
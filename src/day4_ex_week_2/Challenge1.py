"""
In this file i do the challenge part for the robot.mp4.

In this video the robots don't pass before each other, therefore
i can exclude detected feature of the optical flow using a distance
based technic.

"""

import cv2
import numpy as np

cap = cv2.VideoCapture('Robots.mp4')

thresh = 10
radius = 200

ret = True
lastFrame = None
lastFeatAVG = None


def euclidean_dist(a, b):
    return np.sqrt(np.sum((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def sparse_of(lastGray, gray):
    lastFeat = cv2.goodFeaturesToTrack(lastGray, maxCorners=100, qualityLevel=0.3, minDistance=7)
    feat, status, error = cv2.calcOpticalFlowPyrLK(lastGray, gray, lastFeat, None)

    feat_displ = []

    for i in range(len(lastFeat)):
        f10, f11 = int(lastFeat[i][0][0]), int(lastFeat[i][0][1])
        f20, f21 = int(feat[i][0][0]), int(feat[i][0][1])
        dist_squared = (f20 - f10) ** 2 + (f21 - f11) ** 2
        if dist_squared > thresh:
            feat_displ.append(((f10, f11), (f20, f21)))

    return feat_displ


def draw_features(feat, result):
    for (f10, f11), (f20, f21) in feat:
        cv2.line(result, (f10, f11), (f20, f21), (0, 255, 0), 2)
        cv2.circle(result, (f10, f11), 5, (0, 255, 0), -1)

    return result


while ret:
    ret, raw_frame = cap.read()
    if not ret:
        break

    frame = raw_frame.copy()

    if lastFrame is not None:
        lastGray = cv2.cvtColor(lastFrame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

        feat = sparse_of(lastGray, gray)

        if lastFeatAVG is not None:  # filter out points too far away
            feat = [f for f in feat if euclidean_dist(f[0], lastFeatAVG) < radius
                    and euclidean_dist(f[1], lastFeatAVG) < radius]

        if len(feat) > 0:
            lastFeatAVG = [sum([f[0][0] + f[1][0] for f in feat]) / len(feat) / 2,
                           sum([f[0][1] + f[1][1] for f in feat]) / len(feat) / 2]

        cv2.circle(frame, (int(lastFeatAVG[0]), int(lastFeatAVG[1])), radius, (0, 255, 0), 2)

    cv2.imshow('Robot', cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))

    lastFrame = raw_frame
    keys = cv2.waitKey(10) & 0xFF
    if keys == ord('q'):
        break
    elif keys == ord('s'):
        print('s is pressed')

cv2.destroyAllWindows()
cap.release()
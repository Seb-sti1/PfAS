import cv2

cap = cv2.VideoCapture('Robots.mp4')

thresh = 10

ret = True
lastFrame = None


def sparse_of(lastGray, gray, frame):
    result = frame.copy()

    lastFeat = cv2.goodFeaturesToTrack(lastGray, maxCorners=100, qualityLevel=0.3, minDistance=7)
    feat, status, error = cv2.calcOpticalFlowPyrLK(lastGray, gray, lastFeat, None)

    for i in range(len(lastFeat)):
        f10, f11 = int(lastFeat[i][0][0]), int(lastFeat[i][0][1])
        f20, f21 = int(feat[i][0][0]), int(feat[i][0][1])
        dist_squared = (f20 - f10) ** 2 + (f21 - f11) ** 2
        if dist_squared > thresh:
            cv2.line(result, (f10, f11), (f20, f21), (0, 255, 0), 2)
            cv2.circle(result, (f10, f11), 5, (0, 255, 0), -1)

    return result


def dense_flow(lastGray, gray, frame):
    flow = cv2.calcOpticalFlowFarneback(lastGray, gray, None, 0.5, 3, 15, 3, 5, 1.5, 0)
    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])

    # Doesn't show properly (not really visible)
    # alpha = 0.2
    # result = frame.copy()
    # result[:, :, 0] = (1 - alpha) * result[:, :, 0] + alpha * mag
    # result[:, :, 1] = (1 - alpha) * result[:, :, 1] + alpha * mag
    # result[:, :, 2] = (1 - alpha) * result[:, :, 2] + alpha * mag

    return mag


while ret:
    ret, frame = cap.read()
    cv2.imshow('Robot', cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))

    if lastFrame is not None:
        lastGray = cv2.cvtColor(lastFrame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Sparse flow', cv2.resize(sparse_of(lastGray, gray, frame), (0, 0), fx=0.5, fy=0.5))
        cv2.imshow('Dense flow', cv2.resize(dense_flow(lastGray, gray, frame), (0, 0), fx=0.5, fy=0.5))

    lastFrame = frame
    keys = cv2.waitKey(10) & 0xFF
    if keys == ord('q'):
        break
    elif keys == ord('s'):
        print('s is pressed')

cv2.destroyAllWindows()
cap.release()

# Sparse of is way faster than dense of although dense of detect the whole robot when it's moving

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

X = []

with open("clusters.txt") as f:
    for l in f.readlines():
        l = l.strip()
        X.append([int(i) for i in l.split(" ")])

n_l = [2, 3, 4, 5, 6, 7, 8, 9]

r = []

for n in n_l:
    km = KMeans(n_clusters=n)
    km.fit(X)
    r.append(km.inertia_)

print(r)

plt.plot(n_l, r)
plt.show()

x = []
y = []

with open("lr_x.txt") as f:
    x = [float(l.strip()) for l in f.readlines()]

with open("lr_y.txt") as f:
    y = [float(l.strip()) for l in f.readlines()]

lm = LinearRegression()
model = lm.fit(np.array(x).T.reshape(-1, 1), np.array(y).T)
a = model.coef_[0]  # Slope (coefficient)
b = model.intercept_  # Intercept

print(a, b)

import numpy as np
import cv2


def camera_to_world(rvec, tvec, P_cam):
    # Convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Compute the inverse rotation matrix
    R_inv = np.linalg.inv(R)

    # Apply the inverse transformation to find the point in world coordinates
    P_world = R_inv.dot(P_cam - tvec)

    return P_world


# Example usage:
rvec = np.array([-0.05, -1.51, -0.00])  # Rotation vector
tvec = np.array([87.39, -2.25, -24.89])  # Translation vector
P_cam = np.array([-6.71, 0.23, 21.59])  # Point in camera coordinates

P_world = camera_to_world(rvec, tvec, P_cam)
print("Point in world coordinates:", P_world)

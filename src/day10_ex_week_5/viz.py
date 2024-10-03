import open3d as o3d
import numpy as np
import time

# Initialize the visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Create a sample point cloud
pcd = o3d.geometry.PointCloud()

# Create some initial random points
points = np.random.rand(100, 3)
pcd.points = o3d.utility.Vector3dVector(points)

# Add the initial geometry to the visualizer
vis.add_geometry(pcd)

# Update point cloud dynamically
for i in range(50):
    # Generate new random points (simulating a new point cloud)
    new_points = np.random.rand(100, 3)

    # Update the points in the point cloud
    pcd.points = o3d.utility.Vector3dVector(new_points)

    # Update geometry and refresh the window
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # Optional delay to see the updates clearly
    time.sleep(0.1)

# Close the visualizer window after loop
vis.run()
vis.destroy_window()

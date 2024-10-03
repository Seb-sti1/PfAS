import os
import time

import numpy as np
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud
from tqdm import tqdm

from src.day9_ex_week_5.Exercise1 import draw_registrations
from src.day9_ex_week_5.Exercise2 import load_pcd


def compute_normals(pcd: PointCloud, custom=False):
    if custom:
        pcd.estimate_normals()
    else:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                                                 max_nn=30), fast_normal_computation=True)


def ransac(source: PointCloud, target: PointCloud,
           voxel_size=0.05):
    source_sample = source.voxel_down_sample(voxel_size)
    target_sample = target.voxel_down_sample(voxel_size)

    source_fpfh = (o3d.pipelines.registration
                   .compute_fpfh_feature(source_sample,
                                         o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                                                                              max_nn=100)))
    target_fpfh = (o3d.pipelines.registration
                   .compute_fpfh_feature(target_sample,
                                         o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                                                                              max_nn=100)))

    # RANSAC
    point_to_point = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_sample, target_sample,
        source_fpfh, target_fpfh,
        True, voxel_size * 1.5,
        point_to_point
    )
    print(ransac_result)

    return ransac_result


def icp(source, target, threshold, trans_init=None):
    # Parameters
    if trans_init is None:
        trans_init = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Evaluate registration
    evaluation = (o3d.pipelines.registration
                  .evaluate_registration(source, target, threshold, trans_init))
    print(evaluation)

    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                                             max_nn=30), fast_normal_computation=True)
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                                             max_nn=30), fast_normal_computation=True)

    point_to_plane = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        point_to_plane)
    print(icp_result)

    return icp_result


"""
### Task 1
Today, your task is to implement a global registration algorithm.

It should be able to roughly align two point clouds.
Implement the global registration, and then try the following:

1. Can you fit `r1.pcd` and `r2.pcd`?
2. Can you fit `car1.ply` and `car2.ply`?
The corresponding files are in the `global_registration` folder.
"""

if __name__ == "__main__":
    q = "2."

    if q == "1.1":
        r1 = o3d.io.read_point_cloud('global_registration/r1.pcd')
        r2 = o3d.io.read_point_cloud('global_registration/r2.pcd')
        r = ransac(r1, r2)
        draw_registrations(r1, r2, r.transformation, recolor=True)

    if q == "1.2":
        t = time.time()
        car1 = o3d.io.read_point_cloud('global_registration/car1.ply')
        car2 = o3d.io.read_point_cloud('global_registration/car2.ply')
        compute_normals(car1)
        compute_normals(car2)

        r = ransac(car1, car2)
        # draw_registrations(car1, car2, r.transformation, recolor=True)

        source_transformed = car1.transform(r.transformation)
        compute_normals(source_transformed)
        r = icp(source_transformed, car2, 0.2)
        # draw_registrations(car1_transformed, car2, r.transformation, recolor=True)

        print(time.time() - t)

    if q == "2.":
        images = [name.split(".")[0] for name in os.listdir("car_challange/rgb")]
        images = sorted(images, key=int)
        step = 40
        n = min(step * 40, len(images))

        voxel_size = 0.01

        # load first pcd
        source = load_pcd(images[0],
                          "car_challange/rgb/%s.jpg",
                          "car_challange/depth/%s.png")
        source = source.voxel_down_sample(voxel_size=voxel_size)
        compute_normals(source)

        progress = tqdm(images[step:n:step])
        for idx, name in enumerate(progress):
            progress.set_description(f"Processing {name}")

            # load next point cloud
            target = load_pcd(name,
                              "car_challange/rgb/%s.jpg",
                              "car_challange/depth/%s.png")
            target = target.voxel_down_sample(voxel_size=voxel_size)
            compute_normals(target)
            draw_registrations(source, target, recolor=True)

            # global match
            r = ransac(source, target)
            source_transformed = source.transform(r.transformation)
            compute_normals(source_transformed)
            draw_registrations(source_transformed, target, recolor=True)

            # local match
            r = icp(source_transformed, target, 0.2)
            source_transformed = source_transformed.transform(r.transformation)

            # visualization
            draw_registrations(source_transformed, target, recolor=True)

            # merge together
            source = source_transformed.transform(r.transformation) + target
            source.voxel_down_sample(voxel_size=voxel_size)
            compute_normals(source)

        o3d.visualization.draw_geometries([source,
                                           o3d.geometry.TriangleMesh().create_coordinate_frame(0.5, [0., 0., 0.])])

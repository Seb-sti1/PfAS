import numpy as np
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud

from src.day9_ex_week_5.Exercise1 import draw_registrations


def ransac(source: PointCloud, target: PointCloud,
           custom_normals=True,
           voxel_size=0.05, checkers_l=None):
    source_sample = source.voxel_down_sample(voxel_size)
    target_sample = target.voxel_down_sample(voxel_size)
    if custom_normals:
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                                                 max_nn=30), fast_normal_computation=True)
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                                                 max_nn=30), fast_normal_computation=True)
    else:
        source_sample.estimate_normals()
        target_sample.estimate_normals()

    # features
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


def icp(source, target, threshold, trans_init,
        showpc=False, showicp=True,
        custom_normal=False):
    # Parameters
    if trans_init is None:
        trans_init = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Evaluate registration
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
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
    q = "1.2"

    if q == "1.1":
        r1 = o3d.io.read_point_cloud('global_registration/r1.pcd')
        r2 = o3d.io.read_point_cloud('global_registration/r2.pcd')
        r = ransac(r1, r2)
        draw_registrations(r1, r2, r.transformation, recolor=True)

    if q == "1.2":
        car1 = o3d.io.read_point_cloud('global_registration/car1.ply')
        car2 = o3d.io.read_point_cloud('global_registration/car2.ply')
        r = ransac(car1, car2)
        draw_registrations(car1, car2, r.transformation, recolor=True)

        r = ransac(car1, car2, custom_normals=False)
        draw_registrations(car1, car2, r.transformation, recolor=True)

    if q == "2.":
        pass

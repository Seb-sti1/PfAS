import copy

import open3d as o3d
from open3d.cpu.pybind.t.geometry import PointCloud


# helper function for drawing
# If you want it to be more clear set recolor=True
def draw_registrations(source, target, transformation=None, recolor=False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if recolor:
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def full_ransac(source: PointCloud, target: PointCloud, show=False, checkers_l=None):
    if show:
        draw_registrations(source, target)

    # down sampling
    voxel_size = 0.05
    source_sample = source.voxel_down_sample(voxel_size)
    target_sample = target.voxel_down_sample(voxel_size)
    source_sample.estimate_normals()
    target_sample.estimate_normals()

    # features
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source_sample,
                                                                  o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                                                                                                       max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target_sample,
                                                                  o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                                                                                                       max_nn=100))

    # RANSAC
    point_to_point = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)

    if checkers_l is not None:
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_sample, target_sample,
            source_fpfh, target_fpfh,
            True, voxel_size * 1.5,
            point_to_point,
            checkers=checkers_l
        )
    else:

        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_sample, target_sample,
            source_fpfh, target_fpfh,
            True, voxel_size * 1.5,
            point_to_point
        )
    print(ransac_result)
    # show result
    draw_registrations(source_sample, target_sample, ransac_result.transformation, True)


# Example (1.1)

source = o3d.io.read_point_cloud("ICP/r1.pcd")
target = o3d.io.read_point_cloud("ICP/r2.pcd")

full_ransac(source, target)

# Exercise 1 (1.2)

source = o3d.io.read_point_cloud("ICP/r1.pcd")
target = o3d.io.read_point_cloud("ICP/r3.pcd")

full_ransac(source, target)

# Exercise B (1.3)

corr_length = 0.9
voxel_size = 0.05
distance_threshold = voxel_size * 1.5
c0 = o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(corr_length)
c1 = o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
c2 = o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(0.095)

checker_list = [c0, c1, c2]

full_ransac(source, target, checkers_l=checker_list)

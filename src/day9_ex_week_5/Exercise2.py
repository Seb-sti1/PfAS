import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


# Helper function to draw registrations (reccomended)
def draw_registrations(source, target, transformation=None, recolor=False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if recolor:
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        source_temp.transform(transformation)

    o3d.visualization.draw_geometries([source_temp, target_temp,
                                       o3d.geometry.TriangleMesh().create_coordinate_frame(0.5, [0., 0., 0.])])


def load_pcd(name, pyplot=False):
    color_raw = o3d.io.read_image(f"RGBD/color/{name}.jpg")
    depth_raw = o3d.io.read_image(f"RGBD/depth/{name}.png")

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,
        depth_raw,
        convert_rgb_to_intensity=True)

    if pyplot:
        fig = plt.figure(figsize=(15, 15))
        plt.subplot(221)
        plt.title('Redwood grayscale0 image')
        plt.imshow(rgbd_image.color)

        plt.subplot(222)
        plt.title('Redwood depth0 image')
        plt.imshow(rgbd_image.depth)

        plt.show()

    camera = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, camera)

    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd


def compute_transformation(source, target, threshold, trans_init,
                           showpc=False, showicp=True,
                           custom_normal=False):
    # Draw
    if showpc:
        draw_registrations(source, target, recolor=True)

    # Parameters
    trans_init = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Evaluate registration
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)

    if custom_normal:
        source.estimate_normals()
        target.estimate_normals()
    else:
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

    print(icp_result.transformation)
    if showicp:
        draw_registrations(source, target, icp_result.transformation, True)

    return icp_result


if __name__ == '__main__':
    threshold = .02

    # Exercise A
    if False:
        I = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        threshold = .02
        compute_transformation(load_pcd("000000"), load_pcd("000005"), threshold, I)
        compute_transformation(load_pcd("000000"), load_pcd("000300"), threshold, I)

    # Exercise B

    if False:
        threshold = 0.2
        T = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0.2, 0, 0, 1]]
        compute_transformation(load_pcd("000000"), load_pcd("000300"), threshold, T)

    # Exercise C

    if False:
        threshold = 0.2
        compute_transformation(load_pcd("000000"), load_pcd("000300"), threshold, I,
                               custom_normal=True)

    # Exercise D
    # Extremely slow, couldn't optimize it
    I = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    threshold = 0.2

    images = [name.split(".")[0] for name in os.listdir("RGBD/color")]

    source = load_pcd(images[0])

    for name in images[1:2]:
        name = name.split(".")[0]
        target = load_pcd(name)
        r = compute_transformation(source, target, threshold, I,
                                   showicp=True,
                                   custom_normal=True)
        source = source.transform(r.transformation) + target
        source.voxel_down_sample(voxel_size=0.05)

    o3d.visualization.draw_geometries([source,
                                       o3d.geometry.TriangleMesh().create_coordinate_frame(0.5, [0., 0., 0.])])

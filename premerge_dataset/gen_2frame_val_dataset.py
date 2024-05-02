import os
import numpy as np
import torch
import math
import shutil

# Load ground truth poses from file.
#  - pose_path: Complete filename for the pose file
def load_poses(pose_path):
    all_poses = []
    try:
        with open(pose_path, "r") as f:
            lines = f.readlines()
            # Iterate the file line by line
            for line in lines:
                act_poses = np.fromstring(line, dtype=float, sep=" ")
                if len(act_poses) == 12:
                    act_poses = act_poses.reshape(3, 4)
                    act_poses = np.vstack((act_poses, [0, 0, 0, 1]))
                elif len(act_poses) == 16:
                    act_poses = act_poses.reshape(4, 4)
                all_poses.append(act_poses)
    except FileNotFoundError:
        print("Ground truth poses are not avaialble.")
    # Return a numpy array of size nx4x4
    #  - n: poses
    #  - 4x4: transformation matrices
    return np.array(all_poses)

# Load calibrations from file.
#  - calib_path: Complete filename for the calib file
def load_calib(calib_path):
    T_cam_velo = []
    try:
        with open(calib_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Tr:" in line:
                    line = line.replace("Tr:", "")
                    T_cam_velo = np.fromstring(line, dtype=float, sep=" ")
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))
    except FileNotFoundError:
        print("Calibrations are not avaialble.")
    return np.array(T_cam_velo)

def read_poses(path_to_seq):
    pose_file = os.path.join(path_to_seq, "poses.txt")
    calib_file = os.path.join(path_to_seq, "calib.txt")
    poses = np.array(load_poses(pose_file))
    inv_frame0 = np.linalg.inv(poses[0])

    # load calibrations
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # convert kitti poses from camera coord to LiDAR coord
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    poses = np.array(new_poses)
    return poses

def pre_merge_2frame_val(original_idx, velo_paths, labels_paths,poses):
    merge_size = 2
    seq_size = 4071
    all_transformed_points = None
    all_gt_labels = None
    for act_idx in range(merge_size):
        act_idx = original_idx - math.floor(merge_size/2) + act_idx
        if act_idx < 0 or act_idx >= seq_size: continue
        raw_velo_data = np.fromfile(velo_paths[act_idx], dtype=np.float32).reshape((-1, 4))
        tensor_raw_velo_data = torch.tensor(raw_velo_data)
        tensor_points = tensor_raw_velo_data[:,:3]
        remissions = tensor_raw_velo_data[:,3]
        remissions = remissions.unsqueeze(1)
        from_pose = poses[act_idx]
        to_pose = poses[original_idx]
        transformed_tensor_points = transform_point_cloud(tensor_points, from_pose, to_pose)
        transformed_tensor_points = torch.cat((transformed_tensor_points,remissions),1)
        transformed_points = np.array(transformed_tensor_points)
        if all_transformed_points is None: all_transformed_points = transformed_points
        else: all_transformed_points = np.append(all_transformed_points, transformed_points, axis=0)
        gt_labels= np.fromfile(labels_paths[act_idx], dtype=np.uint32).reshape((-1, 1))
        if all_gt_labels is None: all_gt_labels = gt_labels
        else: all_gt_labels = np.append(all_gt_labels, gt_labels, axis=0)
    velo_path = os.path.join("output","velodyne")
    if not os.path.exists(velo_path):os.makedirs(velo_path)
    all_transformed_points.tofile(os.path.join(velo_path, str(original_idx).zfill(6)+".bin"))
    labels_path = os.path.join("output","labels")
    if not os.path.exists(labels_path):os.makedirs(labels_path)
    all_gt_labels.tofile(os.path.join(labels_path, str(original_idx).zfill(6)+".label"))
    
def transform_point_cloud(past_point_clouds, from_pose, to_pose):
    transformation = torch.Tensor(np.linalg.inv(to_pose) @ from_pose)
    NP = past_point_clouds.shape[0]
    xyz1 = torch.hstack((past_point_clouds, torch.ones(NP, 1))).T
    past_point_clouds = (transformation @ xyz1).T[:, :3]
    return past_point_clouds

def file_paths(directory):
    paths = []
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            paths.append(os.path.join(dirpath, f))
    return paths

shutil.copy(os.path.join("input", "poses.txt"), "output")
print("copy poses - done")
shutil.copy(os.path.join("input", "calib.txt"), "output")
print("copy calib - done")
shutil.copytree(os.path.join("input", "image_2"), os.path.join("output","image_2"))
print("copy image_2 - done")

velo_paths = file_paths(os.path.join("input","velodyne"))
labels_paths = file_paths(os.path.join("input","labels"))
poses = read_poses("input")

for i in range(4071):
    pre_merge_2frame_val(i,velo_paths,labels_paths,poses)
    if i%100 == 0: print("gen premerge val dataset","-",i,"done")

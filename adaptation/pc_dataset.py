import os
import yaml
import numpy as np

from PIL import Image
from torch.utils import data
from pathlib import Path
from nuscenes.utils import splits

# Update ------------------------------------------------------------------------------------------
import torch
import math
import re
# -------------------------------------------------------------------------------------------------

REGISTERED_PC_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]


def absoluteFilePaths(directory, num_vote):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            for _ in range(num_vote):
                yield os.path.abspath(os.path.join(dirpath, f))


@register_dataset
class SemanticKITTI(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)

        self.config = config
        self.num_vote = num_vote
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset

        if imageset == 'train':
            split = semkittiyaml['split']['train']
            if config['train_params'].get('trainval', False):
                split += semkittiyaml['split']['valid']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

# Update ------------------------------------------------------------------------------------------
        if self.config["frame_num"] > 1:

            self.poses = np.array([])

            root_path = os.path.join("dataset", "SemanticKitti", "dataset", "sequences")
            for seq_num in split:
                seq_str = "{0:02d}".format(int(seq_num))
                seq_path = os.path.join(root_path, seq_str)
                act_poses = self.read_poses(seq_path)
                self.poses = (act_poses if len(self.poses) == 0 else np.append(self.poses, act_poses, axis=0))
# -------------------------------------------------------------------------------------------------

        self.im_idx = []
        self.proj_matrix = {}

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']), num_vote)
            calib_path = os.path.join(data_path, str(i_folder).zfill(2), "calib.txt")
            calib = self.read_calib(calib_path)
            proj_matrix = np.matmul(calib["P2"], calib["Tr"])
            self.proj_matrix[i_folder] = proj_matrix

        seg_num_per_class = config['dataset_params']['seg_labelweights']
        seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
        self.seg_labelweights = np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

        return calib_out

# Update ------------------------------------------------------------------------------------------

    def to_sparse(self, array):
        sparse = self.config["sparse"]
        if (sparse=='1/1'): return array
        try:
            assert re.search("[0-9]+/[0-9]+", sparse)
            act_slice = int(sparse.split("/",1)[0])
            slice_num = int(sparse.split("/",1)[1])
            assert slice_num >= act_slice
            assert slice_num > 0
            assert act_slice > 0
        except AssertionError:
            print("SPARSE-WARNING - Sparse setting is incorrect! -> Use the full PointCloud")
            return array
        return array[act_slice-1::slice_num]

    # Load ground truth poses from file.
    #  - pose_path: Complete filename for the pose file
    def load_poses(self, pose_path):
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
    def load_calib(self, calib_path):
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

    def read_poses(self, path_to_seq):
        pose_file = os.path.join(path_to_seq, "poses.txt")
        calib_file = os.path.join(path_to_seq, "calib.txt")
        poses = np.array(self.load_poses(pose_file))
        inv_frame0 = np.linalg.inv(poses[0])

        # load calibrations
        T_cam_velo = self.load_calib(calib_file)
        T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
        T_velo_cam = np.linalg.inv(T_cam_velo)

        # convert kitti poses from camera coord to LiDAR coord
        new_poses = []
        for pose in poses:
            new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
        poses = np.array(new_poses)
        return poses

    def transform_point_cloud(self, past_point_clouds, from_pose, to_pose):
        transformation = torch.Tensor(np.linalg.inv(to_pose) @ from_pose)
        NP = past_point_clouds.shape[0]
        xyz1 = torch.hstack((past_point_clouds, torch.ones(NP, 1))).T
        past_point_clouds = (transformation @ xyz1).T[:, :3]
        return past_point_clouds

    def one_frame(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        origin_len = len(raw_data)
        points = raw_data[:, :3]

        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
            instance_label = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            instance_label = annotated_data >> 16
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

            if self.config['dataset_params']['ignore_label'] != 0:
                annotated_data -= 1
                annotated_data[annotated_data == -1] = self.config['dataset_params']['ignore_label']

        image_file = self.im_idx[index].replace('velodyne', 'image_2').replace('.bin', '.png')
        image = Image.open(image_file)
        proj_matrix = self.proj_matrix[int(self.im_idx[index][-22:-20])]

        data_dict = {}
        data_dict['xyz'] = points
        data_dict['labels'] = annotated_data.astype(np.uint8)
        data_dict['instance_label'] = instance_label
        data_dict['signal'] = raw_data[:, 3:4]
        data_dict['origin_len'] = origin_len
        data_dict['img'] = image
        data_dict['proj_matrix'] = proj_matrix

        return data_dict
    
    def multiple_frame(self, index):
        frame_num = self.config["frame_num"]
        step_num = self.config["step_num"]
        all_points = np.array([])
        all_raw_data = np.array([])

        for frame_idx in range(frame_num):
            frame_idx = index - (step_num*math.floor(frame_num/2)) + (step_num*frame_idx)
            if frame_idx < 0 or frame_idx >= len(self.poses): continue
            raw_data = np.fromfile(self.im_idx[frame_idx], dtype=np.float32).reshape((-1, 4))
            raw_data = self.to_sparse(raw_data)
            all_raw_data = (raw_data if len(all_raw_data) == 0 else np.append(all_raw_data, raw_data, axis=0))
            raw_data = torch.tensor(raw_data)
            points = raw_data[:,:3]
            from_pose = self.poses[frame_idx]
            to_pose = self.poses[index]
            t_points = self.transform_point_cloud(points, from_pose, to_pose)
            all_points = (t_points if len(all_points) == 0 else np.append(all_points, t_points, axis=0))
        
        origin_len = len(all_points)

        if self.imageset == 'test':
            all_annotated_data = np.expand_dims(np.zeros_like(all_raw_data[:, 0], dtype=int), axis=1)
            all_instance_label = np.expand_dims(np.zeros_like(all_raw_data[:, 0], dtype=int), axis=1)
        else:
            all_annotated_data = np.array([])
            all_instance_label = np.array([])
            for label_idx in range(frame_num):
                label_idx = index - (step_num*math.floor(frame_num/2)) + (step_num*label_idx)
                if label_idx < 0 or label_idx >= len(self.poses): continue

                annotated_data = np.fromfile(self.im_idx[label_idx].replace('velodyne', 'labels')[:-3] + 'label', 
                                             dtype=np.uint32).reshape((-1, 1))
                annotated_data = self.to_sparse(annotated_data)
                instance_label = annotated_data >> 16
                annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
                annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
                if self.config['dataset_params']['ignore_label'] != 0:
                    annotated_data -= 1
                    annotated_data[annotated_data == -1] = self.config['dataset_params']['ignore_label']

                all_annotated_data = (annotated_data if len(all_annotated_data) == 0 else np.concatenate((all_annotated_data, annotated_data)))
                all_instance_label = (instance_label if len(all_instance_label) == 0 else np.concatenate((all_instance_label, instance_label)))

        image_file = self.im_idx[index].replace('velodyne', 'image_2').replace('.bin', '.png')
        image = Image.open(image_file)
        proj_matrix = self.proj_matrix[int(self.im_idx[index][-22:-20])]

        data_dict = {}
        data_dict['xyz'] = np.array(all_points)
        data_dict['labels'] = all_annotated_data.astype(np.uint8)
        data_dict['instance_label'] = all_instance_label
        data_dict['signal'] = all_raw_data[:, 3:4]
        data_dict['origin_len'] = origin_len
        data_dict['img'] = image
        data_dict['proj_matrix'] = proj_matrix

        return data_dict

    def __getitem__(self, index):
        if self.config["frame_num"] > 1: data_dict = self.multiple_frame(index)
        else: data_dict = self.one_frame(index)
        return data_dict, self.im_idx[index]

# -------------------------------------------------------------------------------------------------

@register_dataset
class nuScenes(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        if config.debug:
            version = 'v1.0-mini'
            scenes = splits.mini_train
        else:
            if imageset != 'test':
                version = 'v1.0-trainval'
                if imageset == 'train':
                    scenes = splits.train
                else:
                    scenes = splits.val
            else:
                version = 'v1.0-test'
                scenes = splits.test

        self.split = imageset
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self.num_vote = num_vote
        self.data_path = data_path
        self.imageset = imageset
        self.img_view = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
                         'CAM_FRONT_LEFT']

        from nuscenes import NuScenes
        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

        self.get_available_scenes()
        available_scene_names = [s['name'] for s in self.available_scenes]
        scenes = list(filter(lambda x: x in available_scene_names, scenes))
        scenes = set([self.available_scenes[available_scene_names.index(s)]['token'] for s in scenes])
        self.get_path_infos_cam_lidar(scenes)

        print('Total %d scenes in the %s split' % (len(self.token_list), imageset))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.token_list)

    def loadDataByIndex(self, index):
        lidar_sample_token = self.token_list[index]['lidar_token']
        lidar_path = os.path.join(self.data_path,
                                  self.nusc.get('sample_data', lidar_sample_token)['filename'])
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))

        if self.split == 'test':
            self.lidarseg_path = None
            annotated_data = np.expand_dims(
                np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            lidarseg_path = os.path.join(self.data_path,
                                         self.nusc.get('lidarseg', lidar_sample_token)['filename'])
            annotated_data = np.fromfile(
                lidarseg_path, dtype=np.uint8).reshape((-1, 1))  # label

        pointcloud = raw_data[:, :4]
        sem_label = annotated_data
        inst_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        return pointcloud, sem_label, inst_label, lidar_sample_token

    def labelMapping(self, sem_label):
        sem_label = np.vectorize(self.map_name_from_general_index_to_segmentation_index.__getitem__)(
            sem_label)  # n, 1
        assert sem_label.shape[-1] == 1
        sem_label = sem_label[:, 0]
        return sem_label

    def loadImage(self, index, image_id):
        cam_sample_token = self.token_list[index]['cam_token'][image_id]
        cam = self.nusc.get('sample_data', cam_sample_token)
        image = Image.open(os.path.join(self.nusc.dataroot, cam['filename']))
        return image, cam_sample_token

    def get_available_scenes(self):
        # only for check if all the files are available
        self.available_scenes = []
        for scene in self.nusc.scene:
            scene_token = scene['token']
            scene_rec = self.nusc.get('scene', scene_token)
            sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            has_more_frames = True
            scene_not_exist = False
            while has_more_frames:
                lidar_path, _, _ = self.nusc.get_sample_data(sd_rec['token'])
                if not Path(lidar_path).exists():
                    scene_not_exist = True
                    break
                else:
                    break

            if scene_not_exist:
                continue
            self.available_scenes.append(scene)

    def get_path_infos_cam_lidar(self, scenes):
        self.token_list = []

        for sample in self.nusc.sample:
            scene_token = sample['scene_token']
            lidar_token = sample['data']['LIDAR_TOP']  # 360 lidar

            if scene_token in scenes:
                for _ in range(self.num_vote):
                    cam_token = []
                    for i in self.img_view:
                        cam_token.append(sample['data'][i])
                    self.token_list.append(
                        {'lidar_token': lidar_token,
                         'cam_token': cam_token}
                    )

    def __getitem__(self, index):
        pointcloud, sem_label, instance_label, lidar_sample_token = self.loadDataByIndex(index)
        sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)

        # get image feature
        image_id = np.random.randint(6)
        image, cam_sample_token = self.loadImage(index, image_id)

        cam_path, boxes_front_cam, cam_intrinsic = self.nusc.get_sample_data(cam_sample_token)
        pointsensor = self.nusc.get('sample_data', lidar_sample_token)
        cs_record_lidar = self.nusc.get('calibrated_sensor',
                                        pointsensor['calibrated_sensor_token'])
        pose_record_lidar = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        cam = self.nusc.get('sample_data', cam_sample_token)
        cs_record_cam = self.nusc.get('calibrated_sensor',
                                      cam['calibrated_sensor_token'])
        pose_record_cam = self.nusc.get('ego_pose', cam['ego_pose_token'])

        calib_infos = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam['translation'],
            "ego2global_rotation_cam": pose_record_cam['rotation'],
            "cam2ego_translation": cs_record_cam['translation'],
            "cam2ego_rotation": cs_record_cam['rotation'],
            "cam_intrinsic": cam_intrinsic,
        }

        data_dict = {}
        data_dict['xyz'] = pointcloud[:, :3]
        data_dict['img'] = image
        data_dict['calib_infos'] = calib_infos
        data_dict['labels'] = sem_label.astype(np.uint8)
        data_dict['signal'] = pointcloud[:, 3:4]
        data_dict['origin_len'] = len(pointcloud)

        return data_dict, lidar_sample_token


def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    return SemKITTI_label_name


from pathlib import Path
import json

import numpy as np
import torch


class ShapeNetParts(torch.utils.data.Dataset):
    num_classes = 50  # We have 50 parts classes to segment
    num_points = 1024
    dataset_path = Path("exercise_2/data/shapenetcore_partanno_segmentation_benchmark_v0/")  # path to point cloud data
    class_name_mapping = json.loads(Path("exercise_2/data/shape_parts_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())
    part_id_to_overall_id = json.loads(Path.read_text(Path(__file__).parent.parent / 'data' / 'partid_to_overallid.json'))

    def __init__(self, split):
        assert split in ['train', 'val', 'overfit']

        self.items = Path(f"exercise_2/data/splits/shapenet_parts/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        item = self.items[index]

        pointcloud, segmentation_labels = ShapeNetParts.get_point_cloud_with_labels(item)

        return {
            'points': pointcloud,
            'segmentation_labels': segmentation_labels
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['points'] = batch['points'].to(device)
        batch['segmentation_labels'] = batch['segmentation_labels'].to(device)

    @staticmethod
    def get_point_cloud_with_labels(shapenet_id):
        """
        Utility method for reading a ShapeNet point cloud from disk, reads points from pts files on disk as 3d numpy arrays, together with their per-point part labels
        :param shapenet_id: Shape ID of the form <shape_class>/<shape_identifier>, e.g. 03001627/f913501826c588e89753496ba23f2183
        :return: tuple: a numpy array representing the point cloud, in shape 3x1024, and the segmentation labels, as numpy array in shape 1024
        """
        category_id, shape_id = shapenet_id.split('/')

        # TODO: Load point cloud and segmentation labels, subsample to 1024 points. Make sure points and labels still correspond afterwards!
        
        # Read point cloud
        pc_filename = "exercise_2/data/shapenetcore_partanno_segmentation_benchmark_v0/{}/points/{}.pts".format(category_id, shape_id)

        with open(pc_filename, 'r') as pc_file:
            pc = pc_file.readlines()
            pc = [[float(k) for k in i.split(" ")] for i in pc]
            pc = np.array(pc)


        # Downsample point cloud to 1024 points
        idx = np.random.choice(pc.shape[0], size=1024)
        pc = pc[idx]

        # Read the labels, and get the corresponding ones for the downsampled point cloud
        label_filename = "exercise_2/data/shapenetcore_partanno_segmentation_benchmark_v0/{}/points_label/{}.seg".format(category_id, shape_id)
        
        with open(label_filename, 'r') as label_file:
            label = label_file.readlines()
            label = [int(i) for i in label]
            label = np.array(label)
        
        label = label[idx]

        # TODO: Important: Use ShapeNetParts.part_id_to_overall_id to convert the part labels you get from the .seg files from local to global ID as they start at 0 for each shape class whereas we want to predict the overall part class.
        # ShapeNetParts.part_id_to_overall_id converts an ID in form <shapenetclass_partlabel> to and integer representing the global part class id
        
        # Map each local ID in labels to their corresponding global ID
        # global_part_class_id = ShapeNetParts.part_id_to_overall_id[category_id]
        global_labels = [ShapeNetParts.part_id_to_overall_id[f"{category_id}_{class_id}"] for class_id in label.tolist()]
        global_labels = np.array(global_labels)

        pc = np.array(pc, dtype=np.float32)
        return pc.T, global_labels

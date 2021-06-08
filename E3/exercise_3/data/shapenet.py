from pathlib import Path
import json

import numpy as np
import torch


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8
    dataset_sdf_path = Path("exercise_3/data/shapenet_dim32_sdf")  # path to voxel data
    dataset_df_path = Path("exercise_3/data/shapenet_dim32_df")  # path to voxel data
    class_name_mapping = json.loads(Path("exercise_3/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.truncation_distance = 3

        self.items = Path(f"exercise_3/data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        sdf_id, df_id = self.items[index].split(' ')

        input_sdf = ShapeNet.get_shape_sdf(sdf_id)
        target_df = ShapeNet.get_shape_df(df_id)

        # TODO Apply truncation to sdf and df
        # TODO Stack (distances, sdf sign) for the input sdf
        # TODO Log-scale target df
        
        # Truncation
        input_sdf = np.clip(input_sdf, -3, 3)
        target_df = np.clip(target_df, -3, 3)
        
        # Separation and stacking
        sdf_ch_1 = np.where(input_sdf>=0, 1, -1)
        input_sdf = np.abs(input_sdf)
        input_sdf = np.stack([input_sdf, sdf_ch_1])
        
        # Log scale 
        target_df = np.log(target_df + 1)

        return {
            'name': f'{sdf_id}-{df_id}',
            'input_sdf': input_sdf,
            'target_df': target_df
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        # TODO add code to move batch to device
        batch["input_sdf"] = batch["input_sdf"].to(device).float()
        batch["target_df"] = batch["target_df"].to(device).float()
        return

    @staticmethod
    def get_shape_sdf(shapenet_id):
        sdf = None        
        # TODO implement sdf data loading
        path = Path(f"exercise_3/data/shapenet_dim32_sdf/{shapenet_id}.sdf")
        dims = np.fromfile(path, dtype=np.uint64, count=3)
        item_size = np.product(dims)
        sdf = np.fromfile(path, dtype=np.float32, offset=24, count=item_size)
        sdf = sdf.reshape(dims)
        return sdf

    @staticmethod
    def get_shape_df(shapenet_id):
        df = None
        # TODO implement df data loading
        path = Path(f"exercise_3/data/shapenet_dim32_df/{shapenet_id}.df")
        dims = np.fromfile(path, dtype=np.uint64, count=3)
        item_size = np.product(dims)
        df = np.fromfile(path, dtype=np.float32, offset=24, count=item_size)
        df = df.reshape(dims)
        df = np.clip(df, -3, 3)
        
        return df

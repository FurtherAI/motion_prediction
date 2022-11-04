from pathlib import Path
import math
import numpy as np
import torch as torch
from torch.utils.data import Dataset
import data_utils
import pandas as pd


import av2.datasets.motion_forecasting.scenario_serialization as scenario_utils
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario
from av2.map.lane_segment import LaneSegment, LaneMarkType
from av2.map.map_api import ArgoverseStaticMap
import av2.geometry.interpolate as interp_utils

import multiprocessing as mp

class AV2(Dataset):
    def __init__(self, root, split="train", pts_per_pl=64, sec_history=2, sec_future=3):
        self.pts_per_pl = pts_per_pl
        self.timesteps_history = sec_history * 10
        self.timesteps_future = sec_future * 10

        self.split = split
        self.root = Path(root)
        self.scenarios_path = self.root / (split + "_scenarios.txt")
        self.dataset_path = self.root / split

        with open(self.scenarios_path, 'r') as data:
            self.scenarios = data.readlines()
            if split == "train":
                del self.scenarios[40578]
                del self.scenarios[145742]  # both not getting any track data past filter_tracks (0 samples)
            self.scenarios = np.array(self.scenarios, dtype=np.bytes_)

    def map_path(self, root : Path, folder : str) -> Path:
        return root / folder / ("log_map_archive_" + folder + ".json")

    def scenario_path(self, root : Path, folder : str) -> Path:
        return root / folder / ("scenario_" + folder + ".parquet")

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        example = self.scenarios[idx].decode('UTF-8').strip()
        input_path = self.dataset_path / example / "pcurvenet_input.npy"
        cls_label_path = self.dataset_path / example / "pcurvenet_cls_label.npy"
        gt_traj_path = self.dataset_path / example / "pcurvenet_gt_trajectories.npy"
        num_lane_pls_path = self.dataset_path / example / "pcurvenet_num_lane_pls.npy"
        input_pls = np.load(input_path, fix_imports=False)
        cls_labels = np.load(cls_label_path, fix_imports=False)
        gt_traj = np.load(gt_traj_path, fix_imports=False)
        num_lane_pls = np.load(num_lane_pls_path, fix_imports=False)
        return input_pls, cls_labels, gt_traj, num_lane_pls

    def collate_fn(self, samples):
        input_pls = []
        labels = []
        max_num_pls = samples[0][0].shape[0]
        for sample in samples:
            pl, label = sample
            if pl.shape[0] > max_num_pls: max_num_pls = pl.shape[0]
            input_pls.append(torch.tensor(pl))  # as_tensor()?
            labels.append(torch.tensor(label))

        input_pls = [torch.cat([torch.zeros((max_num_pls - pl.shape[0], pl.shape[1], pl.shape[2])), pl], dim=0) for pl in input_pls]
        input_pls = torch.stack(input_pls, dim=0)
        labels = torch.stack(labels, dim=0)
        return input_pls, labels

    def load_example(self, example):
        scen_path = self.scenario_path(self.dataset_path, example)
        scenario = scenario_utils.load_argoverse_scenario_parquet(scen_path)
        tracks = scenario_utils._convert_tracks_to_tabular_format(scenario.tracks)
        
        map_loc = self.map_path(self.dataset_path, example)
        avm = ArgoverseStaticMap.from_json(map_loc)
        return avm, tracks

    def process_item(self, idx):
        example = self.scenarios[idx].decode('UTF-8').strip()
        
        avm, tracks = self.load_example(example)
        lanes_processor = data_utils.Lanes(avm, self.pts_per_pl)
        lane_polylines = lanes_processor.get_lane_polylines()

        ped_crossings = data_utils.get_ped_crossings(avm, self.pts_per_pl)

        tracks_processor = data_utils.Tracks(tracks, self.pts_per_pl, self.timesteps_history, self.timesteps_future)
    
        try:
            track_pls, ego_track, ego_loc = tracks_processor.get_track_pls()
        except ValueError:
            print(idx)
            return

        data_utils.normalize_coords(lane_polylines[:, :, :2], ped_crossings, track_pls, ego_loc)

        input_pls = data_utils.join_pls(lane_polylines, ped_crossings, track_pls, self.pts_per_pl)
        np.nan_to_num(input_pls, copy=False)
        
        label = data_utils.get_label(ego_track, self.timesteps_history) if self.split != "test" else np.empty((0, 0), dtype=np.float32)
        np.nan_to_num(label, copy=False)

        pth = str(self.dataset_path / example / "vectornet_")
        np.save(pth + "input.npy", input_pls, fix_imports=False)
        np.save(pth + "label.npy", label, fix_imports=False)
        # return input_pls, label  # collate_fn will transform to tensor

    def process_for_parallel(self, idx):
        example = self.scenarios[idx].decode('UTF-8').strip()
        
        avm, tracks = self.load_example(example)
        lanes_processor = data_utils.Lanes(avm, self.pts_per_pl)
        lane_polylines = lanes_processor.get_lane_polylines()

        ped_crossings = data_utils.get_ped_crossings(avm, self.pts_per_pl)

        tracks_processor = data_utils.Tracks(tracks, self.pts_per_pl, self.timesteps_history, self.timesteps_future)
    
        try:
            track_pls, ego_tracks = tracks_processor.get_tracks_for_parallel()
        except ValueError:
            print(idx)
            return

        data_utils.normalize_by_min(lane_polylines[:, :, :2], ped_crossings, track_pls)

        input_pls = data_utils.join_pls(lane_polylines, ped_crossings, track_pls, self.pts_per_pl)
        np.nan_to_num(input_pls, copy=False)
        
        labels = [data_utils.get_label(ego_track[0], self.timesteps_history) if self.split != "test" else np.empty((0, 0), dtype=np.float32) for ego_track in ego_tracks]
        labels = np.stack(labels, axis=0)
        np.nan_to_num(labels, copy=False)

        # pth = str(self.dataset_path / example / "parallel_vectornet_")
        # np.save(pth + "input.npy", input_pls, fix_imports=False)
        # np.save(pth + "label.npy", labels, fix_imports=False)
        return input_pls, labels  # collate_fn will transform to tensor

        # FOR VALIDATE FUNCTION
        # locs = [track[1] for track in ego_tracks]
        # headings = [track[0][-1, 2] for track in ego_tracks]
        # return input_pls, labels, locs, headings

    def process_parallel_frenet(self, idx):
        example = self.scenarios[idx].decode('UTF-8').strip()
        
        avm, tracks = self.load_example(example)
        lanes_processor = data_utils.Lanes(avm, self.pts_per_pl)
        lane_polylines = lanes_processor.get_lane_polylines()

        ped_crossings = data_utils.get_ped_crossings(avm, self.pts_per_pl)

        tracks_processor = data_utils.Tracks(tracks, self.pts_per_pl, self.timesteps_history, self.timesteps_future)
    
        try:
            track_pls, ego_tracks = tracks_processor.get_tracks_for_parallel()
        except ValueError:
            print(idx)
            return

        mins = data_utils.normalize_by_min(lane_polylines[:, :, :2], ped_crossings, track_pls)
        coords = np.stack([ego_track[0] for ego_track in ego_tracks], axis=0)
        coords[:, :, :2] -= mins

        input_pls = data_utils.join_pls(ped_crossings, lane_polylines, track_pls, self.pts_per_pl)
        np.nan_to_num(input_pls, copy=False)

        # all coordinates are normalized by min
        trajectories = coords[:, -(self.timesteps_future + 1):, :]

        final_locs = coords[:, -1, :]
        cls_labels = data_utils.get_cls_labels(lane_polylines[:, :, :2], final_locs)

        # RETURN NUMBER OF LANE POLYLINES, CLASSIFICATION AND REGRESSION WILL NOT BE DONE ON CROSSWALKS
        # REGRESSION LOSSES ONLY CALCULATED ON NODE CLOSEST TO FINAL POINT (INDEX IN CLS_LABEL)
        # RETURN: INPUT PLS, CLS_LABELS, GT TRAJECTORIES, NUMBER OF LANE PLS

        cls_labels = cls_labels.astype(np.float32)
        trajectories = trajectories.astype(np.float32)
        # pth = str(self.dataset_path / example / "pcurvenet_")
        # np.save(pth + "input.npy", input_pls, fix_imports=False)
        # np.save(pth + "cls_label.npy", cls_labels, fix_imports=False)
        # np.save(pth + "gt_trajectories.npy", trajectories, fix_imports=False)
        # np.save(pth + "num_lane_pls.npy", np.array([lane_polylines.shape[0]]), fix_imports=False)
        return input_pls, cls_labels, trajectories, np.array([lane_polylines.shape[0]]), mins
        

    def process_all_items(self, idx):  # same as process_item, except for all "high quality" tracks in the scene instead of just the ego track
        example = self.scenarios[idx].decode('UTF-8').strip()
        
        avm, tracks = self.load_example(example)
        lanes_processor = data_utils.Lanes(avm, self.pts_per_pl)
        lane_polylines = lanes_processor.get_lane_polylines()

        ped_crossings = data_utils.get_ped_crossings(avm, self.pts_per_pl)

        tracks_processor = data_utils.Tracks(tracks, self.pts_per_pl, self.timesteps_history, self.timesteps_future)

        try:
            track_pls, ego_tracks = tracks_processor.get_all_track_pls()
        except ValueError:
            return
        
        inputs = []
        locs = [track[1] for track in ego_tracks]
        headings = [track[0][-1, 2] for track in ego_tracks]
        for idx in range(len(ego_tracks)):
            lane_polylines_cp = lane_polylines.copy(order='K')
            ped_crossings_cp = ped_crossings.copy(order='K')

            data_utils.normalize_coords(lane_polylines_cp[:, :, :2], ped_crossings_cp, track_pls[idx], ego_loc=ego_tracks[idx][1])

            input_pls = data_utils.join_pls(lane_polylines_cp, ped_crossings_cp, track_pls[idx], self.pts_per_pl)
            np.nan_to_num(input_pls, copy=False)
            
            label = data_utils.get_label(ego_tracks[idx][0], self.timesteps_history) if self.split != "test" else np.empty((0, 0), dtype=np.float32)
            np.nan_to_num(label, copy=False)

            inputs.append((input_pls, label))

        return inputs, locs, headings # inputs - form for collate fn, but corresponding to all inputs for a single scene

    def extra(self):
        prev = "-"
        for idx, example in enumerate(self.scenarios):
            example = example.decode('UTF-8').strip()
            pth = self.dataset_path / example / "pcurvenet_input.npy"
            if not pth.exists():
                print(idx)
                # print(prev)
                # print(pth)
            else:
                input_pls = np.load(pth, fix_imports=False)
                if input_pls.shape[-1] != 21:
                    print(idx)
            prev = pth


DATAROOT = Path("/home/further/argoverse")


# x = AV2(root=DATAROOT, split='train')
# y = AV2(root=DATAROOT, split='val')
# # print(len(x))  # 199,908


# with mp.Pool() as pool:
#     pool.map(x.process_parallel_frenet, range(len(x)))
#     pool.map(y.process_parallel_frenet, range(len(y)))
#     pool.close()
#     pool.join()

# turn_examples = []
# for idx in range(len(x)):
#     input_pls, cls_labels, gt_trajectories, num_lanes = x[idx]
#     gt_trajectories = torch.from_numpy(gt_trajectories)
#     if torch.any(data_utils.angle_between(gt_trajectories[:, 0, 2], gt_trajectories[:, -1, 2]) > (np.pi / 3)):
#         turn_examples.append(idx)
# weights = torch.ones(len(x), dtype=torch.float32)
# weights[turn_examples] = (len(x) - len(turn_examples)) / len(turn_examples)
# print('equal:', weights[turn_examples].sum(), len(x) - len(turn_examples))
# torch.save(weights, 'turn_example_weights.pt')

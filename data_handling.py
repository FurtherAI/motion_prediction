
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import math
import copy
import torch
from pcurvenet import PCurveNet, ForwardPass

import subgraph_net
import globgraph_net

import av2.datasets.motion_forecasting.scenario_serialization as scenario_utils
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectState, ObjectType, Track, TrackCategory  # objtype - vehicle, motorcyclist
from av2.map.lane_segment import LaneSegment, LaneMarkType
from av2.map.map_primitives import Polyline, Point
from av2.utils.typing import NDArrayFloat
import av2.geometry.polyline_utils as polyline_utils
from av2.map.map_api import ArgoverseStaticMap, compute_data_bounds
import av2.geometry.interpolate as interp_utils

import av2.datasets.motion_forecasting.viz.scenario_visualization as map_viz

from dataset import AV2
from vectornet import VectorNet
from pcurvenet import compute_ade
import data_utils

DATAROOT = Path("/home/further/argoverse/test")
SCENARIO = "0000b329-f890-4c2b-93f2-7e2413d4ca5b" # "0000b0f9-99f9-4a1f-a231-5be9e4c523f7"

# ArgoverseStaticMap - from_map_dir
    # segments - left, right, successor, nearby, all, centerline, bool in_intersection, ped crossings
    # rasterize
    # driveable area
# scenario_utils - load_argoverse_scenario_parquet, _convert_tracks_to_tabular_format
# polyline_utils - interp_polyline_by_fixed_waypt_interval

def map_path(root : Path, folder : str) -> Path:
    return root / folder / ("log_map_archive_" + folder + ".json")

def scenario_path(root : Path, folder : str) -> Path:
    return root / folder / ("scenario_" + folder + ".parquet")

def get_predecessors(segments, segment_ids):
    unique_segments = set()
    for segment_id in segment_ids:
        for id in segments[segment_id].predecessors:
            try:
                if len(segments[id].predecessors) == 0:
                    unique_segments.add(segment_id)
            except KeyError:
                unique_segments.add(segment_id)
    return list(unique_segments)


def get_successors(segments, id):
    try:
        if len(segments[id].successors) == 0:
            return [id]
        # if (len(segments[id].successors) > 1):
        return [id] + get_successors(segments, segments[id].successors[0])
    except KeyError:
        # print("Key error")
        return []

def get_multi_successors2(segments, id, cur, base):
    try:
        successors = segments[id].successors
        cur.append(id)
        while len(successors) != 0:
            for suc in successors[1:]:
                try:
                    segments[suc]
                    get_multi_successors2(segments, suc, copy.deepcopy(cur), base)
                except KeyError:
                    pass
            
            id = successors[0]
            successors = segments[id].successors
            cur.append(id)
        base.append(cur)
    except KeyError:
        base.append(cur)

def get_multi_successors(segments, id : int, cur : list, base : list):
    try:
        successors = segments[id].successors
        while len(successors) != 0:
            cur.append(id)
            for suc in successors[1:]:
                try:
                    segments[suc]
                    get_multi_successors(segments, suc, copy.deepcopy(cur), base)
                except KeyError:
                    pass
            
            id = successors[0]
            successors = segments[id].successors
        base.append(cur)
    except KeyError:
        base.append(cur)

def filter_polylines(polylines):
    polylines = [set(pl) for pl in polylines]
    polylines.sort(key=lambda x : len(x))  # larger sets last, so don't need to check for subset against lower index
    filtered_polylines = []
    for idx, a in enumerate(polylines):
        subset = False
        for b in polylines[idx + 1:]:
            if a.issubset(b) and a != b:
                subset = True
                break
        if not subset:
            filtered_polylines.append(a)
    return filtered_polylines

def scenario_stuff():
    scen_path = scenario_path(DATAROOT, SCENARIO)
    scenario : ArgoverseScenario = scenario_utils.load_argoverse_scenario_parquet(scen_path)
    # print('scenario_id: ', scenario.scenario_id)
    # print('timestamps_ns: ', scenario.timestamps_ns)
    # print('focal_track_id: ', scenario.focal_track_id)
    # print('city_name: ', scenario.city_name)
    # print('map_id: ', scenario.map_id)
    # print('slice_id: ', scenario.slice_id)
    # print('----------------------------------')
    # print('----------------------------------')
    # print('num of tracks: ', len(scenario.tracks))
    # track = scenario.tracks[0]
    # print('object type: ', track.object_type)
    # print('category: ', track.category)
    # print(track.object_states)
    tracks = scenario_utils._convert_tracks_to_tabular_format(scenario.tracks)
    # tracks = tracks[tracks['object_type'] == 'vehicle']
    return tracks

def visualize_lanes(avm, polylines, tracks, ped=[]):
    pls = []
    for pl in polylines:
        left = []
        right = []
        for id in pl:
            left.append(avm.vector_lane_segments[id].left_lane_boundary.xyz[:, :2])
            right.append(avm.vector_lane_segments[id].right_lane_boundary.xyz[:, :2])
        pls.append([np.vstack(left), np.vstack(right)])

    pls += ped
    
    all_coords = np.vstack([np.vstack([pl[0] for pl in pls]), np.vstack([pl[1] for pl in pls])])
    xmin, xmax, ymin, ymax = np.min(all_coords[:, 0]), np.max(all_coords[:, 0]), np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
    mins = np.array([xmin, ymin])
    for pl in pls:
        pl[0] -= mins
        pl[1] -= mins
    
    # _, ax = plt.subplots()
    for pl in pls:
        map_viz._plot_polylines(pl, color='grey')

    ego_track = tracks[tracks["object_category"] == 3]
    cur_location = np.array([ego_track["position_x"].iloc[-1], ego_track["position_y"].iloc[-1]]) - mins
    heading = ego_track["heading"].iloc[-1]
    
    (bbox_length, bbox_width) = 4.0, 2.0

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y), bbox_length, bbox_width, np.degrees(heading), zorder=100
    )
    plt.gca().add_patch(vehicle_bounding_box)

    plt.xlim(0, xmax - xmin)
    plt.ylim(0, ymax - ymin)
    plt.gca().set_aspect("equal", adjustable="box")

    # Minimize plot margins and make axes invisible
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("viz.png", dpi=1000)

def map_stuff():
    map_loc = map_path(DATAROOT, SCENARIO)
    hd_map : ArgoverseStaticMap = ArgoverseStaticMap.from_json(map_loc)
    segment = hd_map.get_scenario_lane_segment_ids()

    ## GET FULL LANES INCLUDING MULTIPLE SUCCESSORS, FILTER OUT SUBSETS ONLY, MAY HAVE OVERLAPPING LANES WHEN MERGING/SPLITTING
    # pls = []
    # for ls in segment:
    #     get_multi_successors(hd_map.vector_lane_segments, ls, [], pls)

    ## GET FULL LANES FOR THE MAP (CURRENTLY NOT INCLUDING MULTIPLE SUCCESSORS) BY FILTERING
    # pls = [get_successors(hd_map.vector_lane_segments, segment[i]) for i in range(len(segment))]

    # print('before filtering: ', len(pls))
    # pls_map = {frozenset(pl) : pl for pl in pls}  # frozensets are immutable and hashable and can therefore be used as keys, allows recovery of order after set conversion
    # filter_ = filter_polylines(pls)
    # print('after filtering: ', len(filter_), sum([len(pl) for pl in filter_]))
    # pls = [pls_map[frozenset(pl)] for pl in filter_]

    pls2 = []
    pls3 = []
    unique = get_predecessors(hd_map.vector_lane_segments, segment)
    for id in unique:
        get_multi_successors(hd_map.vector_lane_segments, id, [], pls2)
        get_multi_successors2(hd_map.vector_lane_segments, id, [], pls3)

    print(len(pls2), len(pls3))
    for pl in pls3:
        if pl not in pls2:
            print("pl not in original:", pl)

    ped_crossings = hd_map.get_scenario_ped_crossings()
    print(len(ped_crossings))
    # ped_crossings = [(interp_utils.compute_midpoint_line(*crossing.get_edges_2d(), num_interp_pts=10))[0] for crossing in ped_crossings]
    ped_crossings = [list(crossing.get_edges_2d()) for crossing in ped_crossings]

    # full_list = []
    # for pl in pls:
    #     for id in pl:
    #         full_list.append(id)
    # for id in segment:
    #     if id not in full_list:
    #         print(f"id {id} missing")
    # print("done")

    # for pl in pls:
    #     print('---------------------')
    #     print(pl)
    #     print('---------------------')

    # lrpls = []
    # for pl in pls:
    #     left = []
    #     right = []
    #     for id in pl:
    #         left.append(hd_map.vector_lane_segments[id].left_lane_boundary.xyz[:, :2])
    #         right.append(hd_map.vector_lane_segments[id].right_lane_boundary.xyz[:, :2])
    #     lrpls.append([np.vstack(left) - mins, np.vstack(right) - mins])
    
    # breaks = []
    # for pl in lrpls:
    #     left = pl[0]
    #     right = pl[1]
    #     left = np.linalg.norm(np.diff(left, axis=0), axis=1)
    #     right = np.linalg.norm(np.diff(right, axis=0), axis=1)
    #     print('---------------------------')
    #     print(np.average(left), np.median(left), np.std(left), np.max(left))
    #     print(np.average(right), np.median(right), np.std(right), np.max(right))
    #     print('---------------------------')

    # visualize_lanes(hd_map, pls2[:10], scenario_stuff(), ped=ped_crossings)

def visualize_scenario():
    map_loc = map_path(DATAROOT, SCENARIO)
    hd_map : ArgoverseStaticMap = ArgoverseStaticMap.from_json(map_loc)
    scen_path = scenario_path(DATAROOT, SCENARIO)
    scenario : ArgoverseScenario = scenario_utils.load_argoverse_scenario_parquet(scen_path)
    map_viz.visualize_scenario(scenario, hd_map, Path("vis"))

def test_network():
    map_loc = map_path(DATAROOT, SCENARIO)
    avm : ArgoverseStaticMap = ArgoverseStaticMap.from_json(map_loc)
    segment : LaneSegment = avm.get_scenario_lane_segment_ids()

    pls = []
    for ls in segment:
        get_multi_successors(avm.vector_lane_segments, ls, [], pls)

    pls_map = {frozenset(pl) : pl for pl in pls}
    filter_ = filter_polylines(pls)
    pls = [pls_map[frozenset(pl)] for pl in filter_]

    pls = [interp_utils.interp_arc(20, np.vstack([avm.get_lane_segment_centerline(id)[:, :2] for id in pl])) for pl in pls]
    input_pls = torch.Tensor(np.array(pls))
    input_pls = input_pls.unsqueeze(0)

    print(input_pls.shape)
    assert(input_pls.shape == torch.Size((1, len(pls), 20, 2)))

    subgraph = subgraph_net.SubGraphNet(init_features=2, hidden=64)
    globgraph = globgraph_net.GlobalGraphNet()

    subgraph.cuda()
    globgraph.cuda()
    input_pls = input_pls.cuda()
    print(input_pls.device)

    with torch.no_grad():
        out = subgraph(input_pls).unsqueeze(0)
        out = globgraph(out[:, 0, :].unsqueeze(1), out)
        out.cpu()
    
    print(out.shape)


def rollout_trajectories(trajectories):
    # shape - (batch_size, 30, 2)
    return trajectories.cumsum(dim=1)


def plot_vehicles(tracks):
    objects = tracks[(tracks["timestep"] == 49) & (tracks["object_type"].isin(["vehicle", "bus", "cyclist", "motorcyclist"]))]
    cyclist_length, cyclist_width = 2.0, 0.7
    for idx, obj in objects.iterrows():
        bbox_length, bbox_width = 4.0, 2.0
        color = "#D3E8EF"
        if obj["object_type"] in ["cyclist", "motorcyclist"]: 
            bbox_length, bbox_width = cyclist_length, cyclist_width
        elif obj["object_type"] == "vehicle" and obj["object_category"] == 3:
            color = "#ECA25B"

        # Compute coordinate for pivot point of bounding box
        d = np.hypot(bbox_length, bbox_width)
        theta_2 = math.atan2(bbox_width, bbox_length)

        loc = tuple(obj[["position_x", "position_y"]])
        heading = float(obj["heading"])
        pivot_x = loc[0] - (d / 2) * math.cos(heading + theta_2)
        pivot_y = loc[1] - (d / 2) * math.sin(heading + theta_2)

        vehicle_bounding_box = Rectangle(
            (pivot_x, pivot_y), bbox_length, bbox_width, angle=np.degrees(heading), color=color, # zorder=100
        )
        plt.gca().add_patch(vehicle_bounding_box)


def demo(index):
    data = AV2("/home/further/argoverse", 'val', pts_per_pl=64, sec_history=2, sec_future=3)
    example = data.scenarios[index].decode('UTF-8').strip()

    map_loc = map_path(data.dataset_path, example)
    avm = ArgoverseStaticMap.from_json(map_loc)

    scen_path = scenario_path(data.dataset_path, example)
    scenario = scenario_utils.load_argoverse_scenario_parquet(scen_path)

    xmin, ymin, xmax, ymax = compute_data_bounds(avm.get_scenario_vector_drivable_areas())
    map_viz._plot_static_map_elements(avm)

    df_tracks = scenario_utils._convert_tracks_to_tabular_format(scenario.tracks)
    plot_vehicles(df_tracks)

    vectornet = VectorNet(init_features=18)
    vectornet = vectornet.load_from_checkpoint("parallel_checkpoints/epoch=1-step=80000.ckpt", hparams_file="lightning_logs/version_15/hparams.yaml")
    vectornet.eval()

    # inputs, locs, headings = data.process_all_items(index)
    # input_pls, labels = data.collate_fn(inputs)

    # with torch.inference_mode():
    #     predictions = vectornet(input_pls).view(input_pls.shape[0], -1, 2)

    #     for idx, heading in enumerate(headings):
    #         data_utils.rotate(labels[idx, :, :].numpy(), (math.pi / 2) - heading)
    #         data_utils.rotate(predictions[idx, :, :].numpy(), (math.pi / 2) - heading)

    #     predictions = rollout_trajectories(predictions)
    #     labels = rollout_trajectories(labels)

    #     locs = [torch.tensor(loc) for loc in locs]
    #     locs = torch.stack(locs, dim=0).unsqueeze(1)
    #     predictions += locs
    #     labels += locs


    input_pls, labels, locs, headings = data.process_for_parallel(index)

    with torch.inference_mode():
        input_pls = torch.tensor(input_pls)
        labels = torch.tensor(labels)
        input_pls = input_pls.unsqueeze(0)
        predictions = vectornet(input_pls, num_agents=labels.shape[0]).view(labels.shape[0], -1, 2)

        for idx, heading in enumerate(headings):
            data_utils.rotate(labels[idx, :, :].numpy(), (math.pi / 2) - heading)
            data_utils.rotate(predictions[idx, :, :].numpy(), (math.pi / 2) - heading)

        predictions = rollout_trajectories(predictions)
        labels = rollout_trajectories(labels)

        locs = [torch.tensor(loc) for loc in locs]
        locs = torch.stack(locs, dim=0).unsqueeze(1)
        predictions += locs
        labels += locs

    map_viz._plot_polylines(predictions.numpy(), color='r', alpha=.4)
    map_viz._plot_polylines(labels.numpy(), color='g')
    # plot static map elements, plot tracks, retrieve predicted and gt labels and plot those polylines (map_viz._plot_polylines)
    # Minimize plot margins and make axes invisible
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f"parallel_visualizations/viz{index}.png", dpi=1000)
    plt.cla()

def pcurve(index):
    data = AV2("/home/further/argoverse", 'val', pts_per_pl=64, sec_history=2, sec_future=3)
    example = data.scenarios[index].decode('UTF-8').strip()

    map_loc = map_path(data.dataset_path, example)
    avm = ArgoverseStaticMap.from_json(map_loc)

    scen_path = scenario_path(data.dataset_path, example)
    scenario = scenario_utils.load_argoverse_scenario_parquet(scen_path)

    map_viz._plot_static_map_elements(avm)

    df_tracks = scenario_utils._convert_tracks_to_tabular_format(scenario.tracks)
    plot_vehicles(df_tracks)

    pcurvenet = PCurveNet(init_features=21)
    pcurvenet = pcurvenet.load_from_checkpoint("pcurve_checkpoints/v0/last.ckpt")
    pcurvenet.eval()

    with torch.inference_mode():
        forward = ForwardPass(data, index, 6)
        pts = forward.forward(pcurvenet)
        forward.gt_trajectories.squeeze_(0)
        forward.gt_trajectories[:, :, :2] += forward.mins
    
    pts = pts.view(forward.num_agents, forward.topn, 30, 2)
    # pts = pts.take_along_dim(compute_ade(pts, gt_trajectories[:, None, 1:, :2]).mean(dim=2).argmin(dim=1).view(num_agents, 1, 1, 1), dim=1).squeeze(1)
    pts += forward.mins

    agt = 2
    for agent in forward.gt_trajectories[:, 1:, :]:
        plt.plot(agent[:, 0], agent[:, 1], color='g')

    # for agent in (forward.get_pts(forward.traj_labels) + forward.mins):
    #     plt.plot(agent[:, 0], agent[:, 1], color='b', alpha=.4)
    
    # for agent in pts:
    #     plt.plot(agent[:, 0], agent[:, 1], color='r', alpha=.4)
    for agent in pts[:]:
        for track in agent[:]:
            plt.plot(track[:, 0], track[:, 1], color='r', alpha=.4)

    # t = torch.tensor(
        # [[105.1396, 102.3574],
        # [109.5013, 107.6017],
        # [109.1842, 107.3572],]
        # [[ 72.6533,  98.9052],
        # [ 70.2012, 103.5829],
        # [ 80.9205, 126.6962],
        # [ 80.5712, 127.6171],]
        # [102.0608, 104.7535],
        # [109.0558, 113.4030],
        # [103.2506, 107.8504],
        # [111.2960,  97.5636],
        # [115.4002, 101.1752],
        # [114.5418,  98.5823]]

    # )
    # for pt in (t + mins):
    #     plt.plot(pt[0], pt[1], marker='o', markersize=1, color='b')

    # probs = torch.load(f'prob_maps/prob_map{agt}.pt', map_location='cpu')
    # probs[:, :2] += mins
    # probs[:, 2] = torch.nn.functional.softmax(probs[:, 2], dim=0)
    # probs[:, 2] = (probs[:, 2] - probs[:, 2].min()) / (probs[:, 2].max() - probs[:, 2].min())
    # for pt in probs:
    #     plt.plot(pt[0], pt[1], marker='s', markersize=.4, color='r', alpha=pt[2].item(), zorder=1000)

    # map_viz._plot_polylines(predictions.numpy(), color='r', alpha=.4)
    # map_viz._plot_polylines(labels.numpy(), color='g')
    # plot static map elements, plot tracks, retrieve predicted and gt labels and plot those polylines (map_viz._plot_polylines)
    # Minimize plot margins and make axes invisible
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.savefig(f"prob_maps/viz{index}-{agt}.png", dpi=1000)
    plt.savefig(f"test_viz/viz{index}.png", dpi=1000)
    plt.cla()

def main():
    # for i in [1503, 2314, 6762, 15934, 18635, 23569]:
        # pcurve(16634)
    for i in [i for i in torch.randint(0, 24988, size=(10,))]:
        pcurve(i)
    # tensor([  370,  3436, 14593,  2841, 17843, 21767,  3397, 24412, 16184, 14554])
    # tensor([False,  True,  True, False,  True,  True,  True, False, False, False])


if __name__ == "__main__":
    main()

import torch
import numpy as np
import data_utils
from time import perf_counter
from pcurvenet import PCurveNet, compute_ade, ForwardPass
from dataset import AV2
import matplotlib.path as matPath
import matplotlib.pyplot as plt


def adevt_minade(de, horizons):
    ades = torch.zeros_like(horizons)

    min_traj = de[:, :, -1].argmin(dim=-1).unsqueeze(1)  # minimum trajectory out of topn (6) based on FDE
    min_traj_ade = de.mean(dim=-1).argmin(dim=-1).unsqueeze(1)
    for t in range(len(horizons)):
        # ades[t] = de[:, :, :(t + 1)].mean(dim=2).min(dim=-1)[0].mean()
        ades[t] = de[:, :, :(t + 1)].mean(dim=2).take_along_dim(min_traj_ade, dim=-1).mean()
        # ades[t] = de[:, :, :(t + 1)].mean(dim=2).take_along_dim(min_traj, dim=-1).mean()


    print('ADE:', de.mean(dim=2).min(dim=-1)[0].mean())
    print('ADE:', de.mean(dim=2).take_along_dim(min_traj, dim=-1).mean())
    return ades

def adevt_minfde(de, horizons):
    ades = torch.zeros_like(horizons)
    min_traj = de[:, :, -1].argmin(dim=-1).unsqueeze(1)  # minimum trajectory out of topn (6) based on FDE
    for t in range(len(horizons)):
        # ades[t] = de[:, :, :(t + 1)].mean(dim=2).min(dim=-1)[0].mean()
        ades[t] = de[:, :, :(t + 1)].mean(dim=2).take_along_dim(min_traj, dim=-1).mean()
    return ades

def adevt_min(de, horizons):
    ades = torch.zeros_like(horizons)
    for t in range(len(horizons)):
        ades[t] = de[:, :, :(t + 1)].mean(dim=2).min(dim=-1)[0].mean()
    return ades

def adevt_k1(de, horizons):
    ades = torch.zeros_like(horizons)
    for t in range(len(horizons)):
        ades[t] = de[:, :(t + 1)].mean()
    return ades

def fdevt_minfde(de, horizons):
    fdes = torch.zeros_like(horizons)
    min_traj = de[:, :, -1].argmin(dim=-1).unsqueeze(1)  # minimum trajectory out of topn (6) based on FDE
    for t in range(len(horizons)):  # FDE based on trajectory with min FDE at full timesteps
        fdes[t] = de[:, :, t].take_along_dim(min_traj, dim=1).mean()
    return fdes

def fdevt_minade(de, horizons):
    fdes = torch.zeros_like(horizons)
    min_traj_ade = de.mean(dim=-1).argmin(dim=-1).unsqueeze(1)
    for t in range(len(horizons)):
        fdes[t] = de[:, :, t].take_along_dim(min_traj_ade, dim=1).mean()
    return fdes

def fdevt_min(de, horizons):
    fdes = torch.zeros_like(horizons)
    for t in range(len(horizons)):  # closest pt at each timestep
        fdes[t] = de[:, :, t].min(dim=-1)[0].mean()
    return fdes

def fdevt_k1(de, horizons):
    fdes = torch.zeros_like(horizons)
    for t in range(len(horizons)):  # closest pt at each timestep
        fdes[t] = de[:, t].mean()
    return fdes

def mr(de):
    miss = de[:, :, -1] > 2.0  # threshold for miss is 2m for Argoverse leaderboard
    miss = miss.sum(dim=1) == de.shape[1]  # all trajectories miss
    # print(miss)
    return miss.sum() / miss.numel()

def pcurve_iar():
    validation_data = AV2("/home/further/argoverse", 'val', pts_per_pl=64, sec_history=2, sec_future=3)

    data_samples = torch.randint(0, 24988, size=(10,))
    # data_samples = [i for i in range(1000, 2000)]
    # inputs = [validation_data[i] for i in data_samples]

    pcurvenet = PCurveNet(init_features=21)
    pcurvenet = pcurvenet.load_from_checkpoint("pcurve_checkpoints/last.ckpt")
    pcurvenet.eval()

    de = []
    ias = 0
    total_agents = 0
    topn = 6
    opp = 0
    off = 0
    uniq_lanes_per_agent = 0
    min_dist_to_lane = 0
    illegal_traj_threshold = np.pi / 2
    for idx in data_samples:
        forward = ForwardPass(validation_data, idx, topn=topn)
        total_agents += forward.num_agents
        # total_agents += 1

        with torch.inference_mode():
            pts = forward.forward(pcurvenet)
            lane_coords, clane_angles = forward.lane_coords, forward.clane_angles
            # how close to lane centerline
            end_to_lane = torch.linalg.norm(pts.view(forward.num_agents, topn, 30, 1, 2)[:, :, -1] - lane_coords.view(1, 1, -1, 2), dim=-1)
            end_to_lane = end_to_lane.min(dim=-1)[0]
            min_dist_to_lane += end_to_lane.sum()

            # pts = pts.view(num_agents, topn, 30, 2)
            # shape - (num_agents, topn, 30, total lane pts)
            das = list(forward.avm.vector_drivable_areas.values())  # list[DrivableArea]
            das = [matPath.Path(da.xyz[:, :2]) for da in das]  # list[path]
            trajs = pts.view(forward.num_agents * topn * 30, 2)
            trajs += forward.mins  # coords were normalized by mins, reset to map frame
            in_map = []
            init_in_map = []
            for da in das:
                pt_inside = da.contains_points(trajs)
                in_map.append(torch.tensor(pt_inside).reshape(forward.num_agents, topn, 30))
                pt_inside = da.contains_points(forward.initial_pt[:, 0, :2] + forward.mins)
                init_in_map.append(torch.tensor(pt_inside))
            in_map = torch.stack(in_map, dim=0)
            in_map = in_map.sum(dim=0).bool()  # logical or across all the drivable area polygons. For each traj pt, is it inside a drivable area
            in_map = in_map.sum(dim=-1) == 30
            init_in_map = torch.stack(init_in_map, dim=0)
            init_in_map = init_in_map.sum(dim=0).bool()  # require that the agent be initially within the drivable area for traj to be counted as off map (parking lots aren't illegal trajectories)
            off_map = ~in_map & init_in_map.unsqueeze(1)

            vec_to_pt = forward.final_pt[0, :, :, :2] - forward.initial_pt[:, :, :2] # gt_trajectories[0, :, None, 0, :2]  # difference between final predicted pt and last observed location
            # (num agents, topn, 2)
            angle_to_pt = torch.atan2(vec_to_pt[..., 1], vec_to_pt[..., 0])
            angle_to_pt[angle_to_pt < 0] += 2 * np.pi
            # final angle is selected as the most reasonable lane angle at the given final location (closest angle to initial heading)
            # MAKE OVER ALL PTS
            # ~node_in_opposing_traffic & dist_mask then or along all nodes
            traj_end_nodes = torch.linalg.norm(lane_coords.view(1, 1, -1, 2) - pts.view(forward.num_agents, topn, 30, 2)[:, :, -1, None], dim=-1)
            traj_end_nodes = traj_end_nodes < 2.5  # mask for nodes within approx half lane width of end of trajectory
            # (num_agents, topn, total pts)
            angle_to_node = lane_coords.view(1, -1, 2) - forward.gt_trajectories[0, :, 0:1, :2]
            angle_to_node = torch.atan2(angle_to_node[..., 1], angle_to_node[..., 0])
            angle_to_node[angle_to_node < 0] += 2 * np.pi
            node_in_opposing_traffic = data_utils.angle_between(angle_to_node, clane_angles.view(1, -1)) > illegal_traj_threshold  # (num_agents, total pts)
            node_in_opposing_traffic = node_in_opposing_traffic.view(forward.num_agents, 1, -1)# .expand(1, topn, 1)
            traj_in_same_traffic = (~node_in_opposing_traffic & traj_end_nodes).sum(dim=-1).bool()

            # if moving backward or sideways but not within threshold, could be flagged as opposing. Revert ones that end up in lanes with the same angle
            # especially common with stationary vehicles, predicted final location could be small distance behind
            traj_in_equivalent_lane = data_utils.angle_between(forward.initial_pt[:, :, 2], forward.final_angle[0, :, :]) < (np.pi / 6)
            traj_in_opposing_traffic = ~traj_in_same_traffic & ~traj_in_equivalent_lane & init_in_map.unsqueeze(1)
            # select ego
            # off_map = off_map[forward.ego_idx]
            # traj_in_opposing_traffic = traj_in_opposing_traffic[forward.ego_idx]

            # traj is illegal if it is either off map or in opposing traffic
            ias += torch.logical_or(off_map, traj_in_opposing_traffic).sum()
            opp += traj_in_opposing_traffic.sum()
            off += off_map.sum()
            de.append(compute_ade(pts.view(forward.num_agents, topn, 30, 2), forward.gt_trajectories[0, :, None, 1:, :2]))  # [forward.ego_idx], [0, forward.ego_idx, 1:, :2]
            print(idx, forward.num_agents)
            # if traj_in_opposing_traffic.sum() or off_map.sum():
            #     print(idx)

            # print('Combined:', torch.logical_or(off_map, traj_in_opposing_traffic).sum(dim=-1))
            # print('Opposing:', traj_in_opposing_traffic.sum(dim=-1))
            # print('Off Map :', off_map.sum(dim=-1))

            # print('Opposing')
            # print(traj_in_opposing_traffic.int())
            # print('Off Map')
            # print(off_map.int())

    print(ias / (total_agents * topn))
    print(opp / (total_agents * topn))
    print(off / (total_agents * topn))
    de = torch.cat(de, dim=0)
    print(mr(de))
    # print(min_dist_to_lane / (total_agents * topn))
    horizons = torch.linspace(0.1, 3.0, 30)
    adevt_minade(de, horizons)
    # print(uniq_lanes_per_agent / (total_agents))

def traj_spread(trajs):
    # shape - (num_agents, topn, 30, 2)
    closest_traj = trajs[:, :, None, -1, :] - trajs[:, None, :, -1, :]
    closest_traj = torch.linalg.norm(closest_traj, dim=-1)
    closest_traj[closest_traj == 0] = torch.inf
    closest_traj = closest_traj.min(dim=-1)[0].mean()
    return closest_traj

def pcurve_fps(data, idx, device, iters=10000):
    pcurvenet = PCurveNet(init_features=21)
    pcurvenet = pcurvenet.load_from_checkpoint("pcurve_checkpoints/last.ckpt")  # epoch=14-step=32760
    pcurvenet.eval()
    pcurvenet.to(device=device)

    forward = ForwardPass(data, idx, 6)
    print('num_agents:', forward.num_agents)
    with torch.inference_mode():
        start = perf_counter()
        for _ in range(iters):
            pts = forward.forward(pcurvenet, set_labels=False)
    
    end = perf_counter()
    print((end - start) / (iters * forward.num_agents))
    print(end - start)

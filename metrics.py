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
    miss = miss.sum(dim=1) == 6  # all trajectories miss
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

def pcurve_fps(inp, device, iters=10000):
    pcurvenet = PCurveNet(init_features=21)
    pcurvenet = pcurvenet.load_from_checkpoint("pcurve_checkpoints/last-v2.ckpt")  # epoch=14-step=32760
    pcurvenet.eval()
    pcurvenet.to(device=device)

    input_pls, cls_labels, gt_trajectories, num_lane_pls, mins, idx = inp
    input_pls = torch.as_tensor(input_pls, dtype=torch.float32).unsqueeze(0).to(device=device)

    cls_labels = torch.as_tensor(cls_labels, dtype=torch.float32).unsqueeze(0).to(device=device)
    gt_trajectories = torch.as_tensor(gt_trajectories, dtype=torch.float32).unsqueeze(0).to(device=device)
    num_lanes = num_lane_pls[0]
    num_agents = cls_labels.shape[1]
    print('num agents:', num_agents)


    with torch.inference_mode():
        start = perf_counter()
        for _ in range(iters):
            cls_out, traj_out = pcurvenet(input_pls, num_lanes, num_agents)
            closest_node = cls_out[..., 0].view(-1, num_agents, num_lanes * 64)
            # closest_node = torch.nn.functional.softmax(closest_node, dim=-1)
            # closest_node = closest_node.argmax(dim=-1)
            topn = 6
            vals, closest_node = closest_node.topk(topn, dim=-1)
            # closest_node = closest_node.view(1, -1)
            # closest_node = cls_labels[:, :, 0].long()

            closest_reg_out = cls_out[..., 1:].view(-1, num_agents, 1, num_lanes * 64, 3)
            closest_reg_out = closest_reg_out.take_along_dim(closest_node.view(1, num_agents, topn, 1, 1), dim=3).squeeze(3)

            closest_traj_out = traj_out.view(-1, num_agents, 1, num_lanes * 64, traj_out.shape[-1])
            closest_traj_out = closest_traj_out.take_along_dim(closest_node.view(1, num_agents, topn, 1, 1), dim=3).squeeze(3)
            # batch, num_agents, num_lanes, ptsperpl, 4
            # traj out = batch, num_agents, num_lanes, ptsperpl, 60

            lane_coords = input_pls[:, -(num_lanes + num_agents):-num_agents, :, 2:4]  # xy coordinates for each lane node
            v0 = torch.linalg.norm(input_pls[:, -num_agents:, -1, 13:15], dim=-1).squeeze(0)

            offset = 4
            lane_angles = lane_coords[:, :, offset:, :] - lane_coords[:, :, :-offset, :]
            lane_angles = torch.arctan2(lane_angles[..., 1], lane_angles[..., 0])
            lane_angles[lane_angles < 0] += 2 * np.pi  # shape (batch, num_lanes, 60)

            lane_coords = lane_coords.view(-1, num_lanes * 64, 2)
            batch_range = torch.arange(input_pls.shape[0])
            closest_node_coords = lane_coords[batch_range, closest_node]

            initial_pt = gt_trajectories[0, :, None, 0, :].repeat(1, topn, 1)

            closest5 = torch.linalg.norm(closest_node_coords.unsqueeze(3) - lane_coords.view(1, 1, 1, num_lanes * 64, 2), dim=-1)
            closest5, idx_closest5 = closest5.topk(5, largest=False, dim=-1)

            closest_node = data_utils.unravel_idx(idx_closest5, (num_lanes, 64))  # closest_node

            ## MATCH FINAL ANGLE TO LANE ANGLE POINTING IN THE SAME DIRECTION (WITHIN 60 DEGREES) TO PREVENT SELECTING OPPOSING TRAFFIC LANE WHEN OVERLAPPING
            final_angle = lane_angles[(batch_range, closest_node[0], closest_node[1].clamp(0, 64 - offset - 1))]
            initial_angle = initial_pt[:, :, 2]
            initial_angle[initial_angle < 0] += 2 * np.pi
            angle_between = torch.abs(final_angle - initial_angle.view(1, num_agents, topn, 1))
            angle_between = torch.minimum(angle_between, 2 * np.pi - angle_between)
            eps = 1e-5
            closest5 += eps
            closest5[angle_between <= (np.pi / 3)] = 0
            final_angle = final_angle.take_along_dim(closest5.argmin(dim=-1, keepdim=True), dim=-1).squeeze(-1)

            final_pt = torch.cat([closest_node_coords, final_angle.unsqueeze(3)], dim=3)
            final_pt += closest_reg_out  # += predicted residuals

            pts = torch.stack([initial_pt.reshape(-1, 3), final_pt.reshape(-1, 3)], dim=1)
            intersections = data_utils.batch_intersect(pts, heading_threshold=5)

            p0 = pts[:, 0, :2].unsqueeze(1)
            p1 = intersections.unsqueeze(1)
            p2 = pts[:, 1, :2].unsqueeze(1)
            # gt_trajectories contain last observed point (so skip this one) + headings for each point
            timesteps_future = 30
            ###############################################################################
            ###############################################################################
            s = torch.linspace(0, 1, steps=(128 + 1), device=device)  # could increase precision for inference
            s = s.unsqueeze(1)
            bezier_pts = ((1 - s)**2) * p0 + 2 * s * (1 - s) * p1 + (s**2) * (p2)  # (1-s)^2 p0 + 2s(1-s)p1 + s^2 p2 (shape = (batch, 101, 2))
            s = s.squeeze(1)
            diff = torch.diff(bezier_pts, dim=1)

            orthogonal_vectors = torch.flip(diff, dims=(2,))  # switch x and y coordinates and negate new y (gives orthogonal vector pointing right) (eg. (3, 1) -> (1, -3))
            orthogonal_vectors[:, :, 1] *= -1

            arclength = torch.linalg.norm(diff, dim=2)  # shape (batch, 100)
            orthogonal_vectors /= arclength.unsqueeze(2)
            cum_arclength = arclength.cumsum(dim=1)

            sec_future = timesteps_future / 10
            v0 = v0.repeat(topn)
            acc = (2 * (cum_arclength[:, -1] - (v0 * sec_future))) / (sec_future ** 2)  # assume and calculate constand acceleration along bezier arc
            timesteps = torch.linspace(0.1, sec_future, timesteps_future, device=device)  # curve is between last observed point and end point/first predicted point is at time 0.1
            timesteps = timesteps.unsqueeze(0)
            acc = acc.unsqueeze(1)
            v0 = v0.unsqueeze(1)
            s_t_const_acc = v0 * timesteps + .5 * acc * (timesteps ** 2)  # shape (batch, timesteps_future)

            closest_traj_out = closest_traj_out.view(-1, num_agents * topn, timesteps_future, 2)
            s_t_traj = s_t_const_acc + closest_traj_out[0, :, :, 0]
            stos = torch.abs(s_t_traj[:, :, None] - cum_arclength[:, None, :])
            indices = stos.argmin(dim=-1)
            s_norm = s[indices]

            s_norm.unsqueeze_(2)
            along_track = ((1 - s_norm)**2) * p0 + 2 * s_norm * (1 - s_norm) * p1 + (s_norm**2) * (p2)
            d_t_traj = closest_traj_out[0, :, :, 1].view(num_agents * topn, 30)
            reduced_orth_vectors = orthogonal_vectors.take_along_dim(indices.unsqueeze(2), dim=1)
            cross_track = d_t_traj.unsqueeze(2) * reduced_orth_vectors
            pts = along_track + cross_track
            ###############################################################################
            ###############################################################################
    
    end = perf_counter()
    print((end - start) / (iters * num_agents))
    print(end - start)

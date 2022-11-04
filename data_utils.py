import copy
import math
from random import sample
from turtle import heading
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt


from av2.map.lane_segment import LaneMarkType
import av2.geometry.interpolate as interp_utils


class Lanes():
    def __init__(self, avm, pts_per_pl):
        self.avm = avm
        self.segment_ids = avm.get_scenario_lane_segment_ids()
        self.lane_segments = avm.vector_lane_segments
        self.pts_per_pl = pts_per_pl

    def get_lane_polylines(self):
        pls = []
        unique = self.get_predecessors()
        for id in unique:
            self.get_multi_successors(id, [], pls)
        
        lane_mark_type = {mark : (i + 1) for i, mark in enumerate(LaneMarkType)}  # give enumerated type an integer value
        lane_polylines = []
        for pl in pls:
            left = []
            right = []
            extra_features = np.zeros((len(pl), 3), dtype=np.float32)
            for idx, id in enumerate(pl):
                ls = self.lane_segments[id]
                left.append(ls.left_lane_boundary.xyz[:, :2])
                right.append(ls.right_lane_boundary.xyz[:, :2])
                extra_features[idx] = np.array([ls.is_intersection, lane_mark_type[ls.left_mark_type], lane_mark_type[ls.right_mark_type]], dtype=np.float64)

            coords = interp_utils.compute_midpoint_line(np.vstack(left), np.vstack(right), num_interp_pts=self.pts_per_pl)[0]
            features = np.concatenate(
                [
                    coords, 
                    extra_features.repeat(math.ceil(self.pts_per_pl / len(pl)), axis=0)[:self.pts_per_pl],
                ], axis=1)
            lane_polylines.append(features)

        lane_polylines = np.stack(lane_polylines, axis=0)

        lane_angles = lane_polylines[:, 4:, :] - lane_polylines[:, :-4, :]
        lane_angles = np.arctan2(lane_angles[..., 1], lane_angles[..., 0])
        lane_angles[lane_angles < 0] += 2 * np.pi  # shape (batch, num_lanes, 60)
        clane_angles = np.concatenate([lane_angles, np.broadcast_to(lane_angles[:, -1:], (lane_angles.shape[0], 4))], axis=1)

        lane_curvature = clane_angles[:, 16:] - clane_angles[:, :-16]
        lane_curvature[lane_curvature < 0] += 2 * np.pi
        lane_curvature[lane_curvature > np.pi] = -(2 * np.pi - lane_curvature[lane_curvature > np.pi])  # right turn becomes a negative rotation (preserving information about which direction the lane curves)

        lane_curvature = np.concatenate([lane_curvature, np.broadcast_to(lane_curvature[:, -1:], (lane_curvature.shape[0], 16))], axis=1)
        lane_polylines = np.concatenate([lane_polylines[:, :, :2], np.expand_dims(clane_angles, 2), np.expand_dims(lane_curvature, 2), lane_polylines[:, :, 2:]], axis=2)

        return lane_polylines

    def get_predecessors(self):
        unique_segments = set()
        for segment_id in self.segment_ids:
            for id in self.lane_segments[segment_id].predecessors:
                try:
                    if len(self.lane_segments[id].predecessors) == 0:
                        unique_segments.add(segment_id)
                except KeyError:
                    unique_segments.add(segment_id)
        return list(unique_segments)

    def get_multi_successors(self, id, cur, base):
        try:
            successors = self.lane_segments[id].successors
            cur.append(id)
            while len(successors) != 0:
                for suc in successors[1:]:
                    try:
                        self.lane_segments[suc]
                        if suc not in cur:
                            self.get_multi_successors(suc, copy.deepcopy(cur), base)
                    except KeyError:
                        pass
                
                id = successors[0]
                if id in cur: break  # these checks prevent infinite loops (for some reason, some of the data has these (maybe traffic circles???))
                successors = self.lane_segments[id].successors
                cur.append(id)
            base.append(cur)
        except KeyError:
            base.append(cur)


def get_ped_crossings(avm, pts_per_pl):
    ped_crossings = avm.get_scenario_ped_crossings()
    ped_crossings = [(interp_utils.compute_midpoint_line(*crossing.get_edges_2d(), num_interp_pts=pts_per_pl))[0]
                    for crossing in ped_crossings]
    ped_crossings = np.stack(ped_crossings, axis=0) if len(ped_crossings) > 0 else np.empty((0, 0, 2))
    return ped_crossings


class Tracks():
    def __init__(self, tracks, pts_per_pl, timesteps_history, timesteps_future):
        self.tracks = tracks
        self.pts_per_pl = pts_per_pl
        self.timesteps_history = timesteps_history
        self.timesteps_future = timesteps_future

        self.start_timestep = 50 - timesteps_history
        self.end_timestep = 49 + timesteps_future

    def get_track_pls(self):
        tracks, ego_id = self.filter_tracks()

        ego_track, ego_loc = self.get_ego_track(tracks, ego_id)

        track_ids = tracks["track_id"].unique()
        track_pls = self.select_track_features(tracks, track_ids, ego_id)

        track_pls = self.interp_track_features(track_pls)
        return track_pls, ego_track, ego_loc

    def get_tracks_for_parallel(self):
        tracks, _ = self.filter_tracks()
        track_ids = tracks["track_id"].unique()

        # list[(ego_track, ego_loc)]
        ego_tracks = [self.get_ego_track(tracks, id) for id in track_ids]

        nearby_track_features = self.get_nearby_vehicle(tracks, track_ids, x_threshold=1.5)
        track_pls = self.select_track_features(tracks, track_ids, track_ids[-1])  # track_ids[-1] so it doesn't change the order

        track_pls = self.join_track_features(track_pls, nearby_track_features)

        track_pls = self.interp_for_parallel(track_pls)
        return track_pls, ego_tracks

    def get_all_track_pls(self):
        tracks, _ = self.filter_tracks()
        track_ids = tracks["track_id"].unique()

        # list[(ego_track, ego_loc)]
        ego_tracks = [self.get_ego_track(tracks, ego_id) for ego_id in track_ids]

        track_pls = [self.select_track_features(tracks, track_ids, ego_id) for ego_id in track_ids]

        track_pls = [self.interp_track_features(pls) for pls in track_pls]
        return track_pls, ego_tracks

    def filter_tracks(self):
        ego_track = self.tracks[self.tracks["object_category"] == 3]
        ego_id = ego_track["track_id"].iloc[0] 
        tracks = self.tracks[(
            self.tracks["object_type"].isin(["vehicle", "bus", "motorcyclist"])) & 
            (self.tracks["object_category"] >= 2) & 
            self.tracks["timestep"].between(self.start_timestep, self.end_timestep)
        ]
        if ego_track["object_type"].iloc[0] not in ["vehicle", "bus", "motorcyclist"]:
            ego_id = tracks["track_id"].sample(n=1).iloc[0]  # if ego is not a vehicle, choose a new ego

        pd.options.mode.chained_assignment = None
        tracks["object_type"] = tracks["object_type"].map({"vehicle" : 1, "motorcyclist" : 2, "bus" : 3})
        pd.reset_option("mode.chained_assignment")
        return tracks, ego_id

    def get_nearby_vehicle(self, tracks, track_ids, x_threshold=1.5):
        track_states = tracks[(tracks["track_id"].isin(track_ids)) & (tracks["timestep"] == 49)]
        coords = track_states[["position_x","position_y"]].to_numpy()
        headings = track_states["heading"].to_numpy()

        # each set of coordinates (dim 0 is the batch/set) normalized by its respective ego coordinate
        # eg. coords[0]: shape - (num_agents, 2), normalized by coordinate original coords[0, :]
        coords = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]
        coords = rotate_batch(coords, headings - (np.pi / 2))

        nearest_to_ego_heading = np.where(np.expand_dims((coords[:, :, 1] > 0) & (np.abs(coords[:, :, 0]) < x_threshold), axis=2), coords, np.finfo(np.float32).max)
        
        nearest_indeces = nearest_to_ego_heading[:, :, 1].argmin(axis=1)
        nearest_values = nearest_to_ego_heading[np.arange(0, nearest_to_ego_heading.shape[0]), nearest_indeces]
        mask = nearest_values[:, 0] != np.finfo(np.float32).max

        # select self id if masked, so that when features are calculated as relative values, they will be 0
        nearest_track_ids = [track_ids[track_idx] if msk else track_ids[idx] for idx, (track_idx, msk) in enumerate(zip(nearest_indeces, mask))]
        
        reduced_tracks = tracks[tracks["timestep"].between(self.start_timestep, 49)]
        nearby_track_features = [reduced_tracks[reduced_tracks["track_id"] == track_id][["position_x", "position_y", "velocity_x", "velocity_y"]].to_numpy() 
                                for track_id in nearest_track_ids]
        return nearby_track_features

    def select_track_features(self, tracks, track_ids, ego_id):
        track_ids[:-1] = track_ids[track_ids != ego_id]
        track_ids[-1] = ego_id  # place ego at last index, for finding it when predicting trajectories
        reduced_tracks = tracks[tracks["timestep"].between(self.start_timestep, 49)]
        track_pls = [reduced_tracks[reduced_tracks["track_id"] == track_id][["object_type", "timestep", "position_x", "position_y", "velocity_x", "velocity_y", "heading"]]
                    for track_id in track_ids]
        track_pls = [track_pl.to_numpy() for track_pl in track_pls]
        return track_pls

    def join_track_features(self, track_pls, nearby_track_features):
        track_pls = [np.concatenate([track_pl, nearby_track - track_pl[:, 2:6]], axis=1) for track_pl, nearby_track in zip(track_pls, nearby_track_features)]
        return track_pls

    def interp_track_features(self, track_pls):
        interp_pls = []
        for pl in track_pls:
            coords = pl[:, 2:4]
            velocities = pl[:, 4:]
            interp_pl = pl.repeat(math.ceil(self.pts_per_pl / self.timesteps_history), axis=0)[:self.pts_per_pl]

            eps = np.expand_dims(np.linspace(1e-7, 1e-6, pl.shape[0]), axis=1)
            velocities += eps  # some velocities (static objects) are exactly 0.0, interp_arc creates nan for these

            interp_pl[:self.pts_per_pl, 2:] = np.concatenate([interp_utils.interp_arc(self.pts_per_pl, coords), interp_utils.interp_arc(self.pts_per_pl, velocities)], axis=1)
            interp_pls.append(interp_pl)

        track_pls = np.stack(interp_pls, axis=0)
        track_pls[:, :, 1] -= self.start_timestep
        return track_pls

    def interp_for_parallel(self, track_pls):
        interp_pls = []
        for pl in track_pls:
            xys = pl[:, 2:4]
            velocities = pl[:, 4:6]
            xys_ = pl[:, 7:9]  # 7:9
            velocities_ = pl[:, 9:]  # 9:
            interp_pl = pl.repeat(math.ceil(self.pts_per_pl / self.timesteps_history), axis=0)[:self.pts_per_pl]

            eps = np.expand_dims(np.linspace(1e-7, 1e-6, pl.shape[0]), axis=1)
            velocities += eps  # some velocities (static objects) are exactly 0.0, interp_arc creates nan for these
            velocities_ += eps
            xys_ += eps

            interp_arcs = [interp_utils.interp_arc(self.pts_per_pl, arc) for arc in [xys, velocities, xys_, velocities_]]
            interp_pl[:, 2:6] = np.concatenate(interp_arcs[:2], axis=1)
            interp_pl[:, 7:] = np.concatenate(interp_arcs[2:], axis=1)  # 7:
            interp_pls.append(interp_pl)

        track_pls = np.stack(interp_pls, axis=0)
        track_pls[:, :, 1] -= self.start_timestep
        return track_pls

    def get_ego_track(self, tracks, ego_id):
        ego_track = tracks[tracks["track_id"] == ego_id][["position_x", "position_y", "heading"]].to_numpy()
        ego_loc = ego_track[self.timesteps_history - 1, :2]
        return ego_track, ego_loc


def normalize_coords(lane_polyline_coords, ped_crossings, track_pls, ego_loc):
    lane_polyline_coords -= ego_loc
    if ped_crossings.shape[0] != 0:
        ped_crossings -= ego_loc
    track_pls[:, :, 2:4] -= ego_loc

def normalize_by_min(lane_polyline_coords, ped_crossings, track_pls):
    if ped_crossings.shape[0] != 0:
        xmin = np.min([lane_polyline_coords[:, :, 0].min(), ped_crossings[:, :, 0].min(), track_pls[:, :, 2].min()])
        ymin = np.min([lane_polyline_coords[:, :, 1].min(), ped_crossings[:, :, 1].min(), track_pls[:, :, 3].min()])
    else:
        xmin = np.min([lane_polyline_coords[:, :, 0].min(), track_pls[:, :, 2].min()])
        ymin = np.min([lane_polyline_coords[:, :, 1].min(), track_pls[:, :, 3].min()])
    mins = np.array([xmin, ymin], dtype=np.float32)
    lane_polyline_coords[:, :, :2] -= mins
    if ped_crossings.shape[0] != 0:
        ped_crossings -= mins
    track_pls[:, :, 2:4] -= mins
    return mins

def join_pls(ped_crossings, lane_polylines, track_pls, pts_per_pl):
    num_pls = np.array([ped_crossings.shape[0], lane_polylines.shape[0], track_pls.shape[0]])
    num_features = np.array([ped_crossings.shape[2], lane_polylines.shape[2], track_pls.shape[2]])
    polyline_indeces = np.expand_dims(np.arange(0, num_pls.sum()), axis=1).repeat(pts_per_pl, axis=1)  # extra feature, index of polyline

    input_pls = np.zeros((num_pls.sum(), pts_per_pl, num_features.sum() + 1), dtype=np.float32) # 1 for polyline index

    cumul_num_pls = np.cumsum(num_pls)
    cumul_num_features = np.cumsum(num_features)  # to write the indexing a little simpler

    if ped_crossings.shape[0] != 0:
        input_pls[:cumul_num_pls[0], :, :cumul_num_features[0]] = ped_crossings
    input_pls[cumul_num_pls[0] : cumul_num_pls[1], :, cumul_num_features[0] : cumul_num_features[1]] = lane_polylines
    input_pls[cumul_num_pls[1]:, :, cumul_num_features[1] : -1] = track_pls
    input_pls[:, :, -1] = polyline_indeces

    return input_pls

def get_label(ego_track, timesteps_history):
    ego_coords = ego_track[:, :2]
    ego_heading = ego_track[:, 2]
    label = ego_coords[timesteps_history:] - ego_coords[timesteps_history - 1:-1]  # offsets, as in VectorNet paper
    rotate(label, ego_heading[-1] - (math.pi / 2))
    return label.astype(np.float32)

def left_right(initial_angle, final_angle):
    a1_to_a2 = final_angle - initial_angle
    a1_to_a2[a1_to_a2 < 0] += 2 * np.pi  # positive value for how much to rotate from angle 1 to get to angle 2
    # if you would have to turn ccw less than pi, turning left
    # if you would have to turn ccw more than pi, turning right
    left_turn = a1_to_a2 <= np.pi  
    right_turn = a1_to_a2 > np.pi
    return left_turn, right_turn

def angle_between(angle1, angle2):
    angle_between = torch.abs(angle2 - angle1)
    angle_between = torch.minimum(angle_between, 2 * np.pi - angle_between)
    return angle_between

def batch_intersect(coords, heading_threshold=3):
    '''
    heading_threshold - threshold in degrees for which similar headings will be interpreted as just a line
    coords shape - (N, 2, 3) (2 lines, (x, y, angle))
    out shape - (N, 2) (x, y intersection)
    solve system of equations: -m0x + y = b0
                               -m1x + y = b1    Solving for the point that satisfies the equations of both lines
    '''
    angle1 = coords[:, 0, 2]
    angle2 = coords[:, 1, 2]
    diff = coords[:, 1, :2] - coords[:, 0, :2]
    angle3 = torch.arctan2(diff[:, 1], diff[:, 0])
    angle1[angle1 < 0] += 2 * np.pi
    torch.fmod(angle1, 2 * np.pi, out=angle1)
    angle2[angle2 < 0] += 2 * np.pi  # conditions only work with positive angles (based on less than and greater than)
    torch.fmod(angle2, 2 * np.pi, out=angle2)
    angle3[angle3 < 0] += 2 * np.pi  # arctan2 is -pi to pi. Rotate negative values to positive

    # FOR CASES WHERE END HEADING IS POINTING FURTHER LEFT, BUT END POSITION IS TO THE RIGHT OF INITIAL HEADING OR OTHER WAY AROUND
        # this will cause the bezier curve to point the wrong way
    # MIRROR THE INITIAL HEADING AROUND THE LINE BETWEEN INITIAL AND FINAL POSITION
    went_left, went_right = left_right(angle1, angle3)  # based on initial heading and final location, ie. did you go left or right
    left_turn, right_turn = left_right(angle1, angle2)  # based on initial heading and final heading, ie. are you turning left or right in the end
    cond1 = went_left & right_turn
    cond2 = went_right & left_turn
    coords[:, 0, 2] = torch.where(cond1 | cond2, torch.fmod(2 * angle3 - angle1, 2 * np.pi), angle1)  # angle3 - (angle1 - angle3) = angle3 + (angle3 - angle1) = 2 * angle3 - angle1    
    angle1[angle1 < 0] += 2 * np.pi

    m = torch.tan(coords[:, :, 2])
    b = coords[:, :, 1] - m * coords[:, :, 0]  # b = y - slope * x
    lhs = torch.stack([-m, torch.ones_like(m)], dim=2)
    eps = 1e-6
    lhs[:, :, 1] += eps
    try:
        intersection = torch.linalg.solve(lhs, b)
    except torch._C._LinAlgError:
        intersection, _, _, _ = torch.linalg.lstsq(lhs, b)

    # FILTER CASES WHERE INTERSECTION IS BEHIND THE VEHICLE
    angle4 = intersection - coords[:, 0, :2]
    angle4 = torch.arctan2(angle4[:, 1], angle4[:, 0])
    angle4[angle4 < 0] += 2 * np.pi
    a_between = angle_between(angle1, angle4)
    heading_vector = torch.stack([torch.cos(coords[:, 0, 2]), torch.sin(coords[:, 0, 2])], dim=1)  # already unit length
    intersection = torch.where((a_between >= (np.pi / 2)).unsqueeze(1), coords[:, 0, :2] + heading_vector * 5, intersection)
    
    # FILTER CASES WHERE INTERSECTION IS NOT BETWEEN THE INITIAL AND FINAL POINTS
        # change headings that are too similar to just a line (close headings may lead to far away intersection point)
    midpoint = (coords[:, 1, :2] + coords[:, 0, :2]) /  2
    heading_threshold *= np.pi / 180
    a_between = angle_between(angle1, angle2)
    if not torch.all((a_between <= np.pi) & (a_between >= 0)):
        print(angle1)
        print(angle2)
        print(angle3)

    intersection = torch.where(((a_between <= heading_threshold) | ((np.pi - a_between) <= heading_threshold)).unsqueeze(1), midpoint, intersection)

    return intersection

def get_frenet_labels(p0, p1, p2, v0, trajectories, timesteps_future, device, sample_precision=100):
    '''
    sample points along the curve (100 or so), calculate cumulative arc length along the curve
            calculate constant acceleration, for each timestep (30) find closest cum arc length (s coordinate, d coordinate is 0)
            for each point in each track, find the closest point on the curve (from sampled points) 
                retrieve cum arc length (s) and d coordinate (length of vector from point on curve to the trajectory point)
                    verify vector is perpendicular to the curve (it is)
            calcuate residuals as track frenet point - constant acc frenet point
    '''
    s = torch.linspace(0, 1, steps=(sample_precision + 1), device=device)  # could increase precision for inference
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
    acc = (2 * (cum_arclength[:, -1] - (v0 * sec_future))) / (sec_future ** 2)  # assume and calculate constand acceleration along bezier arc
    timesteps = torch.linspace(0.1, sec_future, timesteps_future, device=device)  # curve is between last observed point and end point/first predicted point is at time 0.1
    timesteps = timesteps.unsqueeze(0)
    acc = acc.unsqueeze(1)
    v0 = v0.unsqueeze(1)
    s_t_const_acc = v0 * timesteps + .5 * acc * (timesteps ** 2)  # shape (batch, timesteps_future)

    pttotraj_vectors = trajectories.unsqueeze(2) - bezier_pts[:, 1:, :].unsqueeze(1)  # only last 100 bezier points, for indexing with cum_arc_length
    dist_to_line = torch.linalg.norm(pttotraj_vectors, dim=-1)
    closest_pt = dist_to_line.argmin(dim=-1)  # index between 0, 99 (not 100 because of pttotraj_ bezier cutoff)
    s_t_traj = torch.take_along_dim(cum_arclength, closest_pt, dim=1)

    # select pttotraj_vectors at closest_pt, dot with orthogonal_vectors at closest_pt, (verify ==+-1) multiply by original vector
    closest_pt.unsqueeze_(2)
    reduced_orth_vectors = torch.take_along_dim(orthogonal_vectors, closest_pt, dim=1)
    reduced_pttraj_vectors = torch.take_along_dim(pttotraj_vectors, closest_pt.unsqueeze(3), dim=2).squeeze(2)
    
    d_sign = (reduced_orth_vectors * reduced_pttraj_vectors).sum(dim=-1)  # dot product between orthogonal vectors and d coordinate vectors
    d_t_traj = torch.take_along_dim(dist_to_line, closest_pt, dim=2).squeeze(2)
    d_t_traj[d_sign < 0] *= -1

    residuals = torch.stack([s_t_traj - s_t_const_acc, d_t_traj], dim=2)
    return residuals

def get_cls_labels(lane_pls, tracks):
    '''
    lane_pls (num_lanes, 64, 2)
    tracks (num_tracks, 3)

    find closest lane_pl coord, calculate angle of lane near point
    return track coord - lane_pl coord, track angle - lane angle
    '''
    tracks[tracks[:, 2] < 0, 2] += 2 * np.pi

    dist = tracks[:, np.newaxis, np.newaxis, :2] - lane_pls[np.newaxis, :, :, :]  # num_agents, num_lanes, pts_per_lane, 2
    dist = np.linalg.norm(dist, axis=-1)
    dist = dist.reshape(dist.shape[0], -1)
    cls_label = dist.argmin(axis=1)

    closest5, idx_closest5 = torch.from_numpy(dist).topk(5, largest=False, dim=-1)
    closest_node = unravel_idx(idx_closest5, (lane_pls.shape[0], lane_pls.shape[1]))
    closest5 = closest5.numpy()
    idx_closest5 = idx_closest5.numpy()
    # final_angle = lane_angles[(batch_range, closest_node[0], closest_node[1].clamp(0, self.pts_per_pl - offset - 1))]

    ## MATCH FINAL ANGLE TO LANE ANGLE POINTING IN THE SAME DIRECTION (WITHIN 60 DEGREES) TO PREVENT SELECTING OPPOSING TRAFFIC LANE WHEN OVERLAPPING
    lane_angles = lane_pls[:, 4:, :] - lane_pls[:, :-4, :]
    lane_angles = np.arctan2(lane_angles[..., 1], lane_angles[..., 0])
    lane_angles[lane_angles < 0] += 2 * np.pi  # shape (batch, num_lanes, 60)

    final_angle = lane_angles[(closest_node[0], closest_node[1].clamp(0, 64 - 4 - 1))]
    agent_angle = tracks[:, 2]
    angle_between = np.abs(final_angle - np.expand_dims(agent_angle, axis=1))
    angle_between = np.minimum(angle_between, 2 * np.pi - angle_between)
    eps = 1e-5
    closest5 += eps
    closest5[angle_between <= (np.pi / 3)] = 0
    cls_label = np.take_along_axis(idx_closest5, closest5.argmin(axis=-1, keepdims=True), axis=-1).squeeze(-1)
    
    indeces = np.unravel_index(cls_label, (lane_pls.shape[0], lane_pls.shape[1]))
    closest_pt = lane_pls[indeces]

    offset = 4
    lane_angle = lane_pls[:, offset:, :] - lane_pls[:, :-offset, :]
    lane_angle = np.arctan2(lane_angle[:, :, 1], lane_angle[:, :, 0])
    lane_angle[lane_angle < 0] += 2 * np.pi
    
    np.clip(indeces[1], 0, lane_pls.shape[1] - offset - 1, out=indeces[1])
    lane_angle = lane_angle[indeces]

    pt_residuals = tracks[:, :2] - closest_pt
    angle_residuals = tracks[:, 2] - lane_angle
    cls_label = np.concatenate([np.expand_dims(cls_label, axis=1), pt_residuals, np.expand_dims(angle_residuals, axis=1)], axis=1)
    return cls_label  # shape (num_tracks, 4) (index for flattened (num_lanes, pts_per_pl), x residual, y residual, angle residual (relative to lane))

def unravel(indices, shape):
    indices = indices.clone()
    idx = []

    for dim in reversed(shape):
        idx.append(indices % dim)
        indices //= dim

    idx = torch.stack(idx[::-1], dim=-1)

    return idx

def unravel_idx(indices, shape):
    shape_tensor = indices.new_tensor(shape)

    coefs = shape_tensor[1:].flipud().cumprod(dim=0).flipud()
    coefs = torch.cat((coefs, coefs.new_tensor((1,))), dim=0)
    coords = torch.div(indices[..., None], coefs, rounding_mode='trunc') % shape_tensor

    return tuple(coords[..., i] for i in range(coords.size(-1)))

def rotate(coords, theta):
    sinx = np.sin(theta)
    cosx = np.cos(theta)
    rot_mat = np.array([[cosx, -sinx], 
                        [sinx, cosx]])  # standard rotation matrix, transposed to match input shape (N, 2)
    np.matmul(coords, rot_mat, out=coords)

def rotate_batch(coords, theta):
    '''
    each member of the batch will be rotated by its respective theta
    coords shape - (batch_size, N, 2)
    theta shape - (batch_size)
    out shape - (batch_size, N, 2)
    '''
    sinx = np.sin(theta)
    cosx = np.cos(theta)
    rot_mat = np.stack([np.stack([cosx, sinx], axis=1), np.stack([-sinx, cosx], axis=1)], axis=2)  # along the last two dims, has the same look as rotate above
    return np.matmul(coords, rot_mat)


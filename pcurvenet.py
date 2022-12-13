from itertools import chain
import os
import argparse
import torch as torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from torch.utils.data import DataLoader

from subgraph_net import SubGraphNet
from globgraph_net import GlobalGraphNet, GlobalGraphNet2
from decoder import MLP
from dataset import AV2
import data_utils

import matplotlib.pyplot as plt

class LaneUtils:
    def __init__(self, input_pls, cls_labels, gt_traj, num_lanes, num_agents, topn=6):
        self.input_pls = input_pls
        self.cls_labels = cls_labels
        self.gt_trajectories = gt_traj
        self.num_lanes = num_lanes
        self.num_agents = num_agents

        self.topn = topn
        self.batch_range = torch.arange(self.input_pls.shape[0])

    def get_lane_coords(self):
        offset = 2
        self.lane_coords = self.input_pls[:, -(self.num_lanes + self.num_agents):-self.num_agents, :, 2:4]  # xy coordinates for each lane node
        lane_angles = self.lane_coords[:, :, offset:, :] - self.lane_coords[:, :, :-offset, :]
        lane_angles = torch.arctan2(lane_angles[..., 1], lane_angles[..., 0])
        lane_angles[lane_angles < 0] += 2 * np.pi  # shape (batch, num_lanes, 60)
        self.clane_angles = torch.cat([lane_angles[:, :, -1:].expand(-1, -1, offset), lane_angles], dim=2)  # (batch, num_lanes, 64)
        return self.lane_coords, self.clane_angles

    def get_initial_pt(self, lane_coords, clane_angles, v0):
        ## REPLACE UNRELIABLE INITIAL ANGLES WITH CLOSEST LANE ANGLE
        initial_pt = self.gt_trajectories[:, :, 0, :]
        initial_node = torch.linalg.norm(self.gt_trajectories[:, :, 0, None, :2] - lane_coords.view(1, 1, self.num_lanes * 64, 2), dim=-1)
        initial_node = initial_node.argmin(dim=-1)
        # (batch, num_agents)
        initial_lane_angle = clane_angles.view(-1, self.num_lanes * 64)[self.batch_range, initial_node]
        # (batch, num_agents)
        agent_stationary = v0 < 0.5  # gt trajectories include angles, these labels are inaccurate for stationary vehicles
        initial_pt[..., 2] = torch.where(agent_stationary.unsqueeze(0), initial_lane_angle, initial_pt[..., 2])
        self.initial_pt = initial_pt[0, :, None, :].repeat(1, self.topn, 1)
        return self.initial_pt

    def get_final_angle(self, lane_coords, clane_angles, initial_pt, closest_node_coords, closest_reg_out):
        ## SELECT FINAL ANGLE FROM 5 CLOSEST NODES TO SELECTED ONE BY THE ANGLE WHICH IS CLOSEST TO THE VECTOR THAT WOULD BE TAKEN TO GET TO THAT NODE
        # for overlapping lanes, this will select a good lane for the base angle used for the traj prediction
        closest5 = torch.linalg.norm((closest_node_coords + closest_reg_out[..., :2]).unsqueeze(3) - lane_coords.view(1, 1, 1, self.num_lanes * 64, 2), dim=-1)
        closest5, idx_closest5 = closest5.topk(10, largest=False, dim=-1)

        vec_to_node = lane_coords[self.batch_range, idx_closest5] - initial_pt[None, :, :, None, :2]
        angle_to_node = torch.atan2(vec_to_node[..., 1], vec_to_node[..., 0])
        angle_to_node[angle_to_node < 0] += 2 * np.pi
        final_angle = clane_angles.view(-1, self.num_lanes * 64)[self.batch_range, idx_closest5]
        lane_vs_path = data_utils.angle_between(final_angle, angle_to_node)
        lane_vs_init = data_utils.angle_between(final_angle, initial_pt[..., 2].view(1, -1, self.topn, 1))
        lane_vs_path[lane_vs_init < (np.pi / 3)] = 0
        self.final_angle = final_angle.take_along_dim(lane_vs_path.argmin(dim=-1, keepdim=True), dim=-1).squeeze(-1)
        return self.final_angle


class ForwardPass:
    def __init__(self, dataset, idx, topn):
        self.topn = topn
        self.avm, _ = dataset.load_example(dataset.get_example(idx)) 

        self.input_pls, self.cls_labels, self.gt_trajectories, self.num_lane_pls, self.mins, self.ego_idx = dataset[idx]
        self.ego_idx = self.ego_idx.item()
        self.input_pls = torch.as_tensor(self.input_pls, dtype=torch.float32).unsqueeze(0)
        self.cls_labels = torch.as_tensor(self.cls_labels, dtype=torch.float32).unsqueeze(0)
        self.gt_trajectories = torch.as_tensor(self.gt_trajectories, dtype=torch.float32).unsqueeze(0)
        self.num_lanes = self.num_lane_pls[0]
        self.num_agents = self.cls_labels.shape[1]
        self.lane_utils = LaneUtils(self.input_pls, self.cls_labels, self.gt_trajectories, self.num_lanes, self.num_agents, topn=self.topn)

        self.batch_range = torch.arange(self.input_pls.shape[0])

        self.closest_reg_out = None
        self.closest_traj_out = None
        self.traj_labels = None

        self.lane_coords = None
        self.clane_angles = None
        self.initial_pt = None
        self.final_pt = None
        self.final_angle = None
        self.p0 = None
        self.p1 = None
        self.p2 = None
        self.v0 = None

    def forward(self, model, set_labels=True):
        cls_out, traj_out = model(self.input_pls, self.num_lanes, self.num_agents)
        closest_node = cls_out[..., 0].view(-1, self.num_agents, self.num_lanes * 64)
        # closest_node = torch.nn.functional.softmax(closest_node, dim=-1)
        vals, closest_node = closest_node.topk(self.topn, dim=-1)

        closest_reg_out = cls_out[..., 1:].view(-1, self.num_agents, 1, self.num_lanes * 64, 3)
        self.closest_reg_out = closest_reg_out.take_along_dim(closest_node.view(1, self.num_agents, self.topn, 1, 1), dim=3).squeeze(3)

        closest_traj_out = traj_out.view(-1, self.num_agents, 1, self.num_lanes * 64, traj_out.shape[-1])
        self.closest_traj_out = closest_traj_out.take_along_dim(closest_node.view(1, self.num_agents, self.topn, 1, 1), dim=3).squeeze(3)
        # batch, num_agents, num_lanes, ptsperpl, 4
        # traj out = batch, num_agents, num_lanes, ptsperpl, 60

        self.v0 = torch.linalg.norm(self.input_pls[:, -self.num_agents:, -1, 13:15], dim=-1).squeeze(0)

        self.lane_coords, self.clane_angles = self.lane_utils.get_lane_coords()

        self.lane_coords = self.lane_coords.view(-1, self.num_lanes * 64, 2)
        closest_node_coords = self.lane_coords[self.batch_range, closest_node]

        self.initial_pt = self.lane_utils.get_initial_pt(self.lane_coords, self.clane_angles, self.v0)

        self.final_angle = self.lane_utils.get_final_angle(self.lane_coords, self.clane_angles, self.initial_pt, closest_node_coords, self.closest_reg_out)

        self.final_pt = torch.cat([closest_node_coords, self.final_angle.unsqueeze(3)], dim=3)
        self.final_pt += self.closest_reg_out  # += predicted residuals

        pts = torch.stack([self.initial_pt.reshape(-1, 3), self.final_pt.reshape(-1, 3)], dim=1)
        intersections = data_utils.batch_intersect(pts, heading_threshold=5)

        self.p0 = pts[:, 0, :2].unsqueeze(1)
        self.p1 = intersections.unsqueeze(1)
        self.p2 = pts[:, 1, :2].unsqueeze(1)
    
        pts = self.get_pts(self.closest_traj_out)
        if set_labels:
            self.set_traj_labels()
        return pts

    def get_pts(self, traj):
        # gt_trajectories contain last observed point (so skip this one) + headings for each point
        timesteps_future = 30
        s = torch.linspace(0, 1, steps=(128 + 1), device='cpu')  # could increase precision for inference
        s = s.unsqueeze(1)
        bezier_pts = ((1 - s)**2) * self.p0 + 2 * s * (1 - s) * self.p1 + (s**2) * (self.p2)  # (1-s)^2 p0 + 2s(1-s)p1 + s^2 p2 (shape = (batch, 101, 2))
        s = s.squeeze(1)
        diff = torch.diff(bezier_pts, dim=1)

        orthogonal_vectors = torch.flip(diff, dims=(2,))  # switch x and y coordinates and negate new y (gives orthogonal vector pointing right) (eg. (3, 1) -> (1, -3))
        orthogonal_vectors[:, :, 1] *= -1

        arclength = torch.linalg.norm(diff, dim=2)  # shape (batch, 100)
        orthogonal_vectors /= arclength.unsqueeze(2)
        cum_arclength = arclength.cumsum(dim=1)

        sec_future = timesteps_future / 10
        v0 = self.v0.repeat(self.topn)
        acc = (2 * (cum_arclength[:, -1] - (v0 * sec_future))) / (sec_future ** 2)  # assume and calculate constand acceleration along bezier arc
        timesteps = torch.linspace(0.1, sec_future, timesteps_future, device='cpu')  # curve is between last observed point and end point/first predicted point is at time 0.1
        timesteps = timesteps.unsqueeze(0)
        acc = acc.unsqueeze(1)
        v0 = v0.unsqueeze(1)
        s_t_const_acc = v0 * timesteps + .5 * acc * (timesteps ** 2)  # shape (batch, timesteps_future)

        traj = traj.view(-1, self.num_agents * self.topn, timesteps_future, 2)
        s_t_traj = s_t_const_acc + traj[0, :, :, 0]
        stos = torch.abs(s_t_traj[:, :, None] - cum_arclength[:, None, :])
        indices = stos.argmin(dim=-1)
        s_norm = s[indices]

        s_norm.unsqueeze_(2)
        along_track = ((1 - s_norm)**2) * self.p0 + 2 * s_norm * (1 - s_norm) * self.p1 + (s_norm**2) * (self.p2)
        d_t_traj = traj[0, :, :, 1].view(self.num_agents * self.topn, 30)
        reduced_orth_vectors = orthogonal_vectors.take_along_dim(indices.unsqueeze(2), dim=1)
        cross_track = d_t_traj.unsqueeze(2) * reduced_orth_vectors
        pts = along_track + cross_track
        return pts

    def set_traj_labels(self):
        timesteps_future = 30
        self.traj_labels = data_utils.get_frenet_labels(self.p0[:self.num_agents], self.p1[:self.num_agents], self.p2[:self.num_agents], self.v0, self.gt_trajectories[0, :, 1:, :2], timesteps_future, device=self.p0.device, sample_precision=128)

    def count_lanes():
        ## TOO MANY LANES OVERLAPPING PREVENT THIS FROM REALLY BEING ACCURATE, NOT FOR USE
        ## compute number of different lanes predicted
        idx_lane_nodes = idx_closest5.take_along_dim(lane_vs_path.argmin(dim=-1, keepdim=True), dim=-1).squeeze(-1)
        idx_lane_nodes = idx_lane_nodes.div(64, rounding_mode='trunc').squeeze(0)
        for agt in idx_lane_nodes:
            uniq_lanes_per_agent += agt.unique().numel()


class TrainStep:
    def __init__(self, input_pls, closest_node, cls_labels, reg_label, gt_trajectories, cls_out, traj_out, num_agents, num_lanes, pts_per_pl, sec_future, offset=4):
        self.input_pls = input_pls
        self.closest_node = closest_node
        self.reg_label = reg_label
        self.gt_trajectories = gt_trajectories
        self.cls_out = cls_out
        self.num_agents = num_agents
        self.num_lanes = num_lanes
        self.pts_per_pl = pts_per_pl
        self.sec_future = sec_future
        self.offset = offset

        self.lane_utils = LaneUtils(self.input_pls, cls_labels, gt_trajectories, self.num_lanes, self.num_agents, topn=1)

        self.reg_out = cls_out[..., 1:].view(-1, num_agents, num_lanes * self.pts_per_pl, 3)
        self.traj_out = traj_out.view(-1, num_agents, num_lanes * self.pts_per_pl, traj_out.shape[-1])

        # self.closest_reg_out = self.reg_out.take_along_dim(closest_node.view(-1, num_agents, 1, 1), dim=2).squeeze(2)
        # self.closest_traj_out = self.traj_out.take_along_dim(closest_node.view(-1, num_agents, 1, 1), dim=2).squeeze(2)
        del traj_out

        self.closest_node_coords = None

        self.batch_range = torch.arange(input_pls.shape[0], device='cuda')
        self.dist_threshold = 3  # dist threshold for negative/positive node mask  

        self.ce_loss = torch.nn.CrossEntropyLoss()

    def get_losses(self, bce_loss, smoothl1_loss):
        with torch.no_grad():
            lane_coords, self.clane_angles = self.lane_utils.get_lane_coords()
            # lane_coords = self.input_pls[:, -(self.num_lanes + self.num_agents):-self.num_agents, :, 2:4]  # xy coordinates for each lane node
            v0 = torch.linalg.norm(self.input_pls[:, -self.num_agents:, -1, 13:15], dim=-1).squeeze(0)

            lane_coords = lane_coords.view(-1, self.num_lanes * self.pts_per_pl, 2)
            self.closest_node_coords = lane_coords[self.batch_range, self.closest_node]
            dist = torch.linalg.norm(self.closest_node_coords.unsqueeze(2) - lane_coords.unsqueeze(1), dim=-1)
            dist_mask = dist > self.dist_threshold  # all nodes more than thresh meters away from the positive node (closest to end position) are negative nodes
            # shape (batch, num_agents, num_lanes * pts_per_pl)

            self.reg_label = self.gt_trajectories[:, :, -1, None, :2] - lane_coords.view(1, 1, self.num_lanes * 64, 2)
            closest64 = torch.linalg.norm(self.reg_label, dim=-1)
            closest64, idx_closest64 = closest64.topk(64, largest=False, dim=-1)
            # (batch, num_agents, 64)
            self.reg_label = self.reg_label.take_along_dim(idx_closest64.unsqueeze(3), dim=2)
            self.reg_label = torch.cat([self.reg_label, torch.zeros_like(self.reg_label[..., 0:1])], dim=-1).view(1, -1, 3)
            self.dist_mask64 = closest64 < self.dist_threshold

            self.closest_reg_out = self.reg_out.take_along_dim(idx_closest64.unsqueeze(3), dim=2).view(1, -1, 3)
            self.closest_traj_out = self.traj_out.take_along_dim(idx_closest64.unsqueeze(3), dim=2)
            self.closest_node_coords = lane_coords.view(1, 1, self.num_lanes * 64, 2).take_along_dim(idx_closest64.unsqueeze(3), dim=2).view(1, -1, 2)

            pts, intersections, final_angle = self.get_pts(lane_coords, self.clane_angles, v0)

            p0 = pts[:, 0, :2].unsqueeze(1)
            p1 = intersections.unsqueeze(1)
            p2 = pts[:, 1, :2].unsqueeze(1)
            timesteps_future = self.sec_future * 10

            ## generate frenet labels (relative to predicted bezier curve)
            # gt_trajectories contain last observed point (so skip this one) + headings for each point
            traj_labels = data_utils.get_frenet_labels(p0, p1, p2, v0.repeat(64), self.gt_trajectories[0, :, 1:, :2].repeat(64, 1, 1), timesteps_future, device='cuda', sample_precision=128)

            pos_pts, alt_mask = self.get_alt_goals(lane_coords, self.clane_angles, v0, dist_mask)

        cls_losses = self.get_cls_loss(bce_loss, alt_mask, pos_pts=pos_pts)  # dist_mask for non-alternate goal version
        # cls_loss = self.get_ce_loss()

        # SELECT ANGLE ACCORDING TO CLOSEST5 + RESIDUAL, SELECT LABEL ACCORDING TO GT_TRAJ
        self.closest_reg_out[:, :, 2] += final_angle
        self.fix_angles(final_angle)

        self.closest_reg_out = self.closest_reg_out.masked_select(self.dist_mask64.view(1, -1, 1))
        self.reg_label = self.reg_label.masked_select(self.dist_mask64.view(1, -1, 1))
        reg_loss = smoothl1_loss(self.closest_reg_out, self.reg_label)

        self.closest_traj_out = self.closest_traj_out.view(1, -1, timesteps_future, 2).masked_select(self.dist_mask64.view(1, -1, 1, 1))
        traj_labels = traj_labels.unsqueeze(0).masked_select(self.dist_mask64.view(1, -1, 1, 1))
        traj_loss = smoothl1_loss(self.closest_traj_out, traj_labels)  # .view(1, -1, timesteps_future, 2), .unsqueeze(0)

        return cls_losses, reg_loss, traj_loss

    def fix_angles(self, final_angle):
        '''
        Corrects the reg_label and sets the prediction and target so as to minimize the angle between them
        '''
        # label - (batch, agents, 3), traj - (batch, agents, timesteps, 3)
        # The labels for heading when the car is not moving are innacurate, they can point in any direction.
        # so, in these cases, predicted residual should be 0/label should be == final_angle (ie. just predict the lane angle as heading)
        dist_moved = torch.linalg.norm((self.gt_trajectories[:, :, -1, :2] - self.gt_trajectories[:, :, 0, :2]), dim=-1)
        dist_moved = dist_moved.repeat(1, 64)
        self.reg_label[:, :, 2] = torch.where(dist_moved < 2, final_angle, self.reg_label[:, :, 2])

        ## Essentially, with setting angle label to 0, minimize the angle between the predicted final angle and the actual final angle
        ## Most correct because angle_between calculates (as a positive angle) the smallest angle between the two, which is what should actually be minimized
            ## any kind of simple difference ignores the (example - occurs whenever abs(a1 - a2) > pi) fact that instead of rotating +3pi/2, 
            ## you could rotate -pi/2, and the therefore the predicted residual angle should be -pi/2 (for stability)
        self.closest_reg_out[:, :, 2] = data_utils.angle_between(self.closest_reg_out[:, :, 2], self.reg_label[:, :, 2])
        self.reg_label[:, :, 2] = 0

    def get_alt_goals(self, lane_coords, lane_angles, v0, dist_mask):
        # create mask of points where lane angle matches direction from agent
        # create point index per lane that matches constant velocity distance, expand to one hot mask along pts_per_pl dim
        # fmod closest_node (gt) by 64, set lane/dist mask to false in that lane (so lane with gt in it does not also have a pt selected via this method)
        # AND of the masks are positive points
        # loop through agents, select positive nodes loc and all lane locs, distance between each (pos, total_pts), select < 2m, bitwise or along positive nodes dim, negate
            # cat these for each agent (agent, total_pts)
            # this is mask for negative nodes (replaces dist_mask)
        # have mask for positive nodes and negative nodes
        dist_to_pt = lane_coords[:, None, :, :] - self.gt_trajectories[:, :, 0:1, :2]  # (batch, num_agents, num_lanes * self.pts_per_pl, 2)
        angle_to_pt = torch.atan2(dist_to_pt[..., 1], dist_to_pt[..., 0])
        angle_to_pt[angle_to_pt < 0] += 2 * np.pi
        valid_lane = data_utils.angle_between(angle_to_pt, lane_angles.view(-1, 1, self.num_lanes * self.pts_per_pl)) <= (np.pi / 3)
        in_front = data_utils.angle_between(self.gt_trajectories[:, :, 0:1, 2], angle_to_pt) <= (100 * np.pi / 180)
        valid_pts = valid_lane & in_front

        proj_dist = v0.unsqueeze(0) * self.sec_future
        dist_to_pt = torch.linalg.norm(dist_to_pt, dim=-1)  # batch, num_agents, num_lanes * self.pts_per_pl
        matching_node = torch.abs(dist_to_pt - proj_dist.unsqueeze(2))

        msk = matching_node < 3  # make sure they are reasonably close to goal distance 
        matching_node[~in_front] = torch.inf  # select correct points in front of agent, not behind
        matching_node = matching_node.view(-1, self.num_agents, self.num_lanes, self.pts_per_pl).argmin(dim=-1)
        matching_node = torch.nn.functional.one_hot(matching_node, self.pts_per_pl)

        closest_node_ = torch.div(self.closest_node, self.pts_per_pl, rounding_mode='trunc')  # batch, num_agents
        matching_node[self.batch_range, torch.arange(self.num_agents, device='cuda'), closest_node_] = 0  # in lane where gt endpoint is, set entire lane to false (so not overlapping with gt)
        matching_node = matching_node.bool().view(-1, self.num_agents, self.num_lanes * self.pts_per_pl)
        pos_pts = valid_pts & matching_node & msk

        dist_masks = []
        for agent in range(self.num_agents):
            pos_coords = lane_coords.masked_select(pos_pts[:, agent].unsqueeze(2)).view(lane_coords.shape[0], -1, 2)
            pos_dist = torch.linalg.norm(pos_coords[:, :, None, :] - lane_coords[:, None, :, :], dim=-1)  # (batch, num_pos_lanes, num_lanes * 64)
            mask = torch.bitwise_not((pos_dist < self.dist_threshold).sum(dim=1).bool())  # bitwise or along num_positives dim. mask = ~ dist < dist_threshold = dist >= dist_threshold
            dist_masks.append(mask)
        alt_mask = torch.bitwise_or(dist_mask, torch.stack(dist_masks, dim=1))

        return pos_pts, alt_mask

    def get_cls_loss(self, bce_loss, dist_mask, pos_pts=None):
        cls_logits = self.cls_out[..., 0].view(-1, self.num_agents, self.num_lanes * self.pts_per_pl)
        positive_samples = cls_logits.take_along_dim(self.closest_node.unsqueeze(2), dim=2).squeeze(2)
        negative_samples = cls_logits.masked_select(dist_mask)

        negative_loss = bce_loss(negative_samples, torch.zeros((1,), device='cuda').expand(negative_samples.shape[0]))  # no reduction, same shape for topk
        
        positive_loss = bce_loss(positive_samples, torch.ones_like(positive_samples)).mean()
        negative_loss = negative_loss.mean()

        alternate_loss = 0
        if pos_pts is not None:
            alternate_samples = cls_logits.masked_select(pos_pts)
            if alternate_samples.numel() != 0:
                alternate_loss = bce_loss(alternate_samples, torch.ones((1,), device='cuda').expand(alternate_samples.shape[0])).mean()
        
        return positive_loss, alternate_loss, negative_loss

    def get_ce_loss(self):
        cls_logits = self.cls_out[..., 0].view(self.num_agents, self.num_lanes * self.pts_per_pl)
        return self.ce_loss(cls_logits, self.closest_node.squeeze(0))

    def get_pts(self, lane_coords, clane_angles, v0):
        initial_pt = self.lane_utils.get_initial_pt(lane_coords, clane_angles, v0)  # (num_agents, topn, 3)
        initial_pt = initial_pt.repeat(64, 1, 1)

        # unsqueeze 2 for topn dimension. Here only one, but in metrics or validate, uses top 6
        final_angle = self.lane_utils.get_final_angle(lane_coords, clane_angles, initial_pt, self.closest_node_coords.unsqueeze(2), self.closest_reg_out.unsqueeze(2))
        final_pt = torch.cat([self.closest_node_coords, final_angle], dim=2)
        final_pt += self.closest_reg_out  # += predicted residuals

        pts = torch.stack([initial_pt.squeeze(1), final_pt.squeeze(0)], dim=1)
        intersections = data_utils.batch_intersect(pts, heading_threshold=5)  # is computing a check for correctness, remove for deployment
        return pts, intersections, final_angle.squeeze(2)


class PCurveNet(pl.LightningModule):
    def __init__(self, init_features=18, hidden=64, pts_per_pl=64, sec_history=2, sec_future=3, total_steps=1000):
        super().__init__()
        self.count = 0
        self.offset = 2
        self.pts_per_pl = pts_per_pl
        self.hidden = hidden
        self.sec_future = sec_future
        self.total_steps = total_steps
        self.save_hyperparameters()

        self.subgraph_net = SubGraphNet(init_features, hidden)
        self.globalgraph_net = GlobalGraphNet(in_features=2*hidden, out_features=2*hidden)
        self.cls_head = MLP(in_features=4*hidden, hidden=hidden, out_features=4)
        self.traj_head = MLP(in_features=4*hidden, hidden=hidden, out_features=60)

        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.smoothl1_loss = torch.nn.SmoothL1Loss()

    def forward(self, x, num_lanes, num_agents):
        gpls, lpls = self.subgraph_net(x)
        queries = gpls[:, -(num_lanes + num_agents):, :]  # ped crossings not passed back, so no query
        gpls = self.globalgraph_net(queries, gpls)
        # gpls = self.globalgraph_net(gpls, query_lastn=(num_lanes + num_agents))

        lpls = lpls[:, -(num_lanes + num_agents):, :, :]  # no predictions on ped crossings
        lpls += gpls.unsqueeze(2)

        # expand (repeat without memory allocation) global agent features, for each lane pl, for each node per lane pl
            # thus trajectories are predicted on each polyline node, for each agent
        apls = lpls[:, -num_agents:, None, :, :].expand(-1, -1, num_lanes, -1, -1)
        lpls = lpls[:, None, :num_lanes, :, :].expand(-1, num_agents, -1, -1, -1)

        heads_in = torch.cat([lpls, apls], dim=-1)

        cls_out = self.cls_head(heads_in)
        traj_out = self.traj_head(heads_in)
        return cls_out, traj_out

    def training_step(self, batch, batch_idx, train=True):
        torch.cuda.empty_cache()
        # batch, total num pls, 64, 18
        input_pls, cls_labels, gt_trajectories, num_lane_pls = batch
        num_agents = cls_labels.shape[1]
        num_lanes = num_lane_pls.item()

        cls_out, traj_out = self.forward(input_pls, num_lanes, num_agents)
        
        # cls_label (1, num_agents, 4 (index, x, y, angle))
        closest_node = cls_labels[:, :, 0].long()

        reg_label = cls_labels[:, :, 1:]
        reg_label[:, :, 2] = gt_trajectories[:, :, -1, 2]  # replace lane angle residual with gt_final angle
        reg_label[:, :, 2][reg_label[:, :, 2] < 0] += 2 * np.pi

        train_step = TrainStep(input_pls, closest_node, cls_labels, reg_label, gt_trajectories, cls_out, traj_out, num_agents, num_lanes, self.pts_per_pl, self.sec_future, self.offset)
        (positive_loss, alternate_loss, negative_loss), reg_loss, traj_loss = train_step.get_losses(self.bce_loss, self.smoothl1_loss)
        # cls_loss, reg_loss, traj_loss = train_step.get_losses(self.bce_loss, self.smoothl1_loss)
        
        cls_loss = positive_loss + negative_loss + 0.05 * alternate_loss
        alpha, beta = 0.5, 0.5
        loss = .5 * cls_loss + alpha * reg_loss + beta * traj_loss

        if train:
            self.log('losses', {'cls': .5 * cls_loss, 'reg': alpha * reg_loss, 'traj': beta * traj_loss})
            self.log('cls_loss', cls_loss)
            self.log('cls_logits', {'min': cls_out[..., 0].min(), 'max': cls_out[..., 0].max()})
            # self.log('cls_l', {'positive': positive_loss, 'alternate': alternate_loss, 'negative': negative_loss})

        return loss
    
    def training_step_end(self, outputs):
        lr = self.lr_schedulers().get_last_lr()[0]
        # lr = self.optimizers().param_groups[0]['lr']
        self.log('learning rate', lr)

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, train=False)
        self.log('validation loss', loss)

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, train=False)
        self.log('validation loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([{'params': self.subgraph_net.parameters()}, 
                                       {'params': self.globalgraph_net.parameters()},
                                       {'params': self.cls_head.parameters(), 'lr': 5e-4}, 
                                       {'params': self.traj_head.parameters()}], lr=1e-3, weight_decay=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=.75, end_factor=0.5, total_iters=self.trainer.estimated_stepping_batches)  #  self.total_steps
        return {"optimizer" : optimizer, "lr_scheduler" : {
                "scheduler" : lr_scheduler,
                "interval" : "step",
                "frequency" : 1
            }
        }
        # return optimizer
        # opt2 = torch.optim.AdamW(self.cls_head.parameters(), lr=5e-4, weight_decay=1e-6)
        # lr2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, factor=.2, patience=10, cooldown=40, min_lr=1e-4)
        # return [optimizer, opt2], [{'scheduler':lr_scheduler, 'interval':'step'}, {'scheduler':lr2, 'monitor':'cls_loss', 'interval':'step'}]


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", default="/home/further/argoverse", help="Directory for data containing train, val and test splits.")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size, if provided, overrides default which is 1.")
    parser.add_argument("--epochs", default=5, type=int, help="Number of training epochs.")
    parser.add_argument("--ckpts_per_epoch", default=5, type=int, help="Number of model checkpoints saved per epoch.")
    parser.add_argument("--pts_per_pl", default=64, type=int, help="Number of interpolated points per polyline node. Default is 64.")
    return parser.parse_args()


def main():
    args = parse()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # torch.cuda.empty_cache()

    batch_size = args.batch_size
    epochs = args.epochs
    ckpts_per_epoch = args.ckpts_per_epoch
    pts_per_pl = args.pts_per_pl
    train_data = AV2(args.data_root_dir, 'train', pts_per_pl=64, sec_history=2, sec_future=3)
    validation_data = AV2(args.data_root_dir, 'val', pts_per_pl=64, sec_history=2, sec_future=3)

    weights = torch.load('turn_example_weights.pt')
    ## weights give sum of turn example weights == sum of non turn example weights. This gives them equal frequency
    ## to calculate a lower frequency, use (1 + fraction) * x = 1 (x is the resulting frequency of non-turn examples)
    ## example to make the frequency of turn examples 1/10
        ## frequency of turn examples = x = 9/10. Fraction = 1/9. Therefore, divide weights by 9.
    weights[weights > 1] /= 9
    train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=os.cpu_count(),
        sampler=torch.utils.data.WeightedRandomSampler(weights, len(train_data), replacement=True),
    )
    validation_dataloader = DataLoader(
        validation_data, 
        batch_size=batch_size,
        num_workers=os.cpu_count(),
    )

    train_frac = 1.0
    steps_per_epoch = ((len(train_data) * train_frac) / batch_size) / 32
    ckpt_callback = ModelCheckpoint(
        dirpath="pcurve_checkpoints/",
        save_last=True,
        # monitor='cls_loss',
        every_n_train_steps=int(steps_per_epoch / ckpts_per_epoch)
    )
    # swa_callback = StochasticWeightAveraging(
    #     [0.25e-3, 0.25e-3, 0.125e-3, 0.125e-3],
    #     swa_epoch_start=0.8,  # fraction of epochs at which to start swa
    #     annealing_epochs=1,  # number of epochs per annealing phase (i think)
    #     annealing_strategy='cos',
    #     device='cpu'  # loc to store
    # )

    total_steps = steps_per_epoch * epochs
    pcurvenet = PCurveNet(init_features=21, hidden=64, pts_per_pl=pts_per_pl, sec_history=2, sec_future=3, total_steps=total_steps)
    trainer =  pl.Trainer(
        accelerator='gpu',
        auto_select_gpus=True,
        max_epochs=epochs,
        limit_train_batches=train_frac,
        limit_val_batches=0.2,
        # overfit_batches=1,
        # profiler="simple", 
        callbacks=[ckpt_callback],
        default_root_dir='pcurve_logs/',
        log_every_n_steps=8,
        accumulate_grad_batches=32,
    )
    trainer.fit(pcurvenet, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader, ckpt_path='pcurve_checkpoints/last.ckpt')
    # 
    # ckpt_path='pcurve_checkpoints/epoch=15-step=29946.ckpt'
    # trainer.validate(pcurvenet, dataloaders=validation_dataloader, ckpt_path='pcurve_checkpoints/last.ckpt')
    # REMOVE BATCH INTERSECT ASSERTIONS


'''
init_features = 21 (used to be 18)
velocity - 13:15 (used to be 11:13) (v0 used in training, validate and demo functions)

cls labels updated to be in semantically correct lane (lane where lane angle matches final vehicle heading)
input_pls updated with lane angle and curvature and agent headings

maybe scale alt loss, traj loss

changed offset to 2, 
IMPLEMENT backward instead of forward diff for lane angles by using clane angles instead
updated final angle calculation
closest5 is now based on closest to selected node + offset
closest10 now
agent stationary threshold

'''
def compute_ade(predictions, labels):
    displacement_errors = torch.linalg.norm(labels - predictions, dim=-1)
    return displacement_errors

import metrics

def fps():
    args = parse()

    validation_data = AV2(args.data_root_dir, 'val', pts_per_pl=64, sec_history=2, sec_future=3)
    idx = 12
    metrics.pcurve_fps(validation_data, idx, 'cuda', iters=1000)

def iar():
    metrics.pcurve_iar()


def validate():
    args = parse()

    validation_data = AV2(args.data_root_dir, 'val', pts_per_pl=64, sec_history=2, sec_future=3)

    data_samples = torch.randint(0, 24988, size=(100,))
    # inputs = [validation_data[i] for i in data_samples]
    # inputs = [validation_data.process_parallel_frenet(i) for i in torch.randint(0, 24988, size=(500,))]

    pcurvenet = PCurveNet(init_features=21)
    pcurvenet = pcurvenet.load_from_checkpoint("pcurve_checkpoints/last.ckpt")  # epoch=14-step=32760
    pcurvenet.eval()

    ade = []
    for idx in data_samples:
        forward = ForwardPass(validation_data, idx, topn=6)
        with torch.inference_mode():
            pts = forward.forward(pcurvenet)
        
        gt = forward.gt_trajectories[0, :, None, 1:, :2]
        ade.append(compute_ade(pts.view(forward.num_agents, forward.topn, 30, 2), gt))

    pcurve_de = torch.cat(ade, dim=0)  # total # agents, topn, timesteps
    min_traj = pcurve_de[:, :, -1].argmin(dim=-1).unsqueeze(1)
    pade = pcurve_de.mean(dim=2).take_along_dim(min_traj, dim=-1).mean()
    pade2 = pcurve_de.mean(dim=2).min(dim=-1)[0].mean()
    print('ade_minfde:', pade)
    print('ade_minade:', pade2)
    print(metrics.mr(pcurve_de))
    return

    pvec_de = torch.load('parallel_vectornet_de.pt')
    vec_de = torch.load('sequential_vectornet_de.pt')
    lanegcn_de = torch.load('LaneGCN/LaneGCN/lanegcn_de.pt')

    horizons = torch.linspace(0.1, 3.0, 30)
    
    # ade for all three + based on min_ade and based on min_fde
    pcurve_ade_minade = adevt_minade(pcurve_de, horizons)
    lanegcn_ade_minade = adevt_minade(lanegcn_de, horizons)

    pcurve_ade_minfde = adevt_minfde(pcurve_de, horizons)
    lanegcn_ade_minfde = adevt_minfde(lanegcn_de, horizons)

    pvec_ade = adevt_k1(pvec_de, horizons)
    vec_ade = adevt_k1(vec_de, horizons)

    # pcurvenet + lanegcn, fde based on min_traj and min_traj ade
    pcurve_fde_minade = fdevt_minade(pcurve_de, horizons)
    lanegcn_fde_minade = fdevt_minade(lanegcn_de, horizons)

    pcurve_fde_minfde = fdevt_minfde(pcurve_de, horizons)
    lanegcn_fde_minfde = fdevt_minfde(lanegcn_de, horizons)

    # All three for plain fde
    pcurve_fde = fdevt_min(pcurve_de, horizons)
    lanegcn_fde = fdevt_min(lanegcn_de, horizons)
    pvec_fde = fdevt_k1(pvec_de, horizons)
    vec_fde = fdevt_k1(vec_de, horizons)

    # plt.plot(horizons, pcurve_fde, label='pcurve_fde')
    # plt.plot(horizons, lanegcn_fde, label='lanegcn_fde')
    # plt.plot(horizons, pvec_fde, label='pvec_fde')
    # plt.plot(horizons, vec_fde, label='vec_fde')
    # # plt.plot(horizons, pvec_ade, label='pvec_ade')
    # # plt.plot(horizons, vec_ade, label='vec_ade')
    # plt.legend(loc='upper left')
    # plt.xlabel('Time Horizon (s)')
    # plt.ylabel('Displacement Errors')
    # plt.title('Displacement Errors Over Increasing Time Horizons')
    # plt.savefig('FDE.png')

    
    # print(ades)


if __name__ == "__main__":
    main()

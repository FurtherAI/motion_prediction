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


class TrainStep:
    def __init__(self, input_pls, closest_node, reg_label, gt_trajectories, cls_out, traj_out, num_agents, num_lanes, pts_per_pl, sec_future, offset=4):
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

        reg_out = cls_out[..., 1:].view(-1, num_agents, num_lanes * self.pts_per_pl, 3)
        traj_out = traj_out.view(-1, num_agents, num_lanes * self.pts_per_pl, traj_out.shape[-1])

        self.closest_reg_out = reg_out.take_along_dim(closest_node.view(-1, num_agents, 1, 1), dim=2).squeeze(2)
        self.closest_traj_out = traj_out.take_along_dim(closest_node.view(-1, num_agents, 1, 1), dim=2).squeeze(2)
        del traj_out

        self.closest_node_coords = None

        self.batch_range = torch.arange(input_pls.shape[0], device='cuda')
        self.dist_threshold = 3  # dist threshold for negative/positive node mask

    def get_losses(self, bce_loss, smoothl1_loss):
        with torch.no_grad():
            lane_coords = self.input_pls[:, -(self.num_lanes + self.num_agents):-self.num_agents, :, 2:4]  # xy coordinates for each lane node
            v0 = torch.linalg.norm(self.input_pls[:, -self.num_agents:, -1, 13:15], dim=-1).squeeze(0)

            lane_angles = lane_coords[:, :, self.offset:, :] - lane_coords[:, :, :-self.offset, :]
            lane_angles = torch.arctan2(lane_angles[..., 1], lane_angles[..., 0])
            lane_angles[lane_angles < 0] += 2 * np.pi  # shape (batch, num_lanes, 60)
            clane_angles = torch.cat([lane_angles, lane_angles[:, :, -1:].expand(-1, -1, self.offset)], dim=2)  # (batch, num_lanes, 64)

            lane_coords = lane_coords.view(-1, self.num_lanes * self.pts_per_pl, 2)
            self.closest_node_coords = lane_coords[self.batch_range, self.closest_node]
            dist = torch.linalg.norm(self.closest_node_coords.unsqueeze(2) - lane_coords.unsqueeze(1), dim=-1)
            dist_mask = dist > self.dist_threshold  # all nodes more than thresh meters away from the positive node (closest to end position) are negative nodes
            # shape (batch, num_agents, num_lanes * pts_per_pl)

            pts, intersections, final_angle = self.get_pts(lane_coords, lane_angles)

            p0 = pts[:, 0, :2].unsqueeze(1)
            p1 = intersections.unsqueeze(1)
            p2 = pts[:, 1, :2].unsqueeze(1)
            timesteps_future = self.sec_future * 10

            ## generate frenet labels (relative to predicted bezier curve)
            # gt_trajectories contain last observed point (so skip this one) + headings for each point
            traj_labels = data_utils.get_frenet_labels(p0, p1, p2, v0, self.gt_trajectories[0, :, 1:, :2], timesteps_future, device='cuda', sample_precision=128)

            pos_pts, alt_mask = self.get_alt_goals(lane_coords, clane_angles, v0, dist_mask)

        cls_losses = self.get_cls_loss(bce_loss, pos_pts, alt_mask)

        # SELECT ANGLE ACCORDING TO CLOSEST5 + RESIDUAL, SELECT LABEL ACCORDING TO GT_TRAJ
        self.closest_reg_out[:, :, 2] += final_angle
        reg_loss = smoothl1_loss(self.closest_reg_out, self.reg_label)
        traj_loss = smoothl1_loss(self.closest_traj_out.view(-1, self.num_agents, timesteps_future, 2), traj_labels.unsqueeze(0))

        return cls_losses, reg_loss, traj_loss

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

        msk = matching_node < 3
        matching_node[~in_front] = torch.inf  # select correct points in front of agent, not behind. Also make sure they are reasonably close to goal distance
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

    def get_cls_loss(self, bce_loss, pos_pts, dist_mask):
        cls_logits = self.cls_out[..., 0].view(-1, self.num_agents, self.num_lanes * self.pts_per_pl)
        positive_samples = cls_logits.take_along_dim(self.closest_node.unsqueeze(2), dim=2).squeeze(2)
        alternate_samples = cls_logits.masked_select(pos_pts)
        negative_samples = cls_logits.masked_select(dist_mask)

        negative_loss = bce_loss(negative_samples, torch.zeros((1,), device='cuda').expand(negative_samples.shape[0]))  # no reduction, same shape for topk
        
        positive_loss = bce_loss(positive_samples, torch.ones_like(positive_samples)).mean()
        if alternate_samples.numel() != 0:
            alternate_loss = bce_loss(alternate_samples, torch.ones((1,), device='cuda').expand(alternate_samples.shape[0])).mean()
        else:
            alternate_loss = torch.tensor([0], dtype=torch.float32, device='cuda', requires_grad=False)
        negative_loss = negative_loss.mean()
        return positive_loss, alternate_loss, negative_loss

    def get_pts(self, lane_coords, lane_angles):
        initial_pt = self.gt_trajectories[0, :, 0, :]

        final_angle = self.get_final_angle(lane_coords, lane_angles, initial_pt)
        
        final_pt = torch.cat([self.closest_node_coords, final_angle.unsqueeze(2)], dim=2)
        final_pt += self.closest_reg_out  # += predicted residuals

        pts = torch.stack([initial_pt, final_pt.squeeze(0)], dim=1)
        intersections = data_utils.batch_intersect(pts, heading_threshold=5)  # is computing a check for correctness, remove for deployment
        return pts, intersections, final_angle

    def get_final_angle(self, lane_coords, lane_angles, initial_pt):
        closest5 = torch.linalg.norm(self.closest_node_coords.unsqueeze(2) - lane_coords.unsqueeze(1), dim=-1)
        closest5, idx_closest5 = closest5.topk(5, largest=False, dim=-1)
        closest_node = data_utils.unravel_idx(idx_closest5, (self.num_lanes, self.pts_per_pl))
        # final_angle = lane_angles[(batch_range, closest_node[0], closest_node[1].clamp(0, self.pts_per_pl - offset - 1))]

        ## MATCH FINAL ANGLE TO LANE ANGLE POINTING IN THE SAME DIRECTION (WITHIN 60 DEGREES) TO PREVENT SELECTING OPPOSING TRAFFIC LANE WHEN OVERLAPPING
        final_angle = lane_angles[(self.batch_range, closest_node[0], closest_node[1].clamp(0, 64 - self.offset - 1))]
        initial_angle = initial_pt[:, 2]
        initial_angle[initial_angle < 0] += 2 * np.pi
        angle_between = torch.abs(final_angle - initial_angle.view(1, self.num_agents, 1))
        angle_between = torch.minimum(angle_between, 2 * np.pi - angle_between)
        eps = 1e-5
        closest5 += eps
        closest5[angle_between <= (np.pi / 3)] = 0
        final_angle = final_angle.take_along_dim(closest5.argmin(dim=-1, keepdim=True), dim=-1).squeeze(-1)
        return final_angle


class PCurveNet(pl.LightningModule):
    def __init__(self, init_features=18, hidden=64, pts_per_pl=64, sec_history=2, sec_future=3, total_steps=1000):
        super().__init__()
        self.count = 0
        self.offset = 4
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
        # heads_in = torch.empty((lpls.shape[0], lpls.shape[1], lpls.shape[2], lpls.shape[3], 2 * lpls.shape[4]), dtype=torch.float32, device=lpls.device, requires_grad=False)
        # heads_in[..., :(2*self.hidden)] = lpls
        # heads_in[..., (2*self.hidden):] = apls

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

        train_step = TrainStep(input_pls, closest_node, reg_label, gt_trajectories, cls_out, traj_out, num_agents, num_lanes, self.pts_per_pl, self.sec_future, self.offset)
        (positive_loss, alternate_loss, negative_loss), reg_loss, traj_loss = train_step.get_losses(self.bce_loss, self.smoothl1_loss)
        
        cls_loss = positive_loss + negative_loss + 0.05 * alternate_loss
        alpha, beta = 0.5, 0.4
        loss = .5 * cls_loss + alpha * reg_loss + beta * traj_loss

        if train:
            self.log('losses', {'cls': .5 * cls_loss, 'reg': alpha * reg_loss, 'traj': beta * traj_loss})
            self.log('cls_loss', cls_loss)
            self.log('cls_logits', {'min': cls_out[..., 0].min(), 'max': cls_out[..., 0].max()})
            self.log('cls_l', {'positive': positive_loss, 'alternate': alternate_loss, 'negative': negative_loss})

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
                                       {'params': self.traj_head.parameters(), 'lr': 5e-4}], lr=1e-3, weight_decay=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=.75, end_factor=0.4, total_iters=self.trainer.estimated_stepping_batches)  #  self.total_steps
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
    torch.backends.cudnn.enabled = True
    torch.cuda.empty_cache()

    batch_size = args.batch_size
    epochs = args.epochs
    ckpts_per_epoch = args.ckpts_per_epoch
    pts_per_pl = args.pts_per_pl
    train_data = AV2(args.data_root_dir, 'train', pts_per_pl=64, sec_history=2, sec_future=3)
    validation_data = AV2(args.data_root_dir, 'val', pts_per_pl=64, sec_history=2, sec_future=3)

    # torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True)
    train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=os.cpu_count(), 
        # pin_memory=True
    )
    validation_dataloader = DataLoader(
        validation_data, 
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        # pin_memory=True
    )

    train_frac = 0.3
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
        limit_train_batches=train_frac,  # ~70,000 samples
        limit_val_batches=0.2,
        # move_metrics_to_cpu=True,  # might help with GPU memory at cost of speed, apparently internal logged metrics are recorded on the GPU
        # overfit_batches=1,
        # profiler="simple", 
        callbacks=[ckpt_callback],
        default_root_dir='pcurve_logs/',
        log_every_n_steps=8,
        accumulate_grad_batches=32,
    )

    trainer.fit(pcurvenet, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader, ckpt_path='pcurve_checkpoints/last-v2.ckpt')  # 
    # trainer.validate(pcurvenet, dataloaders=validation_dataloader, ckpt_path='pcurve_checkpoints/epoch=7-step=18720.ckpt')
    # REMOVE BATCH INTERSECT ASSERTIONS


'''
init_features = 21 (used to be 18)
velocity - 13:15 (used to be 11:13) (v0 used in training, validate and demo functions)

cls labels updated to be in semantically correct lane (lane where lane angle matches final vehicle heading)
input_pls updated with lane angle and curvature and agent headings

maybe scale alt loss, traj loss
find solution to dist threshold encouraging too many too close together. Need to pick from different goals. (could try selecting top 1 in each lane, top 6 among those?)
'''


def compute_ade(predictions, labels):
    displacement_erros = torch.linalg.norm(labels - predictions, dim=-1)
    return displacement_erros


def validate():
    args = parse()

    validation_data = AV2(args.data_root_dir, 'val', pts_per_pl=64, sec_history=2, sec_future=3)

    inputs = [validation_data[i] for i in torch.randint(0, 24988, size=(5000,))]
    # inputs = [validation_data.process_parallel_frenet(i) for i in torch.randint(0, 24988, size=(500,))]

    pcurvenet = PCurveNet(init_features=21)
    pcurvenet = pcurvenet.load_from_checkpoint("pcurve_checkpoints/last-v2.ckpt")  # epoch=14-step=32760
    pcurvenet.eval()

    ade = []
    for inp in inputs:
        input_pls, cls_labels, gt_trajectories, num_lane_pls = inp
        input_pls = torch.as_tensor(input_pls, dtype=torch.float32).unsqueeze(0)
        # input_pls = torch.cat([input_pls[..., :4], input_pls[..., 6:-6], input_pls[..., -5:]], dim=-1)
        cls_labels = torch.as_tensor(cls_labels, dtype=torch.float32).unsqueeze(0)
        gt_trajectories = torch.as_tensor(gt_trajectories, dtype=torch.float32).unsqueeze(0)
        num_lanes = num_lane_pls[0]
        num_agents = cls_labels.shape[1]

        with torch.inference_mode():
            cls_out, traj_out = pcurvenet(input_pls, num_lanes, num_agents)
            closest_node = cls_out[..., 0].long().view(-1, num_agents, num_lanes * 64)
            # closest_node = closest_node.argmax(dim=-1)
            topn = 6
            vals, closest_node = closest_node.topk(topn, dim=-1)
            # closest_node = closest_node.view(1, -1)
            # closest_node = cls_labels[:, :, 0].long()

            closest_reg_out = cls_out[..., 1:].view(-1, num_agents, 1, num_lanes * 64, 3)
            try:
                closest_reg_out = closest_reg_out.take_along_dim(closest_node.view(1, num_agents, topn, 1, 1), dim=3).squeeze(3)
            except RuntimeError:
                print(1)
                continue

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
            s = torch.linspace(0, 1, steps=(128 + 1), device='cpu')  # could increase precision for inference
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
            timesteps = torch.linspace(0.1, sec_future, timesteps_future, device='cpu')  # curve is between last observed point and end point/first predicted point is at time 0.1
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
        
        gt = gt_trajectories[0, :, None, 1:, :2]
        ade.append(compute_ade(pts.view(num_agents, topn, 30, 2), gt))

    ade = torch.cat(ade, dim=0)  # total # agents, topn, timesteps
    print('ADE:', ade.mean(dim=2).min(dim=-1)[0].mean())

    # horizons = torch.linspace(0.1, 3.0, 30)
    # ades = torch.zeros_like(horizons)
    # # for t in range(len(horizons)):
    # #     ades[t] = ade[:, :, :(t + 1)].mean(dim=2).min(dim=-1)[0].mean()
    # min_traj = ade[:, :, -1].argmin(dim=-1).unsqueeze(1)
    # for t in range(len(horizons)):
    #     ades[t] = ade[:, :, t].take_along_dim(min_traj, dim=1).mean()
    
    # print(ades)


if __name__ == "__main__":
    validate()

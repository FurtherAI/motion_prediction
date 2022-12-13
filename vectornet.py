from pathlib import Path
import os
import argparse
import torch as torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from subgraph_net import SubGraphNet
from globgraph_net import GlobalGraphNet
from decoder import Decoder
from dataset import AV2

from av2.datasets.motion_forecasting.eval import metrics
import pcurvenet


class VectorNet(pl.LightningModule):
    def __init__(self, init_features=14, hidden=64, pts_per_pl=64, sec_history=2, sec_future=3, total_steps=1000):
        super().__init__()
        self.batch_size = 4
        self.save_hyperparameters()
        self.total_steps = total_steps

        self.subgraph_net = SubGraphNet(init_features, hidden)
        self.globalgraph_net = GlobalGraphNet(in_features = 2 * hidden, out_features = 2 * hidden)
        self.decoder_net = Decoder(in_features = 2 * hidden, out_features=(2 * sec_future * 10))

        self.loss = torch.nn.MSELoss()

    def forward(self, x, num_agents=1):
        pls, _ = self.subgraph_net(x)
        agent_pl = self.globalgraph_net(pls[:, -num_agents:, :], pls)
        trajectories = self.decoder_net(agent_pl)
        return trajectories

    def training_step(self, batch, batch_idx):
        input_pls, label = batch
        input_pls = torch.cat([input_pls[:, :, :, :6], input_pls[:, :, :, 10:]], dim=3)
        trajectories = self.forward(input_pls, num_agents=label.shape[1])  # , num_agents=label.shape[1]
        
        loss = self.loss(trajectories.view(input_pls.shape[0], label.shape[1], -1, 2), label)  # , label.shape[1]
        self.log('training loss', loss)
        return loss
    
    def training_step_end(self, outputs):
        lr = self.lr_schedulers().get_last_lr()[0]
        self.log('learning rate', lr)

    def test_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        input_pls, label = batch
        input_pls = torch.cat([input_pls[:, :, :, :6], input_pls[:, :, :, 10:]], dim=3)
        trajectories = self.forward(input_pls, num_agents=label.shape[1])  # , num_agents=label.shape[1]
        
        loss = self.loss(trajectories.view(input_pls.shape[0], label.shape[1], -1, 2), label)  # , label.shape[1]
        self.log('validation loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.total_steps)
        return {"optimizer" : optimizer, "lr_scheduler" : {
                "scheduler" : lr_scheduler,
                "interval" : "step",
                "frequency" : 1
            }
        }


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", default="/home/further/argoverse", help="Directory for data containing train, val and test splits.")
    parser.add_argument("--batch_size", default=12, type=int, help="Batch size, if provided, overrides default which is 12.")
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

    train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        # collate_fn=train_data.collate_fn, 
        num_workers=os.cpu_count(), 
        # pin_memory=True
    )
    validation_dataloader = DataLoader(
        validation_data, 
        batch_size=batch_size,
        # collate_fn=train_data.collate_fn, 
        num_workers=os.cpu_count()
    )

    steps_per_epoch = len(train_data) / batch_size
    ckpt_callback = ModelCheckpoint(
        dirpath="parallel_checkpoints2/",
        every_n_train_steps=int(steps_per_epoch / ckpts_per_epoch)
    )

    total_steps = steps_per_epoch * epochs
    vectornet = VectorNet(init_features=14, hidden=64, pts_per_pl=pts_per_pl, sec_history=2, sec_future=3, total_steps=total_steps)
    trainer =  pl.Trainer(
        accelerator='gpu',
        auto_select_gpus=True,
        max_epochs=epochs,
        # overfit_batches=10, 
        # profiler="simple", 
        callbacks=[ckpt_callback]
    ) # auto_scale_batch_size=True
    trainer.fit(vectornet, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
'''
to train sequential version
change in model: num_agents selection, add or stay without unsqueeze(1). In train and val steps
change the call to view when calculating loss (don't need label.shape[1]). In train and val steps
dataset: change getitem back to normal, change len back to normal
set collate_fn, shuffle in dataloaders
use val dataloader
init features = 14

CHANGED JOIN_PLS


lane polylines
heading, select track features
join_pls order
changed/fixed get_label
'''

def rollout_trajectories(trajectories, dim=1):
    # shape - (batch_size, 30, 2)
    return trajectories.cumsum(dim=dim)


def compute_ade(predictions, labels):
    displacement_erros = torch.linalg.norm(labels - predictions, dim=-1)
    ade = displacement_erros.mean()
    return ade


def validate():
    args = parse()

    validation_data = AV2(args.data_root_dir, 'val', pts_per_pl=64, sec_history=2, sec_future=3)

    vectornet = VectorNet()
    vectornet = vectornet.load_from_checkpoint("checkpoints2/epoch=0-step=44982.ckpt")
    vectornet.eval()

    # inputs = [validation_data[i] for i in range(1000)]  # torch.randint(0, 10000, size=(1000,))
    inputs = [validation_data.process_item(i) for i in range(10000)]
    with torch.inference_mode():
        ade = 0
        ades = []
        for input_pls, labels in inputs:
            input_pls = torch.tensor(input_pls)
            labels = torch.tensor(labels)
            input_pls = input_pls.unsqueeze(0)
            labels = labels.unsqueeze(0)
            # input_pls = torch.cat([input_pls[:, :, :, :6], input_pls[:, :, :, 10:]], dim=3)
            # predictions = vectornet(input_pls, num_agents=labels.shape[0]).view(labels.shape[0], -1, 2)
            predictions = vectornet(input_pls).view(1, -1, 2)
            predictions = rollout_trajectories(predictions, dim=1)  # dim=1, below (labels), don't specify dim. Don't unsqueeze labels
            labels = rollout_trajectories(labels)

            # ade += compute_ade(predictions, labels)
            ades.append(pcurvenet.compute_ade(predictions, labels))
        ades = torch.cat(ades, dim=0)

    print(ades.shape)
    print(ades.mean())
    torch.save(ades, 'sequential_vectornet_de.pt')
    # horizons = torch.linspace(0.1, 3.0, 30)
    # ades_ = torch.zeros_like(horizons)
    # for t in range(len(horizons)):
    #     ades_[t] = ades[:, t].mean()
    
    # print(ades_)

    # input_pls, labels = validation_data.collate_fn(inputs)
    # with torch.inference_mode():
    #     predictions = vectornet(input_pls).view(input_pls.shape[0], -1, 2)
    #     predictions = rollout_trajectories(predictions, dim=1)
    #     labels = rollout_trajectories(labels)
    #     ade = compute_ade(predictions, labels)

    # print('ADE:', ade)

if __name__ == "__main__":
    validate()

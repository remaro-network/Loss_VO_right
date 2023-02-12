import argparse
import collections
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils import seed_rng
from utils.loadconfig import ConfigLoader
from trainer.deepvo_trainer import DeepVOTrainer

def main(config, options=()):
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    # lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # trainer = DeepVOTrainer(model, loss, metrics, optimizer,
    #                          config=config,
    #                          data_loader=data_loader,
    #                          valid_data_loader=valid_data_loader,
    #                          lr_scheduler=lr_scheduler,
    #                          options=options)
    trainer = DeepVOTrainer(data_loader = config.data_loader, model_args = config.model.args)

    trainer.train()


if __name__ == '__main__':                     
    config_loader = ConfigLoader()
    cfg = config_loader.merge_cfg('configs/train/deepvo/mimir.yml')

    print(cfg)
    main(cfg)
from utils.loadconfig import ConfigLoader
from trainer.deepvo_trainer import DeepVOTrainer

def main(config, options=()):
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    # lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = DeepVOTrainer(config = config)

    trainer.train()


if __name__ == '__main__':                     
    config_loader = ConfigLoader()
    cfg = config_loader.merge_cfg('configs/train/deepvo_se3/original_paper.yml')

    print(cfg)
    main(cfg)
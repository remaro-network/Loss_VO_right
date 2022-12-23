import torch
from trainer.trainer import Trainer

def to(data, device):
    if isinstance(data, dict):
        return {k: to(data[k], device) for k in data.keys()}
    elif isinstance(data, list):
        return [to(v, device) for v in data]
    else:
        return data.to(device)

class ReleVOTrainer(Trainer):
    def __init__(self, model, loss, metrics, optimizer, config, data_loader, valid_data_loader=None, lr_scheduler=None, options=None):
        super().__init__(model, loss, metrics, optimizer, config, data_loader, valid_data_loader, lr_scheduler, options)
    
    def _train_epoch(self,epoch):
        ''' train logic per epoch 
        '''
        total_loss = 0
        total_loss_dict = {}
        total_metrics = np.zeros(len(self.metrics))
        self.model.train()
        for batch_idx, data in enumerate(self.data_loader):
            # Every data instance is a pair of input data + target result
            data = to(data, self.device)
            
            # Gradients must be zeroed for every batch
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(data)

            # Compute the loss and its gradients
            loss_dict = self.loss(outputs)
            loss_dict = map_fn(loss_dict, torch.mean) # if loss dict, average losses
            loss = loss_dict["loss"]
    
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            loss_dict = map_fn(loss_dict, torch.detach)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            self.writer.add_scalar('loss', loss.item())
            for loss_component, v in loss_dict.items():
                self.writer.add_scalar(f"loss_{loss_component}", v.item())

            total_loss += loss.item()
            total_loss_dict = operator_on_dict(total_loss_dict, loss_dict, operator.add)
            metrics, valid = self._eval_metrics(outputs, training=True)
            total_metrics += metrics

            if self.writer.step % self.log_step == 0:
                img_count = min(outputs["keyframe"].shape[0], 8)

                self.writer.add_image('input', make_grid(data['keyframe'][:img_count].cpu(), nrow=4, normalize=True))
                self.writer.add_image('target', make_grid(data['target'][:img_count].cpu(), nrow=4, normalize=True))
                self.writer.add_image('output', make_grid(outputs['keyframe'][:img_count].cpu(), nrow=4, normalize=True))

                for i, metric in enumerate(self.metrics):
                    self.writer.add_scalar(f'{metric.__name__}', metrics[i])

            if self.verbosity >= 2
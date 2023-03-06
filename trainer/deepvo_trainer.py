import torch
import numpy as np
import os
import operator
from utils.util import map_fn, operator_on_dict
from model.deepvo.deepvo_model import DeepVOModel
from data_loader.data_loaders import SingleDataset, MultiDataset
from torch.utils.data import DataLoader
from trainer.trainer import Trainer

def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(data[k], device) for k in data.keys()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    else:
        return data.to(device)

class DeepVOTrainer(object):
	# def __init__(self, model, loss, metrics, optimizer, config, data_loader, valid_data_loader=None, lr_scheduler=None, options=...):
	# 	super().__init__(model, loss, metrics, optimizer, config, data_loader, valid_data_loader, lr_scheduler, options)
	def __init__(self, data_loader, model_args, config):
		# super().__init__(**kwargs)
		self.device = config.n_gpu

		self.model = DeepVOModel(batchNorm=model_args.batchNorm,checkpoint_location=model_args.checkpoint_location,
									conv_dropout=model_args.conv_dropout, image_size = config.model.args.image_size, rnn_hidden_size=model_args.rnn_hidden_size,
									rnn_dropout_out=model_args.rnn_dropout_out,rnn_dropout_between=model_args.rnn_dropout_between)
		self.model.to(self.device)
		self.trainer_args = config.trainer
		self.metrics = self.trainer_args.metrics

		self.optimizer = getattr(torch.optim,config.optimizer['type'])(self.model.parameters(),**config.optimizer.args)

		self.device = config.n_gpu
        
		cfg_dirs = [os.path.join(os.getcwd(),"configs","data_loader","MIMIR", test_sequence+".yml") for test_sequence in data_loader.dataset_dirs]

		self.data_loader = DataLoader(MultiDataset(cfg_dirs),batch_size=1, shuffle=False, num_workers=0, drop_last=True)
	
	def train(self):
		'''
		Full training logic
		'''
		not_improved_count = 0
		for epoch in range(self.trainer_args.epochs):
			result = self.train_epoch(epoch)

	def train_epoch(self,epoch):
		''' train logic per epoch 
		'''
		total_loss = 0
		total_loss_dict = {}
		total_metrics = np.zeros(len(self.metrics))		

		for batch_idx, data in enumerate(self.data_loader):
			# Every data instance is a pair of input data + target result
			data = to_device(data, self.device)
			
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

				self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Loss_dict: {}'.format(
					epoch,
					self._progress(batch_idx),
					loss.item(),
					loss_dict))

			if batch_idx == self.len_epoch:
				break

		log = {
			'loss': total_loss / self.len_epoch,
			'metrics': (total_metrics / self.len_epoch).tolist()
		}
		for loss_component, v in total_loss_dict.items():
			log[f"loss_{loss_component}"] = v.item() / self.len_epoch

		if self.do_validation:
			val_log = self._valid_epoch(epoch)
			log.update(val_log)

		if self.lr_scheduler is not None:
			self.lr_scheduler.step()

		return log

	def _valid_epoch(self, epoch):
		total_val_loss = 0
		total_val_loss_dict = {}
		total_val_metrics = np.zeros(len(self.metrics))
		self.model.eval()
		with torch.no_grad():
			for batch_idx, data in enumerate(self.valid_data_loader):
				# Every data instance is a pair of input data + target result
				data = to_device(data, self.device)

				# Make predictions for this batch
				outputs = self.model(data)

				# Compute the loss and its gradients
				loss_dict = self.loss(outputs)
				loss_dict = map_fn(loss_dict, torch.mean) # if loss dict, average losses
				loss = loss_dict["loss"]


				self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
				if not self.val_avg:
					self.writer.add_scalar('loss', loss.item())
					for loss_component, v in loss_dict.items():
						self.writer.add_scalar(f"loss_{loss_component}", v.item())

				total_val_loss += loss.item()
				total_val_loss_dict = operator_on_dict(total_val_loss_dict, loss_dict, operator.add)
				metrics, valid = self._eval_metrics(outputs, training=True)
				total_val_metrics += metrics

		if self.val_avg:
			len_val = len(self.valid_data_loader)
			self.writer.add_scalar('loss', total_val_loss / len_val)
			for i, metric in enumerate(self.metrics):
				self.writer.add_scalar('{}'.format(metric.__name__), total_val_metrics[i] / len_val)
			for loss_component, v in total_val_loss_dict.items():
				self.writer.add_scalar(f"loss_{loss_component}", v.item() / len_val)


		result = {
			'val_loss': total_val_loss / len(self.valid_data_loader),
			'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
		}

		for loss_component, v in total_val_loss_dict.items():
			result[f"val_loss_{loss_component}"] = v.item() / len(self.valid_data_loader)

		return result
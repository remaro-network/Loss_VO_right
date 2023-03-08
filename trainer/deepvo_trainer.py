import torch
import numpy as np
import os
import operator
from utils.util import map_fn, operator_on_dict
from model.loss_functions import pose_losses
from model.deepvo.deepvo_model import DeepVOModel
from data_loader.data_loaders import SingleDataset, MultiDataset
from torch.utils.data import DataLoader
from trainer.trainer import Trainer
from base.logger import VOLogger

def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(data[k], device) for k in data.keys()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    else:
        return data.to(device)

class DeepVOTrainer(object):
	def __init__(self, data_loader, model_args, config):

		self.device = config.n_gpu
		# Model setup
		self.model = DeepVOModel(batchNorm=model_args.batchNorm,checkpoint_location=model_args.checkpoint_location,
									conv_dropout=model_args.conv_dropout, image_size = config.model.args.image_size, rnn_hidden_size=model_args.rnn_hidden_size,
									rnn_dropout_out=model_args.rnn_dropout_out,rnn_dropout_between=model_args.rnn_dropout_between)
		self.model.to(self.device)
		# Loss config
		self.loss_cfg = config.loss
		# Trainer config
		self.trainer_args = config.trainer
		self.metrics = self.trainer_args.metrics
		# Optimizer
		self.optimizer = getattr(torch.optim,config.optimizer['type'])(self.model.parameters(),**config.optimizer.args)
        # Writer
		self.writer = VOLogger(log_dir=config.log_dir, log_step=config.log_step)
		# Data loader
		cfg_dirs = [os.path.join(os.getcwd(),"configs","data_loader","MIMIR", test_sequence+".yml") for test_sequence in data_loader.dataset_dirs]
		self.data_loader = DataLoader(MultiDataset(cfg_dirs),batch_size=1, shuffle=False, num_workers=0, drop_last=True)

		self.len_epoch = self.trainer_args.epochs
		# Validation
		self.validation = config.validation.do_validation
		if self.validation:
			cfg_dirs_validation = [os.path.join(os.getcwd(),"configs","data_loader","MIMIR", test_sequence+".yml") for test_sequence in config.validation.sequences]
			self.validation_data_loader = DataLoader(MultiDataset(cfg_dirs_validation),batch_size=1, shuffle=False, num_workers=0, drop_last=True)
		# Model checkpoint saving
		self.checkpoint_dir = config.trainer.save_dir
		self.checkpoint_period = config.trainer.save_period

	def train(self):
		'''
		Full training logic
		'''
		not_improved_count = 0
		for epoch in range(self.len_epoch):
			result = self.train_epoch(epoch)
			if self.validation:
				self.validation_epoch(epoch)
			self.save_checkpoint(epoch)               

	def validation_epoch(self,epoch):
		total_val_loss = 0
		total_val_loss_dict = {}
		total_val_metrics = np.zeros(len(self.metrics))		
		val_lossfcn = getattr(pose_losses, self.loss_cfg)
		# Set validation mode
		self.model.eval()
		with torch.no_grad():
			for batch_idx, data in enumerate(self.validation_data_loader):
				# Every data instance is a pair of input data + target result
				data = to_device(data, self.device)

				# Make predictions for this batch
				outputs = self.model(data)

				# Compute the loss 
				loss_dict = val_lossfcn(outputs)
				loss_dict = map_fn(loss_dict, torch.mean) # if loss dict, average losses
				loss = loss_dict["loss"]

				loss_dict = map_fn(loss_dict, torch.detach)

				total_val_loss += loss.item()
				total_val_loss_dict = operator_on_dict(total_val_loss_dict, loss_dict, lambda x, y: x + y)
				self.writer.log_dictionary(total_val_loss_dict,len(self.validation_data_loader),batch_idx,epoch, 'validation')
				
				

				if batch_idx == self.len_epoch:
					break

		return 
	
	def train_epoch(self,epoch):
		''' 
		Train logic per epoch 
		'''
		total_loss = 0
		total_loss_dict = {}
		total_metrics = np.zeros(len(self.metrics))		
		lossfcn = getattr(pose_losses, self.loss_cfg)
		# set training mode
		self.model.train()
		for batch_idx, data in enumerate(self.data_loader):
			# Every data instance is a pair of input data + target result
			data = to_device(data, self.device)
			
			# Gradients must be zeroed for every batch
			self.optimizer.zero_grad()

			# Make predictions for this batch
			outputs = self.model(data)

			# Compute the loss and its gradients
			loss_dict = lossfcn(outputs)
			loss_dict = map_fn(loss_dict, torch.mean) # if loss dict, average losses
			loss = loss_dict["loss"]
   
			loss.backward()

			# Adjust learning weights
			self.optimizer.step()

			loss_dict = map_fn(loss_dict, torch.detach)

			total_loss += loss.item()
			total_loss_dict = operator_on_dict(total_loss_dict, loss_dict, lambda x, y: x + y)
			self.writer.log_dictionary(total_loss_dict,len(self.data_loader),batch_idx,epoch, 'train')

			# metrics, valid = self._eval_metrics(outputs, training=True)
			# total_metrics += metrics
			
			

			if batch_idx == self.len_epoch:
				break


		# if self.do_validation:
		# 	val_log = self._valid_epoch(epoch)
			# log.update(val_log)

		# if self.lr_scheduler is not None:
		# 	self.lr_scheduler.step()

	def save_checkpoint(self, epoch):
		if epoch % self.checkpoint_period:
			arch = type(self.model).__name__
			state = {
				'arch': arch,
				'epoch': epoch,
				'model_state_dict': self.model.state_dict(),
				'optimizer_state_dict': self.optimizer.state_dict()
			}
			filename = os.path.join(os.getcwd(),self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
			print('saving checkpoint in ',filename)
			torch.save(state,filename)




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
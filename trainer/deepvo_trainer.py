import torch
import numpy as np
import os
import operator
from tqdm import tqdm
from utils.util import map_fn, operator_on_dict
from model.loss_functions import pose_losses
from model.deepvo.deepvo_model import DeepVOModel
from data_loader.data_loaders import MultiDataset
from torch.utils.data import DataLoader
from base.logger import VOLogger
from data_loader import dataset_type_as_directory

def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(data[k], device) for k in data.keys()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    else:
        return data.to(device)

class DeepVOTrainer(object):
	def __init__(self, config):

		self.device = config.n_gpu
		# Model setup
		self.model = self.set_model(config)
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
		self.batch_size = config.data_loader.batch_size
		self.data_loader = self.set_data_loader(config.dataset,config.data_loader.sequences, self.batch_size, config.data_loader.shuffle, 
					  							config.data_loader.num_workers, True)

		self.len_epoch = self.trainer_args.epochs
		# Validation
		self.validation = config.validation.do_validation
		if self.validation:
			self.validation_data_loader = self.set_data_loader(config.dataset,config.validation.sequences, self.batch_size, 
						      								   config.validation.shuffle, config.validation.num_workers, True)
		# Model checkpoint saving
		self.checkpoint_dir = config.trainer.save_dir
		self.checkpoint_period = config.trainer.save_period

	def set_model(self,config):
		model = DeepVOModel(batchNorm=config.model.args.batchNorm,checkpoint_location=config.model.args.checkpoint_location,
									conv_dropout=config.model.args.conv_dropout, image_size = config.data_loader.target_image_size, rnn_hidden_size=config.model.args.rnn_hidden_size,
									rnn_dropout_out=config.model.args.rnn_dropout_out,rnn_dropout_between=config.model.args.rnn_dropout_between)
		return model.to(self.device)
	
	def set_data_loader(self, datasets,dataset_dirs, batch_size, shuffle, num_workers, drop_last):
		# assign dataset stype to config folder
		print(len(datasets))
		for dataset in datasets:
			print("dataset",dataset)
			# for sequence in dataset_dirs.get(dataset):
			# 	print(dataset,sequence)
				# t=os.path.join(os.getcwd(),"configs","data_loader",dataset_type_as_directory[dataset], 
				# 		sequence+".yml")
				# print(t)

		if "mimir" in datasets:
			cfg_dirs = [os.path.join(os.getcwd(),"configs","data_loader",dataset_type_as_directory[dataset], 
						test_sequence+".yml") for dataset in datasets for test_sequence in dataset_dirs.get(dataset) ]
		else:
			cfg_dirs = [os.path.join(os.getcwd(),"configs","data_loader",dataset_type_as_directory[dataset], 
						test_sequence,test_sequence+".yml") for dataset in datasets for test_sequence in dataset_dirs.get(dataset) ]
		data_loader = DataLoader(MultiDataset(cfg_dirs), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
		return data_loader

	def train(self):
		'''
		Full training logic
		'''
		not_improved_count = 0
		self.best_valid_loss=float('inf')
		is_best = False

		for epoch in range(1,self.len_epoch+1):
			train_log = self.train_epoch(epoch)
			if self.validation:
				val_log = self.validation_epoch(epoch)
				if val_log["loss"] < self.best_valid_loss:
					self.best_valid_loss = val_log["loss"]
					print(f"\nBest validation loss: {self.best_valid_loss}")
					print(f"\nSaving best model for epoch: {epoch+1}\n")
					is_best = True
					self.save_checkpoint(epoch,is_best)   
					is_best=False
					
			self.save_checkpoint(epoch,is_best)               

	def validation_epoch(self,epoch):
		total_val_loss = 0
		total_val_loss_dict = {}
		total_val_metrics = np.zeros(len(self.metrics))		
		val_lossfcn = getattr(pose_losses, self.loss_cfg)
		# Set validation mode
		self.model.eval()
		with torch.no_grad():
			for batch_idx, data in tqdm(enumerate(self.validation_data_loader),total=len(self.validation_data_loader)):
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
				
		# Get average loss per epoch
		log_loss = {}
		for loss_key, loss_value in total_val_loss_dict.items():
			log_loss[loss_key] = loss_value.item()/batch_idx
		self.writer.log_epoch(log_loss,len(self.data_loader),batch_idx,epoch, 'validation_epoch')
		return log_loss				

				# if batch_idx == self.len_epoch:
				# 	break
	
	def train_epoch(self,epoch):
		''' 
		Train logic per epoch 
		'''
		total_batch_loss = 0
		total_batch_loss_dict = {}
		total_metrics = np.zeros(len(self.metrics))		
		lossfcn = getattr(pose_losses, self.loss_cfg)
		# set training mode
		self.model.train()
		for batch_idx, data in tqdm(enumerate(self.data_loader),total=len(self.data_loader)):
			# Every data instance is a pair of input data + target result
			data = to_device(data, self.device)
			
			# Gradients must be zeroed for every batch
			self.optimizer.zero_grad()

			# Make predictions for this batch
			outputs = self.model(data)

			# Compute the loss and its gradients
			loss_dict = lossfcn(outputs)
			# loss_dict = map_fn(loss_dict, torch.mean) # if loss dict, average losses
			loss = loss_dict["loss"]
   
			loss.backward()
			# Adjust learning weights
			self.optimizer.step()

			loss_dict = map_fn(loss_dict, torch.detach)

			total_batch_loss += loss
			total_batch_loss_dict = operator_on_dict(total_batch_loss_dict, loss_dict, lambda x, y: x + y)
			self.writer.log_dictionary(total_batch_loss_dict,len(self.data_loader),batch_idx,epoch, 'train_batch')

		# Get average loss per epoch
		log_loss = {}
		for loss_key, loss_value in total_batch_loss_dict.items():
			log_loss[loss_key] = loss_value.item()/batch_idx

		self.writer.log_epoch(log_loss,len(self.data_loader),batch_idx,epoch, 'train_epoch')
		return log_loss
			# metrics, valid = self._eval_metrics(outputs, training=True)
			# total_metrics += metrics
			
			

			# if batch_idx == self.len_epoch:
			# 	break


		# if self.do_validation:
		# 	val_log = self._valid_epoch(epoch)
			# log.update(val_log)

		# if self.lr_scheduler is not None:
		# 	self.lr_scheduler.step()

	def save_checkpoint(self, epoch, is_best):
		arch = type(self.model).__name__
		state = {
			'arch': arch,
			'epoch': epoch,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict()
		}
		if is_best:
			filename = os.path.join(os.getcwd(),self.checkpoint_dir, 'best-checkpoint-epoch.pth')
		else: 
			filename = os.path.join(os.getcwd(),self.checkpoint_dir, 'checkpoint-last.pth')
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
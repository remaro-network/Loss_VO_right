
from torch.utils.tensorboard import SummaryWriter

class VOLogger():
    def __init__(self, log_dir = None, log_step = 100):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_step = log_step

    def log_epoch(self, loss_dict, data_size, batch_idx, epoch, dictionary_str):
        loss_dict = {dictionary_str+'_'+key : value for key, value in loss_dict.items()}
        for loss_name, loss_value in loss_dict.items():
                self.writer.add_scalar(loss_name,loss_value,epoch*data_size + batch_idx)
        return

    def log_dictionary(self, loss_dict, data_size, batch_idx, epoch, dictionary_str):
        loss_dict = {dictionary_str+'_'+key : value for key, value in loss_dict.items()}
        if batch_idx % self.log_step:
            for loss_name, loss_value in loss_dict.items():
                self.writer.add_scalar(loss_name,loss_value,epoch*data_size + batch_idx)
        return







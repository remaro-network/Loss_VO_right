import torch
import torch.nn as nn

# initialize the deep learning model for releVO with forward pass
class releVO_model(nn.Module):
    def __init__(self, checkpoint_location=None):
        super(releVO_model, self).__init__()

        # Checkpoint (if any)
        self.checkpoint_location = checkpoint_location
        if self.checkpoint_location is not None:
            pretrained_flownet = torch.load(checkpoint_location[0], map_location="cpu")
            current_state_dict = self.state_dict()
            update_state_dict = {}
            for k, v in pretrained_flownet["state_dict"].items():
                if k in current_state_dict.keys():
                    update_state_dict[k] = v
            current_state_dict.update(update_state_dict)
            self.load_state_dict(current_state_dict)

    def forward(self, data_dict):
        keyframe = data_dict["keyframe"]
        frames = data_dict["frames"]


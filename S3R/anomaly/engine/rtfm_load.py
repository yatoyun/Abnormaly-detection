from .rtfm_model import Model
import torch

def rtfm_model(device, path_trained_model):
    
    model = Model(2048, 32)
    model_ckpt = torch.load(path_trained_model)
    
    model.load_state_dict(model_ckpt)

    model = model.to(device)

    model.eval()

    return model

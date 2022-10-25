import sys

import torch

try:
    from torch2trt import TRTModule

    torch2trt_found = True
except ImportError:
    torch2trt_found = False


def load_exported_model(torch_script_path, trt_path):
    if torch_script_path is not None:
        device = torch.device('cpu')
        model = torch.jit.load(torch_script_path)
        model = model.to(device)
    elif trt_path is not None:
        if not torch2trt_found:
            print('"torch2trt" is not supported.')
            sys.exit()
        else:
            device = torch.device('cuda')
            model = TRTModule()
            model.load_state_dict(torch.load(trt_path))
    else:
        print('"torch_script_path" or "trt_path" is required.')
        sys.exit()

    return model, device

import sys

import torch

try:
    import torch_tensorrt

    torch_tensorrt_found = True
except ImportError:
    torch_tensorrt_found = False


def load_exported_model(torch_script_path, trt_path):
    if torch_script_path is not None:
        device = torch.device('cpu')
        model = torch.jit.load(torch_script_path)
        model = model.to(device)
    elif trt_path is not None:
        if not torch_tensorrt_found:
            print('"torch_tensorrt" is not supported.')
            sys.exit()
        else:
            device = torch.device('cuda')
            model = torch.jit.load(trt_path)
            model = model.to(device)
    else:
        print('"torch_script_path" or "trt_path" is required.')
        sys.exit()

    return model, device

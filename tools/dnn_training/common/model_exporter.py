import os

import torch

from common.modules import load_checkpoint

try:
    from torch2trt import torch2trt

    torch2trt_found = True
except ImportError:
    torch2trt_found = False


def export_model(model, model_checkpoint, x, output_dir, torch_script_filename, trt_filename, trt_fp16=False,
                 keys_to_remove=None):
    load_checkpoint(model, model_checkpoint, keys_to_remove=keys_to_remove)
    model.eval()

    _export_torch_script(model, x, output_dir, torch_script_filename)
    if torch2trt_found:
        _export_trt(model, x, output_dir, trt_filename, fp16_mode=trt_fp16)


def _export_torch_script(model, x, output_dir, filename):
    traced_model = torch.jit.trace(model, x, check_trace=False)
    traced_model.save(os.path.join(output_dir, filename))


def _export_trt(model, x, output_dir, filename, fp16_mode=False):
    device = torch.device('cuda')
    model = model.to(device)
    model_trt = torch2trt(model, [x.to(device)], fp16_mode=fp16_mode)
    torch.save(model_trt.state_dict(), os.path.join(output_dir, filename))

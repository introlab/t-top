import os

import torch

from common.modules import load_checkpoint

from pathlib import Path

import signal
import sys
from contextlib import contextmanager

try:
    from torch2trt import torch2trt

    torch2trt_found = True
except ImportError:
    torch2trt_found = False

# Temporarily handle signals to remove files if interrupted during export
@contextmanager
def handle_sigs(func):
    signal.signal(signal.SIGINT, func)
    signal.signal(signal.SIGTERM, func)
    yield
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

def export_model(model, model_checkpoint, x, output_dir, torch_script_filename, trt_filename, trt_fp16=False,
                 keys_to_remove=None):
    ts_path: Path = Path(output_dir) / torch_script_filename
    trt_path: Path = Path(output_dir) / trt_filename

    load_checkpoint(model, model_checkpoint, keys_to_remove=keys_to_remove)
    model.eval()

    # Remove files if interrupted during export
    def rm_ts(sig, frame):
        ts_path.unlink(missing_ok=True)
        sys.exit(-1)

    def rm_trt(sig, frame):
        trt_path.unlink(missing_ok=True)
        sys.exit(-1)

    with handle_sigs(rm_ts):
        _export_torch_script(model, x, output_dir, torch_script_filename)
    if torch2trt_found:
        with handle_sigs(rm_trt):
            _export_trt(model, x, output_dir, trt_filename, fp16_mode=trt_fp16)


def _export_torch_script(model, x, output_dir, filename):
    traced_model = torch.jit.trace(model, x, check_trace=False, strict=False)
    traced_model.save(os.path.join(output_dir, filename))


def _export_trt(model, x, output_dir, filename, fp16_mode=False):
    device = torch.device('cuda')
    model = model.to(device)
    model_trt = torch2trt(model, [x.to(device)], fp16_mode=fp16_mode)
    torch.save(model_trt.state_dict(), os.path.join(output_dir, filename))

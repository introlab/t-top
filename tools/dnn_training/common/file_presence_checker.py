from pathlib import Path
import os
import sys
import importlib

def terminate_if_already_exported(output_dir, torch_script_filename, trt_filename, force_export_if_exists):
    ts_path: Path = Path(output_dir) / torch_script_filename
    trt_path: Path = Path(output_dir) / trt_filename

    torch2trt_found = importlib.util.find_spec("torch2trt") is not None

    if torch2trt_found:
        if ts_path.exists() and trt_path.exists() and not force_export_if_exists:
            print(f'TorchScript ({torch_script_filename}) and TensorRT ({trt_filename}) models already exist, skipping export.')
            sys.exit(os.EX_OK)
    else:
        if ts_path.exists() and not force_export_if_exists:
            print(f'TorchScript ({torch_script_filename}) model already exists, skipping export.')
            sys.exit(os.EX_OK)

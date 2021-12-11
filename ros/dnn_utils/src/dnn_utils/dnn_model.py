import time

import torch

try:
    from torch2trt import TRTModule
    TORCH2TRT_FOUND = True
except ImportError:
    TORCH2TRT_FOUND = False

import rospkg


PACKAGE_PATH = rospkg.RosPack().get_path('dnn_utils')


class DnnModel:
    def __init__(self, torch_script_model_path, tensor_rt_model_path, sample_input, inference_type=None):
        self._device = None
        self._model = None
        self._model_latency = None

        if inference_type is None:
            self._assign_cpu_torch_script_model(torch_script_model_path, sample_input)
            self._replace_model_by_gpu_torch_script_model_if_faster(torch_script_model_path, sample_input)
            self._replace_model_by_tensor_rt_model_if_faster(tensor_rt_model_path, sample_input)
        elif inference_type == 'cpu':
            self._assign_cpu_torch_script_model(torch_script_model_path, sample_input)
        elif inference_type == 'torch_gpu':
            self._model_latency = float('inf')
            self._replace_model_by_gpu_torch_script_model_if_faster(torch_script_model_path, sample_input)
        elif inference_type == 'trt_gpu':
            self._model_latency = float('inf')
            self._replace_model_by_tensor_rt_model_if_faster(tensor_rt_model_path, sample_input)
        else:
            raise ValueError('Invalid inference_type')

        if self._model is None:
            raise ValueError('The selected inference_type is not supported.')

    def _assign_cpu_torch_script_model(self, torch_script_model_path, sample_input):
        self._device = torch.device('cpu')
        self._model = torch.jit.load(torch_script_model_path)
        self._model.eval()

        self._model(sample_input.to(self._device)) # JIT step

        start_time = time.time()
        self._model(sample_input.to(self._device))
        self._model_latency = time.time() - start_time

    def _replace_model_by_gpu_torch_script_model_if_faster(self, torch_script_model_path, sample_input):
        if not torch.cuda.is_available() or not _is_gpu_supported():
            return

        device = torch.device('cuda')
        model = torch.jit.load(torch_script_model_path).to(device)
        model.eval()

        model(sample_input.to(device)) # JIT step

        start_time = time.time()
        model(sample_input.to(device))
        model_latency = time.time() - start_time

        if model_latency < self._model_latency:
            self._device = device
            self._model = model
            self._model_latency = model_latency

    def _replace_model_by_tensor_rt_model_if_faster(self, tensor_rt_model_path, sample_input):
        if not torch.cuda.is_available() or not TORCH2TRT_FOUND:
            return

        device = torch.device('cuda')
        model = TRTModule()
        model.load_state_dict(torch.load(tensor_rt_model_path))

        start_time = time.time()
        model(sample_input.to(device))
        model_latency = time.time() - start_time

        if model_latency < self._model_latency:
            self._device = device
            self._model = model
            self._model_latency = model_latency

    def __call__(self, x):
        return self._model(x.to(self._device))

    def device(self):
        return self._device


def _is_gpu_supported():
    capability = torch.cuda.get_device_capability()
    return capability[0] >= 4 or capability[0] >= 3 and capability[1] >= 5

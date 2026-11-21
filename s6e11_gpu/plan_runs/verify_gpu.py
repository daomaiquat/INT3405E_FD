import argparse
import ctypes as ct
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np


class LightGBMLog:
    def __init__(self):
        self.devices = []

    def info(self, message):
        print(message, flush=True)
        if 'Using GPU Device:' in message:
            self.devices.append(message)

    def warning(self, message):
        print(message, flush=True)


def find_nvidia_opencl():
    cl = ct.WinDLL('OpenCL.dll')
    handle = ct.c_void_p
    uint = ct.c_uint
    cl.clGetPlatformIDs.argtypes = [uint, ct.POINTER(handle), ct.POINTER(uint)]
    cl.clGetDeviceIDs.argtypes = [handle, ct.c_ulonglong, uint, ct.POINTER(handle), ct.POINTER(uint)]
    for function in (cl.clGetPlatformInfo, cl.clGetDeviceInfo):
        function.argtypes = [handle, uint, ct.c_size_t, handle, ct.POINTER(ct.c_size_t)]

    def name(info, item, property_id):
        size = ct.c_size_t()
        assert info(item, property_id, 0, None, ct.byref(size)) == 0
        value = ct.create_string_buffer(size.value)
        assert info(item, property_id, size.value, value, None) == 0
        return value.value.decode()

    count = uint()
    assert cl.clGetPlatformIDs(0, None, ct.byref(count)) == 0
    platforms = (handle * count.value)()
    assert cl.clGetPlatformIDs(count.value, platforms, None) == 0
    nvidia = []
    for platform_id, platform in enumerate(platforms):
        platform_name = name(cl.clGetPlatformInfo, platform, 0x0902)
        result = cl.clGetDeviceIDs(platform, 0xFFFFFFFF, 0, None, ct.byref(count))
        if result == -1:
            continue
        assert result == 0
        devices = (handle * count.value)()
        assert cl.clGetDeviceIDs(platform, 0xFFFFFFFF, count.value, devices, None) == 0
        for device_id, device in enumerate(devices):
            device_name = name(cl.clGetDeviceInfo, device, 0x102B)
            vendor = name(cl.clGetDeviceInfo, device, 0x102C)
            print(f'OpenCL [{platform_id}, {device_id}]: {platform_name} / {vendor} / {device_name}', flush=True)
            if 'NVIDIA' in vendor and 'RTX 4060' in device_name:
                nvidia.append((platform_id, device_id))
    assert len(nvidia) == 1, nvidia
    return nvidia[0]


def check_boost():
    import lightgbm as lgb
    import xgboost as xgb

    rng = np.random.default_rng(1729)
    inputs = rng.normal(size=(2048, 8)).astype(np.float32)
    labels = (inputs[:, 0] + inputs[:, 1] > 0).astype(np.int32)
    print(f'XGBoost {xgb.__version__} CUDA build: {xgb.build_info()}', flush=True)
    data = xgb.QuantileDMatrix(inputs, label=labels)
    booster = xgb.train({'device': 'cuda:0', 'tree_method': 'hist', 'objective': 'binary:logistic', 'max_depth': 3}, data, num_boost_round=3)
    device = json.loads(booster.save_config())['learner']['generic_param']['device']
    assert device == 'cuda:0', device
    print(f'XGBoost trained on {device}', flush=True)

    platform_id, device_id = find_nvidia_opencl()
    print(f'LightGBM {lgb.__version__} selected platform={platform_id}, device={device_id}', flush=True)
    log = LightGBMLog()
    lgb.register_logger(log)
    params = {
        'objective': 'binary', 'device_type': 'gpu', 'gpu_platform_id': platform_id,
        'gpu_device_id': device_id, 'max_bin': 63, 'verbosity': 1,
    }
    model = lgb.train(params, lgb.Dataset(inputs, label=labels), num_boost_round=3)
    assert model.current_iteration() == 3
    assert len(log.devices) == 1 and 'RTX 4060' in log.devices[0] and 'NVIDIA Corporation' in log.devices[0], log.devices


def check_torch():
    import torch

    assert torch.cuda.is_available()
    name = torch.cuda.get_device_name(0)
    assert 'NVIDIA' in name and 'RTX 4060' in name, name
    inputs = torch.arange(4, dtype=torch.float32, device='cuda:0').reshape(2, 2)
    outputs = inputs @ inputs
    assert outputs.is_cuda
    torch.cuda.synchronize(0)
    values = outputs.tolist()
    assert values == [[2.0, 3.0], [6.0, 11.0]], values
    print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {name}: {values}', flush=True)


def check_notebook_kernel():
    from nbclient import NotebookClient
    from nbformat.v4 import new_code_cell, new_notebook

    notebook = new_notebook(cells=[new_code_cell('import sys; print(sys.executable)')])
    NotebookClient(notebook, kernel_name='python3', timeout=45).execute()
    executable = Path(notebook.cells[0].outputs[0].text.strip()).resolve()
    assert executable == Path(sys.executable).resolve(), executable
    print(f'Notebook executed with {executable}', flush=True)
    for package in ('numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'nbformat', 'nbclient', 'ipykernel'):
        print(f'{package} {version(package)}', flush=True)


if __name__ == '__main__':
    checks = {'boost': check_boost, 'torch': check_torch, 'notebook': check_notebook_kernel}
    parser = argparse.ArgumentParser()
    parser.add_argument('backend', choices=checks)
    checks[parser.parse_args().backend]()

import os

import torch


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        """
        Resolve and configure computation device for the experiment.

        Priority:
          1. Explicit args.device_type (cpu/cuda/mps).
          2. Legacy behaviour based on args.use_gpu.
        """
        device_type = getattr(self.args, 'device_type', None)
        if device_type is not None:
            return self._select_device_by_explicit_type(device_type)
        return self._select_device_legacy()

    def _select_device_by_explicit_type(self, device_type):
        """
        Select device when device_type is explicitly provided.

        Parameters
        ----------
        device_type : str
            Expected values: "cpu", "cuda", "mps".
        """
        normalized_type = str(device_type).strip().lower()

        if normalized_type == 'cpu':
            device = torch.device('cpu')
            print(f'Use device type: {normalized_type}')
            return device

        if normalized_type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA device requested, but torch.cuda.is_available() is False.')
            print(f'Use device type: {normalized_type}')
            return self._configure_cuda_device()

        if normalized_type == 'mps':
            mps_backend = getattr(torch.backends, 'mps', None)
            if mps_backend is None or not mps_backend.is_available():
                raise RuntimeError('MPS device requested, but torch.backends.mps.is_available() is False.')
            device = torch.device('mps')
            print(f'Use device type: {normalized_type}')
            return device

        raise ValueError(f'Unsupported device_type "{device_type}". Expected one of ["cpu", "cuda", "mps"].')

    def _select_device_legacy(self):
        """
        Legacy device selection that relies on args.use_gpu.

        This path is used when args.device_type is not provided.
        """
        use_gpu = bool(getattr(self.args, 'use_gpu', False))
        cuda_available = torch.cuda.is_available()

        if use_gpu and cuda_available:
            print('Use legacy CUDA configuration because --device_type is not set')
            return self._configure_cuda_device()

        device = torch.device('cpu')
        print('Use device type: cpu (legacy fallback)')
        return device

    def _configure_cuda_device(self):
        """
        Configure CUDA device (single- or multi-GPU) based on args.

        Uses:
          * args.gpu
          * args.use_multi_gpu
          * args.devices
        """
        use_multi_gpu = bool(getattr(self.args, 'use_multi_gpu', False))
        gpu_index = int(getattr(self.args, 'gpu', 0))

        if use_multi_gpu:
            devices_raw = getattr(self.args, 'devices', str(gpu_index))
            devices_clean = str(devices_raw).replace(' ', '')
            device_ids = [int(identifier) for identifier in devices_clean.split(',') if identifier]

            if not device_ids:
                raise ValueError('CUDA multi-GPU requested, but no valid device ids were provided.')

            self.args.device_ids = device_ids
            self.args.gpu = device_ids[0]
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(identifier) for identifier in device_ids)
            print(f'Use CUDA multi-GPU with device_ids={device_ids}')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
            self.args.gpu = gpu_index
            print(f'Use CUDA single GPU: cuda:{gpu_index}')

        device = torch.device(f'cuda:{self.args.gpu}')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
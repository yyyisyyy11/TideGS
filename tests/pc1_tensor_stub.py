"""Small NumPy tensor backend for executing PC1 transport/control behavior.

This is not a CUDA numerical oracle. Real autograd/NCCL equivalence is covered
by the opt-in CUDA suite; here collective buffers, packet order and API cuts
execute against arrays instead of being asserted through source strings.
"""

from contextlib import contextmanager, nullcontext
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np


class Device:
    type = 'cuda'
    index = 0

    def __str__(self):
        return 'cuda:0'


class Tensor:
    def __init__(self, values, requires_grad=False, source=None):
        self.values = np.asarray(values)
        self.requires_grad = requires_grad
        self.source = source
        self.grad = None
        self.grad_fn = None
        self.device = Device()
        self.recorded_streams = []

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def ndim(self):
        return self.values.ndim

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self.values, axis=dim))

    def detach(self):
        return Tensor(self.values)

    def requires_grad_(self, enabled):
        self.requires_grad = enabled
        return self

    def reshape(self, *shape):
        return Tensor(self.values.reshape(shape), self.requires_grad, self.source)

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.values), self.requires_grad, self.source)

    def new_empty(self, shape):
        return Tensor(np.empty(shape, dtype=self.dtype))

    def copy_(self, other):
        self.values[...] = other.values
        return self

    def split(self, widths, dim=0):
        return tuple(Tensor(value) for value in np.split(
            self.values, np.cumsum(widths)[:-1], axis=dim))

    def cumsum(self, dim):
        return Tensor(np.cumsum(self.values, axis=dim))

    def repeat_interleave(self, repeats, output_size):
        output = np.repeat(self.values, repeats.values)
        assert len(output) == output_size
        return Tensor(output)

    def record_stream(self, stream):
        self.recorded_streams.append(stream)

    def to(self, *, dtype):
        return Tensor(self.values.astype(dtype))

    def cpu(self):
        return self

    def tolist(self):
        return self.values.tolist()

    def __getitem__(self, index):
        return Tensor(self.values[index], self.requires_grad, self.source)

    def __add__(self, other):
        return Tensor(self.values + other.values)

    def __sub__(self, other):
        return Tensor(self.values - other.values)

    def __gt__(self, other):
        return Tensor(self.values > other)


class Event:
    def __init__(self, enable_timing=True):
        self.ready = True
        self.stream = None

    def record(self, stream):
        self.stream = stream

    def query(self):
        return self.ready

    def elapsed_time(self, other):
        return 1.0


class Stream:
    def __init__(self, device=None):
        self.device = device or Device()
        self.cuda_stream = id(self)
        self.waits = []

    def wait_event(self, event):
        self.waits.append(event)


class Autograd:
    def __init__(self):
        self.grad_calls = []
        self.backward_calls = []

    def grad(self, loss, inputs, **kwargs):
        self.grad_calls.append((loss, inputs, kwargs))
        return tuple(None if value is None else Tensor(value)
                     for value in loss.gradient_values)

    def backward(self, outputs, *, grad_tensors):
        self.backward_calls.append((outputs, grad_tensors))
        for output, gradient in zip(outputs, grad_tensors):
            param, coefficient = output.source
            value = float(np.sum(coefficient * gradient.values))
            param.grad = (param.grad or 0.0) + value


class Backend:
    def __init__(self, world_size=1):
        self.local = threading.local()
        self.default_stream = Stream()
        self.calls = []
        self.waited_work = []
        self.device_initializations = []
        self.world_size = world_size
        self.autograd = Autograd()
        self.torch = ModuleType('torch')
        self.dist = ModuleType('torch.distributed')
        self.dist.get_world_size = lambda: self.world_size
        self.dist.get_rank = lambda: 0
        self.dist.is_initialized = lambda: True
        self.dist.all_to_all_single = self.exchange
        self.torch.distributed = self.dist
        self.torch.Tensor = Tensor
        self.torch.is_tensor = lambda value: isinstance(value, Tensor)
        self.torch.device = lambda value: Device()
        self.torch.int64 = np.dtype('int64')
        self.torch.tensor = lambda values, dtype=None, device=None: Tensor(np.array(values, dtype=dtype))
        self.torch.zeros_like = lambda value: Tensor(np.zeros_like(value.values))
        self.torch.where = lambda mask, yes, no: Tensor(np.where(mask.values, yes.values, no.values))
        self.torch.stack = lambda values, dim=0: Tensor(np.stack([value.values for value in values], axis=dim))
        self.torch.cat = lambda values, dim=0: Tensor(np.concatenate([value.values for value in values], axis=dim))
        self.torch.no_grad = nullcontext
        self.torch.autograd = self.autograd
        self.torch.cuda = SimpleNamespace(
            is_available=lambda: True, current_stream=self.current_stream,
            Stream=Stream, Event=Event, stream=self.stream,
            set_device=lambda device: self.device_initializations.append(threading.get_native_id()),
            device=lambda device: nullcontext(),
            nvtx=SimpleNamespace(range_push=lambda name: None, range_pop=lambda: None))

    def current_stream(self, device=None):
        return getattr(self.local, 'stream', self.default_stream)

    @contextmanager
    def stream(self, stream):
        previous = self.current_stream()
        self.local.stream = stream
        try:
            yield stream
        finally:
            self.local.stream = previous

    def exchange(self, output, data, **kwargs):
        stage = self.local.stage
        self.calls.append((stage, data.values.copy(), kwargs))
        if data.values.ndim == 1:
            output.values[...] = np.array([2, 1], dtype=data.dtype)
        elif stage == 'C1':
            output.values[...] = data.values[[2, 0, 1]]
        else:
            output.values[...] = data.values[[1, 2, 0]]
        backend = self

        class Work:
            def wait(self):
                backend.waited_work.append(backend.current_stream())

        return Work()

    def load(self):
        root = Path(__file__).resolve().parents[1] / 'submodules/gsplat/gsplat'
        package_name = f'pc1_tensor_backend_{id(self)}'
        package = ModuleType(package_name)
        package.__path__ = [str(root)]
        wrappers = ModuleType(package_name + '.cuda._wrapper')
        for name in ('fully_fused_projection', 'isect_offset_encode', 'isect_tiles',
                     'rasterize_to_pixels', 'spherical_harmonics'):
            setattr(wrappers, name, lambda *args, **kwargs: None)
        distributed = ModuleType(package_name + '.distributed')
        distributed.all_gather_int32 = lambda *args, **kwargs: None
        distributed.all_gather_tensor_list = lambda *args, **kwargs: None
        modules = {'torch': self.torch, 'torch.distributed': self.dist,
                   package_name: package, package_name + '.cuda._wrapper': wrappers,
                   package_name + '.distributed': distributed}
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location(package_name + '.distributed_pipeline',
                                                          root / 'distributed_pipeline.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        return module

    def worker(self, module, **kwargs):
        worker = module.PipelineCommWorker(device=Device(), **kwargs)

        def execute(stage, request):
            self.local.stage = stage
            return worker._execute_request(stage, request)

        worker._execute = execute
        return worker

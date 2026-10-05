"""A stand-in for the ``tensorrt`` python module, enough to drive ``trt_build`` and ``TrtBackend`` without a GPU.

It implements the API surface the code uses and enforces the 8.6-vs-10 differences that matter:

* 8.6: ``create_network`` must get ``1 << EXPLICIT_BATCH`` or the ONNX parser refuses the model;
* 10.x: ``create_network(0)`` is right (EXPLICIT_BATCH is the default), ``max_workspace_size`` and the builder's
  ``platform_has_fast_fp16`` do not exist, and ``set_calibration_profile`` fails ("causes internal errors", as
  Ultralytics' own TensorRT builder documents);
* a timing cache from another version is refused (``set_timing_cache`` -> False, ``get_timing_cache`` -> None);
* an invalid optimisation profile makes ``add_optimization_profile`` return -1.

"Engines" are small JSON blobs pointing at the ONNX; executing one runs ONNX Runtime on the raw device pointers the
backend bound with ``set_tensor_address`` (CPU tensors in the tests), so the whole control flow of
``TrtBackend`` (binding, dynamic shapes, dtypes, output views) is exercised against real numbers.
"""

from __future__ import annotations

import ctypes
import enum
import json
from types import SimpleNamespace
from typing import Any, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, shape_inference


class DataType(enum.Enum):
    FLOAT = 0
    HALF = 1
    INT8 = 2
    INT32 = 3
    BOOL = 4
    UINT8 = 5


_ONNX_TO_TRT = {TensorProto.FLOAT: DataType.FLOAT, TensorProto.FLOAT16: DataType.HALF, TensorProto.INT32: DataType.INT32,
                TensorProto.INT64: DataType.INT32, TensorProto.UINT8: DataType.UINT8, TensorProto.BOOL: DataType.BOOL,
                TensorProto.INT8: DataType.INT8}
_TRT_TO_NP = {DataType.FLOAT: np.float32, DataType.HALF: np.float16, DataType.INT32: np.int32, DataType.UINT8: np.uint8,
              DataType.BOOL: np.bool_, DataType.INT8: np.int8}


class LayerType(enum.Enum):
    CONSTANT = 0
    OTHER = 1


class TensorIOMode(enum.Enum):
    NONE = 0
    INPUT = 1
    OUTPUT = 2


class BuilderFlag(enum.Enum):
    FP16 = 0
    INT8 = 1
    OBEY_PRECISION_CONSTRAINTS = 2
    PREFER_PRECISION_CONSTRAINTS = 3


class MemoryPoolType(enum.Enum):
    WORKSPACE = 0


class NetworkDefinitionCreationFlag(enum.IntEnum):
    EXPLICIT_BATCH = 0


class _Tensor:
    def __init__(self, name, dtype, shape):
        self.name, self.dtype, self.shape = name, dtype, tuple(shape)


class _Layer:
    def __init__(self, node, out_types, in_types):
        self.name = node.name
        self.type = LayerType.CONSTANT if node.op_type == "Constant" else LayerType.OTHER
        self.num_outputs = len(node.output)
        self._out = [_Tensor(o, t, ()) for o, t in zip(node.output, out_types)]
        self._in = [_Tensor(i, t, ()) if i else None for i, t in zip(node.input, in_types)]
        self.num_inputs = len(self._in)
        self.precision = None
        self.output_types: dict = {}

    def get_output(self, j):
        return self._out[j]

    def get_input(self, j):
        return self._in[j]

    def set_output_type(self, j, dtype):
        self.output_types[j] = dtype


class _Network:
    def __init__(self, flags: int):
        self.flags, self.layers, self.inputs, self.outputs, self.onnx_path = flags, [], [], [], None

    @property
    def num_layers(self):
        return len(self.layers)

    def get_layer(self, i):
        return self.layers[i]

    def get_input(self, i):
        return self.inputs[i]

    def get_output(self, i):
        return self.outputs[i]

    @property
    def num_outputs(self):
        return len(self.outputs)


class _Parser:
    def __init__(self, network: _Network, logger, major: int):
        self.network, self.major, self.errors = network, major, []

    @property
    def num_errors(self):
        return len(self.errors)

    def get_error(self, i):
        return self.errors[i]

    def parse_from_file(self, path: str) -> bool:
        if self.major < 10 and not self.network.flags & (1 << int(NetworkDefinitionCreationFlag.EXPLICIT_BATCH)):
            self.errors.append("Network must have explicit batch dimension (EXPLICIT_BATCH flag required before TensorRT 10)")
            return False
        m = shape_inference.infer_shapes(onnx.load(path))
        types = {v.name: v.type.tensor_type.elem_type for v in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input)}
        types.update({i.name: i.data_type for i in m.graph.initializer})

        def trt_type(name):
            return _ONNX_TO_TRT.get(types.get(name, TensorProto.FLOAT), DataType.FLOAT)

        for node in m.graph.node:
            self.network.layers.append(_Layer(node, [trt_type(o) for o in node.output], [trt_type(i) for i in node.input]))

        def shp(v):
            return tuple(d.dim_value if d.dim_value > 0 else -1 for d in v.type.tensor_type.shape.dim)

        self.network.inputs = [_Tensor(v.name, _ONNX_TO_TRT[v.type.tensor_type.elem_type], shp(v)) for v in m.graph.input]
        self.network.outputs = [_Tensor(v.name, _ONNX_TO_TRT[v.type.tensor_type.elem_type], shp(v)) for v in m.graph.output]
        self.network.onnx_path = path
        return True


class _Profile:
    def __init__(self):
        self.shapes = {}

    def set_shape(self, name, mn, op, mx):
        self.shapes[name] = (tuple(mn), tuple(op), tuple(mx))


class _TimingCache:
    def __init__(self, blob: bytes):
        self.blob = blob

    def serialize(self):
        return b"timing:" + self.blob


class _Config:
    def __init__(self, major: int):
        self.major = major
        self.flags, self.pools, self.profiles, self.int8_calibrator = set(), {}, [], None
        self.calibration_profile, self.timing_cache, self.builder_optimization_level = None, None, None
        if major < 10:
            self.max_workspace_size = 0  # deprecated alias that exists before 10; the code must not need it

    def set_flag(self, f):
        self.flags.add(f)

    def set_memory_pool_limit(self, pool, size):
        self.pools[pool] = size

    def add_optimization_profile(self, p):
        for mn, op, mx in p.shapes.values():
            if not (mn[0] <= op[0] <= mx[0]):
                return -1
        self.profiles.append(p)
        return len(self.profiles) - 1

    def set_calibration_profile(self, p):
        if self.major >= 10:
            raise RuntimeError("internal error: set_calibration_profile is deprecated in TensorRT 10")
        self.calibration_profile = p
        return True

    def create_timing_cache(self, blob):
        return _TimingCache(blob)

    def set_timing_cache(self, cache, ignore_mismatch):
        if cache.blob.startswith(b"foreign") and not ignore_mismatch:
            return False  # cache built by another TensorRT version / GPU
        self.timing_cache = cache
        return True

    def get_timing_cache(self):
        return self.timing_cache


class _Builder:
    def __init__(self, logger, major):
        self.major, self.last = major, None
        if major < 10:  # removed from the Builder in TensorRT 10
            self.platform_has_fast_fp16 = True
            self.platform_has_fast_int8 = True

    def create_network(self, flags=0):
        return _Network(flags)

    def create_builder_config(self):
        self.last = _Config(self.major)
        return self.last

    def create_optimization_profile(self):
        return _Profile()

    def build_serialized_network(self, network: _Network, config: _Config):
        self.config, self.network = config, network
        if BuilderFlag.INT8 in config.flags and config.int8_calibrator is not None:
            cal = config.int8_calibrator
            cal.calibration_seen = []  # what the builder received: (batch, 3, H, W) float arrays
            while True:
                ptrs = cal.get_batch([network.inputs[0].name])
                if ptrs is None:
                    break
                shape = (cal.get_batch_size(), 3, *network.inputs[0].shape[2:])
                buf = (ctypes.c_float * int(np.prod(shape))).from_address(ptrs[0])
                cal.calibration_seen.append(np.frombuffer(buf, np.float32).reshape(shape).copy())
            cal.write_calibration_cache(b"calibration-cache")
        prof = config.profiles[0].shapes if config.profiles else None
        blob = {"onnx": network.onnx_path, "profile": {k: list(v) for k, v in prof.items()} if prof else None,
                "inputs": [(t.name, t.dtype.name, list(t.shape)) for t in network.inputs],
                "outputs": [(t.name, t.dtype.name, list(t.shape)) for t in network.outputs]}
        return json.dumps(blob).encode()


# --------------------------------------------------------------------------- runtime
class _Engine:
    def __init__(self, blob: dict):
        self.blob = blob
        self.num_io_tensors = len(blob["inputs"]) + len(blob["outputs"])
        self._names = [i[0] for i in blob["inputs"]] + [o[0] for o in blob["outputs"]]
        self._info = {n: (dt, tuple(sh)) for n, dt, sh in blob["inputs"] + blob["outputs"]}

    def get_tensor_name(self, i):
        return self._names[i]

    def get_tensor_mode(self, name):
        return TensorIOMode.INPUT if name in [i[0] for i in self.blob["inputs"]] else TensorIOMode.OUTPUT

    def get_tensor_shape(self, name):
        return self._info[name][1]

    def get_tensor_dtype(self, name):
        return DataType[self._info[name][0]]

    def get_tensor_profile_shape(self, name, idx):
        mn, op, mx = self.blob["profile"][name]
        return tuple(mn), tuple(op), tuple(mx)

    def create_execution_context(self):
        return _Context(self)


class _Context:
    def __init__(self, engine: _Engine):
        self.engine, self.addr, self.in_shape = engine, {}, None
        self.session = ort.InferenceSession(engine.blob["onnx"], providers=["CPUExecutionProvider"])
        self.calls: List[Any] = []

    def set_input_shape(self, name, shape):
        assert name == self.engine._names[0]
        prof = self.engine.blob["profile"]
        if prof:
            mn, _, mx = prof[name]
            assert mn[0] <= shape[0] <= mx[0], f"batch {shape[0]} outside the optimisation profile {mn[0]}..{mx[0]}"
        self.in_shape = tuple(shape)
        return True

    def get_tensor_shape(self, name):
        shape = self.engine._info[name][1]
        if shape and shape[0] < 0:
            assert self.in_shape is not None, "set_input_shape must be called before querying dynamic output shapes"
            shape = (self.in_shape[0], *shape[1:])
        return shape

    def set_tensor_address(self, name, ptr):
        self.addr[name] = int(ptr)
        return True

    def execute_async_v3(self, stream):
        eng = self.engine
        name_in = eng._names[0]
        shape = self.in_shape or eng._info[name_in][1]
        assert all(n in self.addr for n in eng._names), "every I/O tensor must have an address"
        assert all(s > 0 for s in shape), "dynamic input shape was never set"
        buf = (ctypes.c_float * int(np.prod(shape))).from_address(self.addr[name_in])
        x = np.frombuffer(buf, np.float32).reshape(shape).copy()
        names = [o[0] for o in eng.blob["outputs"]]
        outs = self.session.run(names, {name_in: x})
        for n, arr in zip(names, outs):
            want = _TRT_TO_NP[eng.get_tensor_dtype(n)]
            arr = np.ascontiguousarray(arr.astype(want))
            ctypes.memmove(self.addr[n], arr.ctypes.data, arr.nbytes)
        self.calls.append((stream, shape))
        return True


class _Runtime:
    def __init__(self, logger):
        pass

    def deserialize_cuda_engine(self, blob: bytes):
        try:
            return _Engine(json.loads(bytes(blob).decode()))
        except (ValueError, UnicodeDecodeError):
            return None


class _Logger:
    WARNING, VERBOSE = 1, 2

    def __init__(self, level=1):
        self.level = level


def make_fake_trt(version: str = "10.3.0") -> SimpleNamespace:
    major = int(version.split(".")[0])

    class IInt8EntropyCalibrator2:
        def __init__(self):
            pass

    builders: List[_Builder] = []

    def Builder(logger):  # noqa: N802
        b = _Builder(logger, major)
        builders.append(b)
        return b

    ns = SimpleNamespace(builders=builders,
        __version__=version, Logger=_Logger, DataType=DataType, float16=DataType.HALF, float32=DataType.FLOAT,
        LayerType=LayerType, TensorIOMode=TensorIOMode, BuilderFlag=BuilderFlag, MemoryPoolType=MemoryPoolType,
        NetworkDefinitionCreationFlag=NetworkDefinitionCreationFlag, IInt8EntropyCalibrator2=IInt8EntropyCalibrator2,
        Builder=Builder, Runtime=_Runtime,
    )
    ns.OnnxParser = lambda network, logger: _Parser(network, logger, major)
    return ns

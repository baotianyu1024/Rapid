from __future__ import absolute_import
from .Node import Op

import numpy as np
import time
import os
import atexit
import json
import ctypes

class SharedTableOp(Op):
    def __init__(self, node, rank, nrank, table_shape, learning_rate, ctx,
        root_arr=None, storage_arr=None, bounds=(0, 0), init = None):

        from .. import ndarray, placeholder_op

        self.fake_param = placeholder_op("fake", value=ndarray.empty([1], ctx=ctx))

        super().__init__(SharedTableOp, [self.fake_param], ctx)

        import hetuCTR

        assert(ndarray.is_gpu_ctx(ctx))
        if "HETUCTR_INIT_ADDR" in os.environ:
            ip = os.environ["HETUCTR_INIT_ADDR"]
        else:
            ip = "127.0.0.1"
        port = 23456

        self.tbl_length, self.tbl_width = table_shape
        pull_bound, push_bound = bounds

        if not init:
            init = hetuCTR.Initializer(hetuCTR.InitType.Normal, 0 , 0.01)
        if root_arr is None:
            random_state = np.random.get_state()
            np.random.seed(0)
            root_arr = np.random.randint(0, nrank, self.tbl_length)
            np.random.set_state(random_state)
        if storage_arr is None:
            storage_arr = np.where(root_arr == rank)[0]

        # initialize dataloaders
        self.dl_node = node
        for d in node.dataloaders.values():
            d.init_states()
        self._gpu_input = None
        if os.environ.get('RAPID_HET_GPU_INPUT') == '1':
            preload_start = time.perf_counter()
            if os.environ.get('RAPID_HET_PIN_INPUT') == '1' or len(node.dataloaders) != 1:
                raise ValueError('GPU input diagnostic requires one loader and no host pinning')
            array = next(iter(node.dataloaders.values())).raw_data
            if not array.flags.c_contiguous:
                raise ValueError('HET GPU input diagnostic requires contiguous sparse input')
            runtime = ctypes.CDLL('libcudart.so.12')
            runtime.cudaSetDevice.argtypes = [ctypes.c_int]
            runtime.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            runtime.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            runtime.cudaGetErrorString.restype = ctypes.c_char_p
            pointer = ctypes.c_void_p()
            code = runtime.cudaSetDevice(ctx.device_id)
            if code:
                raise RuntimeError(f'cudaSetDevice failed: {runtime.cudaGetErrorString(code)!r}')
            code = runtime.cudaMalloc(ctypes.byref(pointer), array.nbytes)
            if code:
                raise RuntimeError(f'cudaMalloc failed: {runtime.cudaGetErrorString(code)!r}')
            code = runtime.cudaMemcpy(pointer, ctypes.c_void_p(array.ctypes.data), array.nbytes, 1)
            if code:
                raise RuntimeError(f'cudaMemcpy failed: {runtime.cudaGetErrorString(code)!r}')
            self._gpu_input = (array.ctypes.data, array.nbytes, pointer.value)
            print(f'HET_GPU_INPUT rank={rank} bytes={array.nbytes} preload_seconds={time.perf_counter() - preload_start:.6f}', flush=True)
        if os.environ.get('RAPID_HET_PIN_INPUT') == '1':
            pin_start = time.perf_counter()
            # Sparse IDs are consumed from host memory by HetuTable's async preprocessor.
            runtime = ctypes.CDLL('libcudart.so.12')
            runtime.cudaSetDevice.argtypes = [ctypes.c_int]
            runtime.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            runtime.cudaGetErrorString.restype = ctypes.c_char_p
            code = runtime.cudaSetDevice(ctx.device_id)
            if code:
                raise RuntimeError(f'cudaSetDevice failed: {runtime.cudaGetErrorString(code)!r}')
            for loader in node.dataloaders.values():
                array = loader.raw_data
                if not array.flags.c_contiguous:
                    raise ValueError('HET pinned sparse input must be contiguous')
                code = runtime.cudaHostRegister(ctypes.c_void_p(array.ctypes.data), array.nbytes, 0)
                if code:
                    raise RuntimeError(f'cudaHostRegister failed: {runtime.cudaGetErrorString(code)!r}')
            print(f'HET_PINNED_INPUT rank={rank} bytes={sum(d.raw_data.nbytes for d in node.dataloaders.values())} pin_seconds={time.perf_counter() - pin_start:.6f}', flush=True)

        self.table = hetuCTR.HetuTable(
            rank=rank, nrank=nrank, device_id=ctx.device_id, ip=ip, port=port,
            pull_bound = pull_bound, push_bound = push_bound, init=init, learning_rate=learning_rate,
            length = self.tbl_length, width = self.tbl_width,
            root_arr = root_arr, storage_arr = storage_arr, verbose=0
        )
        self.pipeline_ready = False

        self.task = None
        self._profile_dir = os.environ.get('RAPID_HET_INPUT_PROFILE_DIR')
        self._profile = dict(calls=0, wait_ns=0, input_views_ns=0, submit_ns=0,
                             backward_submit_ns=0, backward_calls=0)
        self._profile_seen = 0
        self._profile_backward_seen = 0
        if self._profile_dir:
            atexit.register(self._write_profile)

    def _write_profile(self):
        rank = os.environ.get('OMPI_COMM_WORLD_RANK', 'unknown')
        with open(os.path.join(self._profile_dir, f'shared_table_{rank}_{id(self)}.json'), 'w') as f:
            json.dump(dict(rank=rank, **self._profile), f)

    def _input_pointer(self, batch):
        if self._gpu_input is None:
            return batch.ctypes.data
        host_base, nbytes, device_base = self._gpu_input
        offset = batch.ctypes.data - host_base
        if offset < 0 or offset + batch.nbytes > nbytes:
            raise ValueError('HET batch is not a view of the preloaded GPU input')
        return device_base + offset

    def compute(self, input_vals, output_val, stream_handle=None, inference=False):
        start = time.perf_counter_ns() if self._profile_dir else 0
        if self.task:
            self.task.wait()
        after_wait = time.perf_counter_ns() if self._profile_dir else 0

        self.batch_idx = self.dl_node.get_arr("validate") if inference else self.dl_node.get_arr("train")
        self.next_batch_idx  = self.dl_node.get_next_arr("validate") if inference else self.dl_node.get_next_arr("train")
        after_views = time.perf_counter_ns() if self._profile_dir else 0

        if not self.pipeline_ready or inference:
            self.lookup_output = output_val
            self.table.preprocess(self._input_pointer(self.batch_idx), np.prod(self.batch_idx.shape))
            self.table.push_pull(0, output_val.handle.contents.data)

        if inference:
            self.pipeline_ready = False
        else:
            self.task = self.table.async_preprocess(self._input_pointer(self.next_batch_idx), np.prod(self.next_batch_idx.shape))
            assert(output_val is self.lookup_output)
        if self._profile_dir:
            self._profile_seen += 1
            if self._profile_seen > 30:
                self._profile['calls'] += 1
                self._profile['wait_ns'] += after_wait - start
                self._profile['input_views_ns'] += after_views - after_wait
                self._profile['submit_ns'] += time.perf_counter_ns() - after_views

    def compute_backward(self, gradients):
        start = time.perf_counter_ns() if self._profile_dir else 0
        self.pipeline_ready = True
        self.task = self.table.async_push_pull(gradients.handle.contents.data, self.lookup_output.handle.contents.data)
        if self._profile_dir:
            self._profile_backward_seen += 1
            if self._profile_backward_seen > 30:
                self._profile['backward_calls'] += 1
                self._profile['backward_submit_ns'] += time.perf_counter_ns() - start

    def gradient(self, output_grad):
        self.grad_node = SharedTableGradientOp(output_grad, self, ctx=self.ctx)
        return [self.grad_node]

    def infer_shape(self, input_shapes):
        output_shape = list(self.dl_node.get_cur_shape("train")) + [self.tbl_width]
        return tuple(output_shape)


class SharedTableGradientOp(Op):
    def __init__(self, output_grad, forward_node, ctx=None):
        super().__init__(SharedTableGradientOp, [output_grad], ctx)
        self.forward_node = forward_node

    def compute(self, input_vals, output_val, stream_handle=None):
        # we have to make sure gradient is ready
        if stream_handle:
            stream_handle.sync()
        self.forward_node.compute_backward(input_vals[0])

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return (1, )

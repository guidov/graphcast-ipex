import tensorrt as trt
import cupy as cp

# 初始化 Plugins
trt.init_libnvinfer_plugins(None, "")

class TrtRuntime:
    def __init__(self, context, engine, device_mem_holder):
        self.context = context
        self.engine = engine
        self.device_mem_holder = device_mem_holder
        self.context.device_memory = device_mem_holder.ptr
            
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.metadata = {
            name: {
                "dtype": trt.nptype(self.engine.get_tensor_dtype(name)),
                "mode": self.engine.get_tensor_mode(name),
            } for name in self.tensor_names
        }

    def run(self, output_feed=None, input_feed=None):
        input_feed = input_feed or {}
        
        for name, data in input_feed.items():
            if name in self.metadata:
                self.context.set_input_shape(name, data.shape)
                self.context.set_tensor_address(name, data.data.ptr)

        if output_feed is not None: output_names = list(output_feed.keys())
        else:
            output_names = [n for n, m in self.metadata.items() if m["mode"] == trt.TensorIOMode.OUTPUT]

        results = {} if output_feed is not None else []

        # 3. 绑定输出地址
        for name in output_names:
            if name not in self.metadata: continue
            real_shape = tuple(self.context.get_tensor_shape(name))
            
            if output_feed is not None:
                out_tensor = output_feed[name]
                if out_tensor.shape != real_shape:
                    raise ValueError(f"Output {name} shape mismatch. Expected {real_shape}, got {out_tensor.shape}")
                
                self.context.set_tensor_address(name, out_tensor.data.ptr)
                results[name] = out_tensor
            else:
                out_tensor = cp.empty(real_shape, dtype=self.metadata[name]["dtype"])
                self.context.set_tensor_address(name, out_tensor.data.ptr)
                results.append(out_tensor)
        # print(results)
        self.context.execute_async_v3(stream_handle=cp.cuda.Stream.null.ptr)
        cp.cuda.Stream.null.synchronize()
        return results

def InferenceSession(path, memory_pool=None):  
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)

    
    if isinstance(path, str):
        with open(path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    else: engine = runtime.deserialize_cuda_engine(path)
        
    required_size = engine.device_memory_size
    mem_holder = None

    # --- 核心逻辑 ---
    if memory_pool is not None:
        if required_size > memory_pool.size:
            raise MemoryError(f"Pool size({memory_pool.size}) < Required({required_size})")
        mem_holder = memory_pool 
    elif required_size > 0:
        mem_holder = cp.cuda.alloc(required_size)
    
    context = engine.create_execution_context_without_device_memory()
    return TrtRuntime(context, engine, mem_holder)

import numpy as np
import cupy as cp
import os
from time import time
import trtruntime as trt

class PoolManager:
    def __init__(self, size):
        self.pool = cp.cuda.alloc(size)
        self.size = size
        self.ptr, self.cur = self.pool.ptr, 0

    def empty(self, shape, dtype, offset=None):
        if offset is not None: self.cur = offset
        start = (self.cur + 255) & ~255
        n_bytes = int(np.prod(shape)) * cp.dtype(dtype).itemsize
        assert (self.cur + n_bytes) <= self.size, 'pool memory out!'
        mem = cp.cuda.UnownedMemory(self.ptr + start, n_bytes, owner=None)
        arr = cp.ndarray(shape, dtype=dtype, memptr=cp.cuda.MemoryPointer(mem, 0))
        self.cur = start + n_bytes
        return arr

    def zeros(self, shape, dtype, offset=None):
        arr = self.empty(shape, dtype, offset)
        arr.fill(0)
        return arr

class GraphCastInference:  
    def __init__(self, model_root, graph_body, norm_vector):
        with open(model_root+'/GraphCastInfer.json') as f:
            config = json.loads(f.read())
        # 1. 解析基础超参数与分块配置
        self.timespan = config['timespan']
        self.batch = config['batch']
        self.nlatent = config['nlatent']
        self.grid_size = config['grid_size']
        self.echunk = config['echunk']
        self.gchunk = config['gchunk']
        self.cursor = cp.zeros(1, dtype=cp.int64)
        
        # 解析切片索引
        slices = config['grid_slices']
        self.slice_const = slice(*slices['const'])
        self.slice_dynamic1 = slice(*slices['dynamic1'])
        self.slice_dynamic2 = slice(*slices['dynamic2'])
        self.slice_force1 = slice(*slices['force1'])
        self.slice_force2 = slice(*slices['force2'])
        self.slice_force3 = slice(*slices['force3'])
        self.slice_dynamic = slice(*slices['dynamic_all'])
        self.slice_g2m_feats = slice(*slices['g2m_feats'])

        # 2. 转换常量数据并持久化到 GPU
        self.graph_body = {k: cp.asarray(v) for k, v in graph_body.items()}
        
        # 归一化参数准备
        ser = ['const', 'dynamic', 'dynamic', 'force', 'force', 'force']
        self.b_norm = cp.asarray(np.concatenate([norm_vector['b_'+i] for i in ser]))
        self.k_norm = cp.asarray(np.concatenate([norm_vector['k_'+i] for i in ser]))
        self.k_diff_dynamic = cp.asarray(norm_vector['k_diff_dynamic'])

        # 维度计算
        self.ngrid = self.graph_body['g2m_grid_feats'].shape[0]
        self.nmesh = self.graph_body['g2m_mesh_feats'].shape[0]
        self.g2m_edges = self.graph_body['g2m_edge_feats'].shape[0]
        self.out_dim = self.k_diff_dynamic.shape[0]
        self.input_dim = len(self.b_norm) + self.graph_body['g2m_grid_feats'].shape[-1]
        
        # 3. 在内部写死动态缓存公式，并计算各个引擎所需的总显存
        g2m_ef = self.graph_body['g2m_edge_feats'][None, :]
        m2g_ef = self.graph_body['m2g_edge_feats'][None, :]

        g2m_volume = ((self.batch * self.nmesh * self.nlatent) + (self.batch * self.echunk * (self.input_dim + g2m_ef.shape[-1] + self.nlatent))) * 4 + self.echunk * 8
        m2g_volume = ((self.batch * self.gchunk * (self.input_dim + self.out_dim + self.nlatent * 3 + m2g_ef.shape[-1] * 3))) * 4 + (self.batch * self.ngrid * (self.out_dim*2 + 5)) * 4
        
        # 遍历外部引擎配置，根据引擎 name 自动累加代码内的写死逻辑
        volumes = []
        for eng in config['engines']:
            base_ctx = eng['contextlen']
            if eng['name'] == 'net_g2m':
                extra_vol = g2m_volume
            elif eng['name'] == 'net_m2g':
                extra_vol = m2g_volume
            else: extra_vol = 0
            volumes.append(base_ctx + extra_vol)
            
        total_volume = max(volumes) + 2048

        # 4. 初始化显存池
        self.infer_buf = PoolManager(total_volume)
        print("Dynamic total_volume:", total_volume)
        
        # 5. 动态抓取专属的 contextlen 锚点作为 offset，并循环加载引擎
        self.g2m_ctx = next(eng['contextlen'] for eng in config['engines'] if eng['name'] == 'net_g2m')
        self.m2g_ctx = next(eng['contextlen'] for eng in config['engines'] if eng['name'] == 'net_m2g')

        for eng in config['engines']:
            path = os.path.join(model_root, eng['file'])
            session = trt.InferenceSession(path, memory_pool=self.infer_buf)
            setattr(self, eng['name'], session)
            print(eng['file'], session.engine.device_memory_size_v2)
        
        # 6. 分配大缓存并建立视图
        self.x_grid = cp.empty((self.batch, self.ngrid, self.input_dim), dtype=cp.float32)
        self.const = self.x_grid[..., self.slice_const]
        self.dynamic1 = self.x_grid[..., self.slice_dynamic1] 
        self.dynamic2 = self.x_grid[..., self.slice_dynamic2]
        self.force1 = self.x_grid[..., self.slice_force1]
        self.force2 = self.x_grid[..., self.slice_force2]
        self.force3 = self.x_grid[..., self.slice_force3]
        self.x_dynamic = self.x_grid[..., self.slice_dynamic]
        self.g2m_grid_feats = self.x_grid[..., self.slice_g2m_feats]

        self.l_mesh = cp.empty((self.batch, self.nmesh, self.nlatent), dtype=cp.float32)
        self.m2g_idx = self.graph_body['m2g_idx']
        self.sort_idx = cp.argsort(self.m2g_idx[:, 1])
        self.s_idx_sorted = self.m2g_idx[self.sort_idx, 0]

        # 7. 静态分配功能功能缓冲区
        self.agg_msg = self.infer_buf.empty((self.batch, self.nmesh, self.nlatent), dtype=cp.float32, offset=int(self.g2m_ctx))
        self.g2m_grid_buf = self.infer_buf.empty((self.batch, self.echunk, self.input_dim), dtype=cp.float32)
        self.g2m_mesh_buf = self.infer_buf.empty((self.batch, self.echunk, self.nlatent), dtype=cp.float32)
        self.g2m_edge_buf = self.infer_buf.empty((self.batch, self.echunk, g2m_ef.shape[-1]), dtype=cp.float32)
        self.g2m_ridx_buf = self.infer_buf.empty(self.echunk, dtype=cp.int64)

        self.m2m_delta = self.infer_buf.empty((self.batch, self.ngrid, self.out_dim), dtype=cp.float32, offset=int(self.m2g_ctx))
        self.dynamic_buf = self.infer_buf.empty((self.batch, self.ngrid, self.out_dim), dtype=cp.float32)
        self.force_buf = self.infer_buf.empty((self.batch, *self.grid_size, 5), dtype=cp.float32)
        self.m2g_delta_buf = self.infer_buf.empty((self.batch, self.gchunk, self.out_dim), dtype=cp.float32)
        self.m2g_grid_buf = self.infer_buf.empty((self.batch, self.gchunk, self.input_dim), dtype=cp.float32)
        self.m2g_mesh_buf = self.infer_buf.empty((self.batch, self.gchunk * 3, self.nlatent), dtype=cp.float32)
        self.m2g_edge_buf = self.infer_buf.empty((self.batch, self.gchunk * 3, m2g_ef.shape[-1]), dtype=cp.float32)

    def set_status(self, const, dynamic1, dynamic2, t1):
        const_buf = self.infer_buf.empty((self.const.shape), dtype=cp.float32, offset=0)
        const_buf.set(const.reshape(const_buf.shape))
        self.const[:] = const_buf

        dynamic_buf = self.infer_buf.empty((self.dynamic1.shape), dtype=cp.float32, offset=0)
        dynamic_buf.set(dynamic1.reshape(dynamic_buf.shape))
        self.dynamic1[:] = dynamic_buf

        dynamic_buf.set(dynamic2.reshape(dynamic_buf.shape))
        self.dynamic2[:] = dynamic_buf

        self.cursor[0] = t1
        self.net_force.run({'forcevars': self.force_buf}, {'timestamp': self.cursor})
        self.force1[:] = self.force_buf.reshape(self.force1.shape)
        
        self.cursor += self.timespan
        self.net_force.run({'forcevars': self.force_buf}, {'timestamp': self.cursor})
        self.force2[:] = self.force_buf.reshape(self.force2.shape)
        
        self.g2m_grid_feats[:] = self.graph_body['g2m_grid_feats']

        self.x_dynamic -= self.b_norm
        self.x_dynamic /= self.k_norm
        
    def forward(self):        
        self.cursor += self.timespan
        self.net_force.run({'forcevars': self.force_buf}, {'timestamp': self.cursor})
        self.force3[:] = self.force_buf.reshape(self.force3.shape)
        self.force3 -= self.b_norm[self.slice_force3]
        self.force3 /= self.k_norm[self.slice_force3]
        
        a = time()
        # --- Step B: MeshEncoder ---
        m_static = self.graph_body['g2m_mesh_feats'][None, :]
        self.net_mesh.run({'latent_mesh': self.l_mesh}, {'g2m_mesh_feats': m_static})
        
        b = time()
        # --- Step C: G2MAggregate ---
        g2m_idx = self.graph_body['g2m_idx']
        g2m_ef = self.graph_body['g2m_edge_feats'][None, :]

        self.agg_msg.fill(0)
        
        for i in range(0, self.g2m_edges, self.echunk):
            s, e = i, i + self.echunk
            s_idx, r_idx = g2m_idx[s:e, 0], g2m_idx[s:e, 1]
            cp.take(self.x_grid, s_idx, axis=1, out=self.g2m_grid_buf)
            cp.take(self.l_mesh, r_idx, axis=1, out=self.g2m_mesh_buf)
            self.g2m_edge_buf[:] = g2m_ef[:, s:e, :]
            self.g2m_ridx_buf[:] = r_idx
            self.net_g2m.run({'next': self.agg_msg}, {
                's_feat': self.g2m_grid_buf, 'r_feat': self.g2m_mesh_buf, 'e_feat': self.g2m_edge_buf,
                'r_idx': self.g2m_ridx_buf, 'current': self.agg_msg
            })

        c = time()
        self.l_mesh[:] = self.agg_msg
        # --- Step D: M2MProcessorFull ---
        m2m_ef = self.graph_body['m2m_edge_feats'][None, :]
        cur_node = self.net_m2m.run({'node_out': self.l_mesh}, {
            'm_out_init': m_static, 'agg_msg': self.l_mesh, 'm2m_edge_feats': m2m_ef
        })['node_out']

        d = time()
        # --- Step E: M2GInteraction ---
        m2g_ef = self.graph_body['m2g_edge_feats'][None, :]
        
        for i in range(0, self.ngrid, self.gchunk):
            g_s, g_e = i, i + self.gchunk
            e_s, e_e = g_s * 3, g_e * 3
            self.m2g_grid_buf[:] = self.x_grid[:, g_s:g_e, :]
            cp.take(cur_node, self.s_idx_sorted[e_s:e_e], axis=1, out=self.m2g_mesh_buf)
            cp.take(m2g_ef, self.sort_idx[e_s:e_e], axis=1, out=self.m2g_edge_buf)

            self.net_m2g.run({'grid_delta_out': self.m2g_delta_buf}, {
                'r_grid_feat': self.m2g_grid_buf, 's_mesh_feat': self.m2g_mesh_buf, 'm2g_edge_feat': self.m2g_edge_buf
            })
            self.m2m_delta[:, g_s:g_e, :] = self.m2g_delta_buf
        e = time()
        return self.m2m_delta

    def step(self):
        self.dynamic_buf[:] = self.dynamic2
        self.dynamic1[:] = self.dynamic_buf

        force_buf = self.force_buf.reshape(self.force1.shape)
        force_buf[:] = self.force2
        self.force1[:] = force_buf

        force_buf[:] = self.force3
        self.force2[:] = force_buf

        dynamic_b_norm = self.b_norm[self.slice_dynamic2]
        dynamic_k_norm = self.k_norm[self.slice_dynamic2]

        self.dynamic2 *= dynamic_k_norm
        self.dynamic2 += dynamic_b_norm

        self.m2m_delta *= self.k_diff_dynamic
        self.m2m_delta += self.dynamic2

        self.dynamic2[:] = self.m2m_delta
        self.dynamic2 -= dynamic_b_norm
        self.dynamic2 /= dynamic_k_norm
        
        return self.m2m_delta, int(self.cursor[0])

if __name__ == '__main__':
    import json, ncutil
    from time import time
    
    model_dir = '../weights/turbo_5mesh_13level_1deg'
    # 1. 加载权重与配置
    graph_data = dict(np.load(model_dir + '/GraphBody.npz'))
    norm_data = dict(np.load(model_dir + '/NormVector.npz'))
    with open(model_dir + '/DataConfig.json') as f:
        data_config = json.loads(f.read())

    # 2. 提取需要的各种变量 Key 列表与层级
    const_keys = list(data_config['const_vars'].keys())
    dynamic_keys = list(data_config['dynamic_vars'].keys())
    levels = data_config['levels']

    predictor = GraphCastInference(
        model_root=model_dir, graph_body=graph_data, norm_vector=norm_data,
    )

    ds_data = ncutil.nc.Dataset('../weights/testdata_and_normvector/input_data_20220101_10_step4.nc')
    t0, t1 = ncutil.get_times(ds_data)[:2]

    # 获取一个静态数据，两个动态数据
    const = ncutil.get_data(ds_data, t0, keys=const_keys, levels=levels)
    dynamic_t1 = ncutil.get_data(ds_data, t0, keys=dynamic_keys, levels=levels)
    dynamic_t2 = ncutil.get_data(ds_data, t1, keys=dynamic_keys, levels=levels)

    # 设置推理器的初始状态
    predictor.set_status(const, dynamic_t1, dynamic_t2, t0)

    # 5. 自回归推理循环
    cp.cuda.Stream.null.synchronize()
    
    start = time()
    for i in range(4):
        delta = predictor.forward()
        dynamic_next, t = predictor.step()
        # 如果要保存，需要copy一份，或者get到内存，否则将在下个步骤被污染
    cp.cuda.Stream.null.synchronize()

    print("Inference Time:", time()-start)

    # 输出结果校验 (如果是 CuPy 数组则安全调用 .mean(), .min(), .max())
    print(dynamic_next.mean(), dynamic_next.min(), dynamic_next.max(),
          '\n13 levels after 4 step should be 13425.764 -2993.4424 204544.72')

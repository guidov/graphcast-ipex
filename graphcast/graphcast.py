import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

class MLPBlock(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, use_norm=True):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, mid_dim)
        self.lin2 = nn.Linear(mid_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity()

    def forward(self, x):
        x = self.lin1(x)
        x = F.silu(x) # Swish
        x = self.lin2(x)
        return self.norm(x)

class GraphInteraction(nn.Module):
    def __init__(self, edge_in_dim, edge_out_dim, node_in_dim, node_out_dim):
        super().__init__()
        self.edge_mlp = MLPBlock(edge_in_dim, edge_out_dim, edge_out_dim)
        self.node_mlp = MLPBlock(node_in_dim, node_out_dim, node_out_dim)

    def forward(self, s_nodes, r_nodes, edges, idx):
        senders, receivers = idx[:, 0], idx[:, 1]
        edge_in = torch.cat([edges, s_nodes[:, senders], r_nodes[:, receivers]], dim=-1)
        edge_delta = self.edge_mlp(edge_in)
        agg_msg = r_nodes * 0
        receivers = receivers.view(1, -1, 1)
        receivers = receivers.expand(edge_delta.size(0), -1, edge_delta.size(2))
        agg_msg.scatter_add_(1, receivers, edge_delta)
        node_in = torch.cat([r_nodes, agg_msg], dim=-1)
        node_delta = self.node_mlp(node_in)
        
        return edge_delta, node_delta

class GraphCastModel(nn.Module):
    def __init__(self, graph_body, norm_vector, latdim=512, layers=16):
        super().__init__()
        self.ngrid = graph_body['g2m_grid_feats'].shape[0]
        self.nmesh = graph_body['g2m_mesh_feats'].shape[0]
        # 1. 注册 Buffer (保持不变)
        self.register_buffer('g2m_gf', torch.tensor(graph_body['g2m_grid_feats'])[None, :], persistent=False)
        self.register_buffer('g2m_mf', torch.tensor(graph_body['g2m_mesh_feats'])[None, :], persistent=False)
        self.register_buffer('g2m_ef', torch.tensor(graph_body['g2m_edge_feats'])[None, :], persistent=False)
        self.register_buffer('g2m_idx', torch.tensor(graph_body['g2m_idx'], dtype=torch.long), persistent=False)
        self.register_buffer('m2m_ef', torch.tensor(graph_body['m2m_edge_feats'])[None, :], persistent=False)
        self.register_buffer('m2m_idx', torch.tensor(graph_body['m2m_idx'], dtype=torch.long), persistent=False)
        self.register_buffer('m2g_ef', torch.tensor(graph_body['m2g_edge_feats'])[None, :], persistent=False)
        self.register_buffer('m2g_idx', torch.tensor(graph_body['m2g_idx'], dtype=torch.long), persistent=False)

        self.register_buffer('norm_b', torch.cat([
            torch.tensor(norm_vector['b_const']),
            torch.tensor(norm_vector['b_dynamic']).repeat(2),
            torch.tensor(norm_vector['b_force']).repeat(3)
        ]), persistent=False)
        self.register_buffer('norm_k', torch.cat([
            torch.tensor(norm_vector['k_const']),
            torch.tensor(norm_vector['k_dynamic']).repeat(2),
            torch.tensor(norm_vector['k_force']).repeat(3)
        ]), persistent=False)
        self.register_buffer('k_diff_dynamic', torch.tensor(norm_vector['k_diff_dynamic']), persistent=False)

        # Grid2Mesh
        self.g2m_enc_grid = MLPBlock(self.norm_b.shape[0]+graph_body['g2m_grid_feats'].shape[-1], latdim, latdim, True)
        # make_mlp_from_weights(weights_dict['g2m_enc_grid'])
        self.g2m_enc_mesh = MLPBlock(self.norm_b.shape[0]+graph_body['g2m_mesh_feats'].shape[-1], latdim, latdim, True)
        self.g2m_enc_edge = MLPBlock(graph_body['g2m_edge_feats'].shape[-1], latdim, latdim, True)
        self.g2m_int = GraphInteraction(latdim*3, latdim, latdim*2, latdim)
        self.g2m_grid_self = MLPBlock(latdim, latdim, latdim, True)

        # Mesh2Mesh
        self.m2m_enc_edge = MLPBlock(graph_body['m2m_edge_feats'].shape[-1], latdim, latdim, True)
        self.processors = nn.ModuleList([
            GraphInteraction(latdim*3, latdim, latdim*2, latdim) for i in range(layers)
        ])

        # Mesh2Grid
        self.m2g_enc_edge = MLPBlock(graph_body['m2g_edge_feats'].shape[-1], latdim, latdim, True)
        self.m2g_int = GraphInteraction(latdim*3, latdim, latdim*2, latdim)
        self.head = MLPBlock(latdim, latdim, norm_vector['k_dynamic'].shape[0], False)

        # 3. 加载权重
        # load_weights(self, weights_dict)

    def _empty_cache(self):
        device = next(self.parameters()).device
        if device.type == 'xpu':
            torch.xpu.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()

    def grid2mesh(self, x):
        latent_grid = self.g2m_enc_grid(torch.cat([x, self.g2m_gf], dim=-1))
        latent_mesh = self.g2m_enc_mesh(torch.cat([x[:, :self.nmesh] * 0, self.g2m_mf], dim=-1))
        
        # Memory-efficient chunking to avoid OOM on GPUs with <= 12GB VRAM
        chunk_size = 32768
        g2m_edges = self.g2m_ef.shape[1]
        senders, receivers = self.g2m_idx[:, 0], self.g2m_idx[:, 1]
        agg_msg = torch.zeros_like(latent_mesh)
        
        for i in range(0, g2m_edges, chunk_size):
            s_idx = senders[i:i+chunk_size]
            r_idx = receivers[i:i+chunk_size]
            
            # Encode edges chunk
            ef_chunk = self.g2m_ef[:, i:i+chunk_size, :]
            latent_edge_chunk = self.g2m_enc_edge(ef_chunk)
            
            # Gather nodes chunk
            s_nodes_chunk = latent_grid[:, s_idx, :]
            r_nodes_chunk = latent_mesh[:, r_idx, :]
            
            # Concatenate and run Edge MLP
            edge_in = torch.cat([latent_edge_chunk, s_nodes_chunk, r_nodes_chunk], dim=-1)
            edge_delta = self.g2m_int.edge_mlp(edge_in)
            
            # Scatter add message
            r_idx_view = r_idx.view(1, -1, 1).expand(edge_delta.size(0), -1, edge_delta.size(2))
            agg_msg.scatter_add_(1, r_idx_view, edge_delta)
            
            # Clean up loop variables to release VRAM immediately
            del ef_chunk, latent_edge_chunk, s_nodes_chunk, r_nodes_chunk, edge_in, edge_delta, r_idx_view
            self._empty_cache()
            
        # Node MLP
        node_in = torch.cat([latent_mesh, agg_msg], dim=-1)
        node_delta = self.g2m_int.node_mlp(node_in)
        
        grid_out = latent_grid + self.g2m_grid_self(latent_grid)
        self._empty_cache()
        return grid_out, latent_mesh + node_delta

    def mesh2mesh(self, m2m_node_in):
        cur_node = m2m_node_in
        cur_edge = self.m2m_enc_edge(self.m2m_ef)
        
        for n, processor in enumerate(self.processors):
            # print(n)
            e_delta, n_delta = processor(cur_node, cur_node, cur_edge, self.m2m_idx)
            cur_edge = cur_edge + e_delta
            cur_node = cur_node + n_delta
        self._empty_cache()
        return cur_node

    def mesh2grid(self, mesh_node, grid_node):
        # Memory-efficient chunking to avoid OOM on GPUs with <= 12GB VRAM
        chunk_size = 32768
        m2g_edges = self.m2g_ef.shape[1]
        senders, receivers = self.m2g_idx[:, 0], self.m2g_idx[:, 1]
        agg_msg = torch.zeros_like(grid_node)
        
        for i in range(0, m2g_edges, chunk_size):
            s_idx = senders[i:i+chunk_size]
            r_idx = receivers[i:i+chunk_size]
            
            # Encode edges chunk
            ef_chunk = self.m2g_ef[:, i:i+chunk_size, :]
            latent_edge_chunk = self.m2g_enc_edge(ef_chunk)
            
            # Gather nodes chunk
            s_nodes_chunk = mesh_node[:, s_idx, :]
            r_nodes_chunk = grid_node[:, r_idx, :]
            
            # Concatenate and run Edge MLP
            edge_in = torch.cat([latent_edge_chunk, s_nodes_chunk, r_nodes_chunk], dim=-1)
            edge_delta = self.m2g_int.edge_mlp(edge_in)
            
            # Scatter add message
            r_idx_view = r_idx.view(1, -1, 1).expand(edge_delta.size(0), -1, edge_delta.size(2))
            agg_msg.scatter_add_(1, r_idx_view, edge_delta)
            
            # Clean up loop variables to release VRAM immediately
            del ef_chunk, latent_edge_chunk, s_nodes_chunk, r_nodes_chunk, edge_in, edge_delta, r_idx_view
            self._empty_cache()
            
        # Node MLP
        node_in = torch.cat([grid_node, agg_msg], dim=-1)
        node_delta = self.m2g_int.node_mlp(node_in)
        
        final_grid = grid_node + node_delta
        self._empty_cache()
        return self.head(final_grid)

    def forward(self, const, dynamic1, dynamic2, force1, force2, force3, target=None):
        if self.training:
            dynamic1 = dynamic1.requires_grad_(True)
            dynamic2 = dynamic2.requires_grad_(True)
        
        x = torch.cat([const, dynamic1, dynamic2, force1, force2, force3], dim=-1)
        x = x.view(1, -1, x.shape[-1])
        x = (x - self.norm_b) / self.norm_k

        if self.training:
            g2m_gf_out, g2m_mf_out = checkpoint(self.grid2mesh, x, use_reentrant=False)
            m2m_mf_out = checkpoint(self.mesh2mesh, g2m_mf_out, use_reentrant=False)
            dynamic_delta = checkpoint(self.mesh2grid, m2m_mf_out, g2m_gf_out, use_reentrant=False)
        else:
            g2m_gf_out, g2m_mf_out = self.grid2mesh(x)
            m2m_mf_out = self.mesh2mesh(g2m_mf_out)
            dynamic_delta = self.mesh2grid(m2m_mf_out, g2m_gf_out)

        dynamic_delta = dynamic_delta.view(dynamic1.shape) * self.k_diff_dynamic
        if target is None: return dynamic2 + dynamic_delta
        
        target_delta = target / self.k_diff_dynamic
        diff = dynamic_delta - target_delta
        return dynamic_next, diff

if __name__ == '__main__':
    import pickle
    import numpy as np
    
    model_dir = '../weights/lite_5mesh_13level_1deg'
    graph_body = dict(np.load(model_dir + '/GraphBody.npz'))
    
    norm_vector = dict(np.load(model_dir + '/NormVector.npz'))

    
    model = GraphCastModel(graph_body, norm_vector)    
    model.load_state_dict(torch.load(model_dir + '/GraphWeight.pth'))
    
    const = torch.zeros([181, 360, 2], dtype=torch.float32)
    dynamic_t1 = dynamic_t2 = torch.zeros([181, 360, 83], dtype=torch.float32)
    force_t1 = force_t2 = force_t3 = torch.zeros([181, 360, 5], dtype=torch.float32)

    cmd = 'infer'
    
    if cmd == 'infer':
        with torch.no_grad():
            y = model.forward(const, dynamic_t1, dynamic_t2, force_t1, force_t2, force_t3)
        print(y.mean(), y.max(), y.min(), '\n should be 9.505517 816.1084 -551.87756')

    if cmd == 'export':
        with torch.no_grad():
            torch.onnx.export(
                model, (const, dynamic_t1, dynamic_t2, force_t1, force_t2, force_t3),
                model_dir + 'GraphWeight.onnx', 
                dynamo=False, export_params=True, opset_version=18, do_constant_folding=False, keep_initializers_as_inputs=False,
                input_names=['const', 'dynamic_t1', 'dynamic_t2', 'force_t1', 'force_t2', 'force_next'],
                output_names=['dynamic_next'], dynamic_axes=None, verbose=True, training=torch.onnx.TrainingMode.EVAL
            )
    
    

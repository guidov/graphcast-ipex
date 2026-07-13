import numpy as np

def icosahedron():
    verts = [[ 0.4911234495,  0.8506507874,  0.1875924650],
             [-0.3035309916,  0.5257310867,  0.7946544527],
             [ 0.6070619876,  0.0000000000,  0.7946544411],
             [ 0.4911234495, -0.8506507874,  0.1875924650],
             [ 0.3035309916,  0.5257310867, -0.7946544527],
             [-0.9822469177,  0.0000000000,  0.1875924579],
             [-0.4911234495,  0.8506507874, -0.1875924650],
             [-0.3035309916, -0.5257310867,  0.7946544527],
             [ 0.9822469177,  0.0000000000, -0.1875924579],
             [-0.4911234495, -0.8506507874, -0.1875924650],
             [ 0.3035309916, -0.5257310867, -0.7946544527],
             [-0.6070619876,  0.0000000000, -0.7946544411]]
    
    faces = [(0, 1, 2 ), (0, 6, 1 ), (8, 0, 2 ), (8, 4, 0 ),
             (3, 8, 2 ), (3, 2, 7 ), (7, 2, 1 ), (0, 4, 6 ),
             (4, 11,6 ), (6, 11,5 ), (1, 5, 7 ), (4, 10,11),
             (4, 8, 10), (10,8, 3 ), (10,3, 9 ), (11,10,9 ),
             (11,9, 5 ), (5, 9, 7 ), (9, 3, 7 ), (1, 6, 5 )]
    return np.array(verts, np.float32), np.array(faces, np.int32)

def multi_mesh(level):
    verts, faces = icosahedron()
    layers = [(verts, faces)]

    for i in range(level):
        verts, faces = layers[-1]
        edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        edges_sorted = np.sort(edges, axis=1)
        unique_edges, first_idx, inv = np.unique(
            edges_sorted, axis=0, return_index=True, return_inverse=True)
        reorder = np.argsort(first_idx)
        rev_reorder = np.argsort(reorder)
        inv = rev_reorder[inv].astype(np.int32)
        new_verts = verts[unique_edges[reorder]].mean(axis=1)
        new_verts /= np.linalg.norm(new_verts, axis=1)[:, None]
        all_verts = np.concatenate((verts, new_verts), axis=0)
        mid_idx = inv.reshape(-1, 3) + len(verts)
        ef = np.concatenate((faces, mid_idx), axis=1)
        new_faces = ef[:, [0, 3, 5, 3, 1, 4, 5, 4, 2, 3, 4, 5]].reshape(-1, 3)
        layers.append((all_verts, new_faces))
        
    all_faces = np.concatenate([i[1] for i in layers], axis=0)
    finest = np.diff(all_verts[new_faces[::4,[0,1,2,0]]], axis=1)
    return all_verts, all_faces, np.linalg.norm(finest, axis=2).max()

if __name__ == '__main__':
    a = multi_mesh(6)

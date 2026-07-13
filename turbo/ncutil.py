import netCDF4 as nc
import numpy as np
import json, os

def extract(data, key, n, levelidx, dims=('lat', 'lon', 'level')):
    if key not in data.variables:
        return None
    
    dimensions = list(data[key].dimensions)
    sli = [slice(None) if i in dims else 0 for i in dimensions]
    if 'time' in dimensions: sli[dimensions.index('time')] = n
    if 'level' in dimensions: sli[dimensions.index('level')] = levelidx
    
    dimensions = [i for i in dimensions if i in dims]
    v = data[key][tuple(sli)].data
        
    v = v.transpose([dimensions.index(i) for i in dims if i in dimensions])
    if v.ndim == 2 or v.ndim == 0: 
        v = v[..., None]
    return v

def extract_all(data, keys, n, levelidx=slice(None), dims=('lat', 'lon', 'level')):
    variables = {}
    for key in keys:
        v = extract(data, key, n, levelidx, dims)
        if v is not None:
            variables[key] = v
    return variables

def get_times(data):
    times = nc.num2date(data['datetime'], units=data['datetime'].units)[:].data[0]
    return times.astype("datetime64[s]").astype(np.int64)

def stack(data, index=None, sort=True):
    if index is None: index = list(data.keys())
    if sort: index = sorted(index)
    return np.concatenate([data[i].astype(np.float32) for i in index if i in data], axis=-1)

def unstack(data, var_dims_dict):
    index = sorted(var_dims_dict)
    arrs = np.split(data, np.cumsum([var_dims_dict[i] for i in index[:-1]]), axis=-1)
    return dict(zip(index, arrs))

def get_data(ds_data, t, keys, levels=[0]):
    alllevels = ds_data['level'][:].data.tolist()
    levelidx = [alllevels.index(i) for i in levels if i in alllevels]
    
    times = nc.num2date(ds_data['datetime'], units=ds_data['datetime'].units)[:].data[0]
    times = times.astype("datetime64[s]").astype(np.int64).tolist()
    n = times.index(t) if t in times else -1
    
    if n == -1: return None
        
    dict_data = extract_all(ds_data, keys, n, levelidx)
    return stack(dict_data, index=keys, sort=True) 

def init_nc(path, dynamic_vars={}, lats=[], lons=[], levels=[], **key):
    if len(lats) == 3: lats = np.linspace(*lats, dtype=np.float32)
    if len(lons) == 3: lons = np.linspace(*lons, dtype=np.float32)
    
    with nc.Dataset(path, 'w', format='NETCDF4') as ds:
        ds.createDimension('time', None)
        ds.createDimension('level', len(levels))
        ds.createDimension('lat', len(lats))
        ds.createDimension('lon', len(lons))
        
        times_v = ds.createVariable('time', 'f8', ('time',))
        levels_v = ds.createVariable('level', 'f4', ('level',))
        lats_v = ds.createVariable('lat', 'f4', ('lat',))
        lons_v = ds.createVariable('lon', 'f4', ('lon',))
        
        levels_v[:], lats_v[:], lons_v[:] = levels, lats, lons

        for var_name, ndim in dynamic_vars.items():
            if ndim > 1: ds.createVariable(var_name, 'f4', ('time', 'level', 'lat', 'lon'), zlib=True)
            else: ds.createVariable(var_name, 'f4', ('time', 'lat', 'lon'), zlib=True)

def write_nc(path, data, t):
    with nc.Dataset(path, 'a') as ds:
        curr_idx = len(ds.dimensions['time'])
        ds.variables['time'][curr_idx] = t
        
        for var_name, var_data in data.items():
            if var_name not in ds.variables: continue

            if var_data.shape[-1] == 1:
                ds.variables[var_name][curr_idx, :, :] = var_data[..., 0]
            else:
                ds.variables[var_name][curr_idx, :, :, :] = var_data.transpose(2, 0, 1)


if __name__ == '__main__':
    # 加载配置
    with open('../weights/data_era5_37level_0.25deg.json') as f: 
        data_conf = json.loads(f.read())
        
    levels = data_conf['levels']
    
    # --- 2. 归一化特征生成 (原 get_norm_vector 逻辑平铺在此) ---
    mean = nc.Dataset('../data/mean_by_level.nc')
    std = nc.Dataset('../data/stddev_by_level.nc')
    diff = nc.Dataset('../data/diffs_stddev_by_level.nc')
    
    alllevels = mean['level'][:].data.tolist()
    levelidx = [alllevels.index(i) for i in levels]
    
    # 根据原配置的分类，直接利用通用的 extract_all 和 stack 在这里动态组装
    vars3 = [data_conf['const_vars'], data_conf['dynamic_vars'], data_conf['force_vars']]
    
    norm_mean = [stack(extract_all(mean, i, 0, levelidx)) for i in vars3]
    norm_std = [stack(extract_all(std, i, 0, levelidx)) for i in vars3]
    norm_diff_dynamic = stack(extract_all(diff, data_conf['dynamic_vars'], 0, levelidx))
    
    norm_fact = {
        'b_const': norm_mean[0],      
        'b_dynamic': norm_mean[1],    
        'b_force': norm_mean[2],      
        'k_const': norm_std[0],
        'k_dynamic': norm_std[1],     
        'k_force': norm_std[2],       
        'k_diff_dynamic': norm_diff_dynamic  
    }
    
    np.savez_compressed('../weights/norm_era5_37level.npz', **norm_fact)

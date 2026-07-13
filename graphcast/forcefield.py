import numpy as np

_TIME_COORDS = np.arange(1951.5, 2035.5, 1.0, dtype=np.float32)
_TSI_RAW = np.array([
    1365.7765, 1365.7676, 1365.6284, 1365.6564, 1365.7773, 1366.3109, 1366.6681, 1366.6328, 1366.3828, 1366.2767,
    1365.9199, 1365.7484, 1365.6963, 1365.6976, 1365.7341, 1365.9178, 1366.1143, 1366.1644, 1366.2476, 1366.2426,
    1365.9580, 1366.0525, 1365.7991, 1365.7271, 1365.5345, 1365.6453, 1365.8331, 1366.2747, 1366.6348, 1366.6482,
    1366.6951, 1366.2859, 1366.1992, 1365.8103, 1365.6416, 1365.6379, 1365.7899, 1366.0826, 1366.6479, 1366.5533,
    1366.4457, 1366.3021, 1366.0286, 1365.7971, 1365.6996, 1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836,
    1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545, 1365.8107, 1365.7240, 1365.6918, 1365.6121, 1365.7399,
    1366.1021, 1366.3851, 1366.6836, 1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545, 1365.8107, 1365.7240,
    1365.6918, 1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836, 1366.6022, 1366.6807, 1366.2300, 1366.0480,
    1365.8545, 1365.8107, 1365.7240, 1365.6918
], dtype=np.float32)

TSI_VALUES = 0.9965 * _TSI_RAW

SEC_PER_DAY = np.float32(86400.0)
DEG_TO_RAD = np.float32(np.pi / 180.0)

# 计算太阳辐射
def toa(timestamp, lats, lons, window_sec=3600.0):
    ts_mid = timestamp - (window_sec * 0.5)
    d_shifted = (ts_mid // 86400) + 731
    cycle, cycle_rem = d_shifted // 1461, d_shifted % 1461
    y_in_c = (cycle_rem - 1) // 365 * (cycle_rem != 0)
    y_int = 1968 + cycle * 4 + y_in_c
    y_len = 365.0 + (y_in_c == 0)
    doy0 = cycle_rem - (366 + (y_in_c - 1) * 365) * (y_in_c != 0)
    
    f_yr = y_int + (doy0 + ((ts_mid % 86400) / 86400.0)) / y_len
    tsi = np.interp(f_yr, _TIME_COORDS, TSI_VALUES)
    
    j2k = (ts_mid - 946728000.0) / 86400.0
    theta = j2k / 365.25
    rel = 1.7535 + 6.283076 * theta
    rlls = 4.8951 + 6.283076 * theta
    ra = 6.240041 + 6.283020 * theta
    rllls = (4.8952 + 6.283320 * theta - 0.0075 * np.sin(rel) - 0.0326 * np.cos(rel) -
             0.0003 * np.sin(2 * rel) + 0.0002 * np.cos(2 * rel))
    
    s_dec = np.sin(0.409093) * np.sin(rllls)
    c_dec = np.sqrt(np.maximum(0.0, 1.0 - s_dec**2))
    dist2 = (1.0001 - 0.0163 * np.sin(rel) + 0.0037 * np.cos(rel))**2
    eq_s = 591.8 * np.sin(2 * rlls) - 459.4 * np.sin(ra) + \
           39.5 * np.sin(ra) * np.cos(2 * rlls) - 12.7 * np.sin(4 * rlls) - 4.8 * np.sin(2 * ra)
    
    lat_r = (lats * (np.pi / 180.0)).reshape(-1, 1)
    lon_r = (lons * (np.pi / 180.0)).reshape(1, -1)
    A = np.sin(lat_r) * s_dec
    B = np.cos(lat_r) * c_dec

    p2_raw = 2.0 * np.pi * ((timestamp % 86400.0) / 86400.0 - 0.5 + eq_s / 86400.0) + lon_r
    p2 = (p2_raw + np.pi) % (2.0 * np.pi) - np.pi
    p1 = p2 - 2.0 * np.pi * (window_sec / 86400.0)

    crit = np.arccos(np.clip(-A / (B + 1e-12), -1.0, 1.0))
    crit = np.where((A - B) >= -1e-7, np.pi, crit)
    crit = np.where((A + B) <= 1e-7, 0.0, crit)

    t1_min, t1_max = np.maximum(p1, -crit), np.minimum(p2, crit)
    int1 = A * (t1_max - t1_min) + B * (np.sin(t1_max) - np.sin(t1_min))
    int1 *= t1_max > t1_min
    p1_v2, p2_v2 = p1 + 2.0 * np.pi, np.pi
    t2_min, t2_max = np.maximum(p1_v2, -crit), np.minimum(p2_v2, crit)
    int2 = A * (t2_max - t2_min) + B * (np.sin(t2_max) - np.sin(t2_min))
    int2 *= t2_max > t2_min
    return np.maximum(0.0, (tsi / dist2) * (int2+int1) * (86400.0 / (2.0 * np.pi)))

def term(t, lat, lon):
    year_progress_f64 = np.mod(t / (3600 * 24 * 365.24219), 1.0) * (2 * np.pi)
    day_progress_greenwich_f64 = np.mod(t, 3600 * 24) / (3600 * 24)

    longitude_offsets = (np.deg2rad(lon) / (2 * np.pi)).astype(np.float32)
    
    year_progress = np.float32(year_progress_f64)
    day_progress_greenwich = np.float32(day_progress_greenwich_f64)
    
    y_sin_val = np.sin(year_progress)
    y_cos_val = np.cos(year_progress)
    
    day_progress = np.mod(day_progress_greenwich + longitude_offsets, 1.0) * np.float32(2 * np.pi)
    
    d_sin_val = np.sin(day_progress) # [N]
    d_cos_val = np.cos(day_progress) # [N]
    target_shape = (len(lat), len(lon), 1)

    y_sin = np.broadcast_to(np.array([[y_sin_val]], dtype=np.float32), target_shape)
    y_cos = np.broadcast_to(np.array([[y_cos_val]], dtype=np.float32), target_shape)
    
    d_sin = np.broadcast_to(d_sin_val[None, :, None], target_shape)
    d_cos = np.broadcast_to(d_cos_val[None, :, None], target_shape)
    
    return y_sin, y_cos, d_sin, d_cos

def force_5cn(t, lat_rg, lon_rg):
    lat = np.linspace(*lat_rg, dtype=np.float32)
    lon = np.linspace(*lon_rg, dtype=np.float32)
    toas = toa(t, lat, lon)[...,None]
    y_sin, y_cos, d_sin, d_cos = term(t, lat, lon)
    return np.concatenate((d_cos, d_sin, toas, y_cos, y_sin), axis=-1)

if __name__ == "__main__":
    import datetime, matplotlib.pyplot as plt
    
    test_lats = np.arange(-90, 90.01, 0.25, dtype=np.float32)
    test_lons = np.arange(-180, 180, 0.25, dtype=np.float32)
    
    from time import time
    start = time()
    timestamp_str = "2020-06-21 12:00:00"
    unix_ts = np.int64(1592740800)
    result = force_5cn(unix_ts, [-90.0, 90.0, 181], [0.0, 359.0, 360])

    import onnxruntime as ort
    net = ort.InferenceSession('./ForceField.onnx', providers=['CPUExecutionProvider'])
    aaaa
    y = net.run(None, {'timestamp': np.array(unix_ts)})[0]
    print(time()-start)
 

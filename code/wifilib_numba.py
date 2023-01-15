import numpy as np
import math
import os
from numba import jit


def read_bf_file(filename):
    with open(filename, "rb") as f:
        bfee_list = []
        field_len = int.from_bytes(f.read(2), byteorder='big', signed=False)
        while field_len != 0:
            arr = f.read(field_len)
            if len(arr) == field_len:  # 读取的数据长度与预期相等才保留
                bfee_list.append(arr)
            field_len = int.from_bytes(f.read(2), byteorder='big', signed=False)

    dicts = []

    count = 0  # % Number of records output
    broken_perm = 0  # % Flag marking whether we've encountered a broken CSI yet
    triangle = [0, 1, 3]  # % What perm should sum to for 1,2,3 antennas
    # triangle = [1, 3, 6]

    for array in bfee_list:
        # % Read size and code
        code = array[0]

        # there is CSI in field if code == 187，If unhandled code skip (seek over) the record and continue
        # 如果长度小于213会报错，故直接跳过
        if code != 187:
            # % skip all other info
            continue
        else:
            # get beamforming or phy data
            count = count + 1

            timestamp_low = int.from_bytes(array[1:5], byteorder='little', signed=False)
            bfee_count = int.from_bytes(array[5:7], byteorder='little', signed=False)

            Nrx = array[9]
            Ntx = array[10]
            rssi_a = array[11]
            rssi_b = array[12]
            rssi_c = array[13]
            noise = array[14] - 256
            agc = array[15]
            antenna_sel = array[16]
            b_len = int.from_bytes(array[17:19], byteorder='little', signed=False)
            fake_rate_n_flags = int.from_bytes(array[19:21], byteorder='little', signed=False)
            payload = array[21:]  # get payload

            calc_len = (30 * (Nrx * Ntx * 8 * 2 + 3) + 6) / 8
            perm = [1, 2, 3]
            perm[0] = ((antenna_sel) & 0x3)
            perm[1] = ((antenna_sel >> 2) & 0x3)
            perm[2] = ((antenna_sel >> 4) & 0x3)

            # Check that length matches what it should
            if (b_len != calc_len):
                print("MIMOToolbox:read_bfee_new:size", "Wrong beamforming matrix size.")
                continue

            # Compute CSI from all this crap :
            csi = parse_csi(payload, Ntx, Nrx)

            # % matrix does not contain default values
            if sum(perm) != triangle[Nrx - 1]:
                print('WARN ONCE: Found CSI (', filename, ') with Nrx=', Nrx, ' and invalid perm=[', perm, ']\n')
                continue
            else:
                csi[:, perm, :] = csi[:, [0, 1, 2], :]

            # dict,and return
            bfee_dict = {
                'timestamp_low': timestamp_low,
                'bfee_count': bfee_count,
                'Nrx': Nrx,
                'Ntx': Ntx,
                'rssi_a': rssi_a,
                'rssi_b': rssi_b,
                'rssi_c': rssi_c,
                'noise': noise,
                'agc': agc,
                'antenna_sel': antenna_sel,
                'perm': perm,
                'len': b_len,
                'fake_rate_n_flags': fake_rate_n_flags,
                'calc_len': calc_len,
                'csi': csi
            }

            dicts.append(bfee_dict)

    return dicts


# def parse_csi_new(payload, Ntx, Nrx):
#     # Compute CSI from all this crap
#     csi = np.zeros(shape=(30, Nrx, Ntx), dtype=np.dtype(complex))
#     index = 0

#     for i in range(30):
#         index += 3
#         remainder = index % 8
#         for j in range(Nrx):
#             for k in range(Ntx):
#                 real_bin = (int.from_bytes(payload[int(index / 8):int(index / 8 + 2)], byteorder='big',
#                                            signed=True) >> remainder) & 0b11111111
#                 real = real_bin
#                 imag_bin = bytes([(payload[int(index / 8 + 1)] >> remainder) | (
#                         payload[int(index / 8 + 2)] << (8 - remainder)) & 0b11111111])
#                 imag = int.from_bytes(imag_bin, byteorder='little', signed=True)
#                 tmp = complex(float(real), float(imag))
#                 csi[i, j, k] = tmp
#                 index += 16
#     return csi


@jit(nopython=True)
def parse_csi(payload, Ntx, Nrx):
    # Compute CSI from all this crap
    csi = np.zeros(shape=(Ntx, Nrx, 30), dtype=np.dtype(np.complex64))
    index = 0

    for i in np.arange(30):
        index += 3
        remainder = index % 8
        for j in np.arange(Nrx):
            for k in np.arange(Ntx):
                start = index // 8
                
                # 原写法（不能numba加速,慢）
                # real_bin = bytes([(payload[start] >> remainder) | (payload[start + 1] << (8 - remainder)) & 0b11111111])
                # real = int.from_bytes(real_bin, byteorder='little', signed=True)
                # imag_bin = bytes([(payload[start + 1] >> remainder) | (payload[start + 2] << (8 - remainder)) & 0b11111111])
                # imag = int.from_bytes(imag_bin, byteorder='little', signed=True)
                # tmp = complex(float(real), float(imag))
                # 优化写法（支持numba加速）
                real_bin = (payload[start] >> remainder) | (payload[start + 1] << (8 - remainder)) & 0b11111111
                real=real_bin if real_bin<128 else real_bin-256
                imag_bin = (payload[start + 1] >> remainder) | (payload[start + 2] << (8 - remainder)) & 0b11111111
                imag=imag_bin if imag_bin<128 else imag_bin-256
                tmp = real+imag*1j
                
                csi[k, j, i] = tmp
                index += 16
    return csi

def get_scale_csi(csi_st):
    return get_scale_csi2(csi_st["csi"],csi_st["noise"],csi_st["Nrx"],
                          csi_st["Ntx"],csi_st["rssi_a"],csi_st["rssi_b"],
                          csi_st["rssi_c"],csi_st["agc"])

@jit(nopython=True)
def get_scale_csi2(csi,noise,Nrx,Ntx,rssi_a,rssi_b,rssi_c,agc):
    # Pull out csi
    # Calculate the scale factor between normalized CSI and RSSI (mW)
    csi_sq = np.multiply(csi, np.conj(csi)).real
    csi_pwr = np.sum(csi_sq, axis=0)
    csi_pwr = csi_pwr.reshape(1, csi_pwr.shape[0], -1)
    rssi_pwr = dbinv(get_total_rss(rssi_a,rssi_b,rssi_c,agc))
    
    # # 原写法（除数为0问题未解决）
    scale = rssi_pwr / (csi_pwr / 30)
    
    # 修订方案一：忽略错误，将inf和nan改为0
    # with np.errstate(divide='ignore', invalid='ignore'):
        # scale = rssi_pwr / (csi_pwr / 30)
        # scale[scale == - np.inf] = 0
        # scale[scale == np.inf] = 0
        # scale[scale == np.nan] = 0

    # 修订方案二：使用np除法，将默认答案直接设为0
    # csi_pwr = csi_pwr / 30
    # scale = np.divide(rssi_pwr, csi_pwr, out=np.zeros_like(csi_pwr), where=csi_pwr != 0)

    if noise == -127:
        noise_db = -92
    else:
        noise_db = noise
    thermal_noise_pwr = dbinv(noise_db)

    quant_error_pwr = scale * (Nrx * Ntx)

    total_noise_pwr = thermal_noise_pwr + quant_error_pwr
    ret = csi * np.sqrt(scale / total_noise_pwr)
    if Ntx == 2:
        ret = ret * math.sqrt(2)
    elif Ntx == 3:
        ret = ret * math.sqrt(dbinv(4.5))
    return ret.astype(np.complex64)


@jit(nopython=True)
def db(X, U):
    R = 1
    if 'power'.startswith(U):
        assert X >= 0
    else:
        X = math.pow(abs(X), 2) / R

    return (10 * math.log10(X) + 300) - 300

@jit(nopython=True)
def dbinv(x):
    return math.pow(10, x / 10)

@jit(nopython=True)
def get_total_rss(rssi_a,rssi_b,rssi_c,agc):
    # Careful here: rssis could be zero
    rssi_mag = 0
    if rssi_a != 0:
        rssi_mag = rssi_mag + dbinv(rssi_a)
    if rssi_b != 0:
        rssi_mag = rssi_mag + dbinv(rssi_b)
    if rssi_c != 0:
        rssi_mag = rssi_mag + dbinv(rssi_c)
    return db(rssi_mag, 'power') - 44 - agc
from numba import jit, njit, int32
from numba.typed import List
import numpy as np
import math

# old method
# def read_bf_file(filename, maxlen=0):
#     with open(filename, "rb") as f:
#         bfee_list = List()
#         field_len = int.from_bytes(f.read(2), byteorder='big', signed=False)
#         while field_len != 0:
#             arr = f.read(field_len)
#             if len(arr) == field_len:  # 读取的数据长度与预期相等才保留
#                 bfee_list.append(arr)
#             field_len = int.from_bytes(f.read(2), byteorder='big', signed=False)
#             # 限制读取长度
#             if maxlen != 0 and len(bfee_list) >= maxlen:
#                 break
#     return analyse_bf(bfee_list)

# read all the file and then split with numba
def read_bf_file(path):
    with open(path, mode='rb') as f:
        byte_array = f.read()
    # byte_array=np.fromfile(path, dtype=np.uint8,offset=0)
    bfee_list = split_bytes(byte_array)
    return analyse_bf(bfee_list)


@njit(cache=True)
def split_bytes(byte_array):
    p = 0
    bfee_list = List()
    field_len = bytes2int(byte_array[p:p + 2], "big")
    while field_len != 0 and p + field_len <= len(byte_array):
        bfee_list.append(byte_array[p + 2:p + 2 + field_len])
        p += field_len + 2
        field_len = bytes2int(byte_array[p:p + 2], "big")
    return bfee_list


@jit(nopython=True, cache=True)
def analyse_bf(bfee_list):
    dicts = List()
    csis = List()
    count = 0  # % Number of records output
    broken_perm = 0  # % Flag marking whether we've encountered a broken CSI yet
    triangle = [0, 1, 3]  # % What perm should sum to for 1,2,3 antennas
    # triangle = [1, 3, 6]
    for array in bfee_list:
        # % Read size and code
        code = array[0]

        # there is CSI in field if code == 187，If unhandled code skip (seek over) the record and continue
        if code != 187:
            # % skip all other info
            continue

        # get beamforming or phy data
        count = count + 1
        # timestamp_low = int.from_bytes(array[1:5], byteorder='little', signed=False)
        # bfee_count = int.from_bytes(array[5:7], byteorder='little', signed=False)
        timestamp_low = bytes2int(array[1:5], byteorder='little')
        bfee_count = bytes2int(array[5:7], byteorder='little')

        Nrx = int32(array[9])
        Ntx = int32(array[10])
        rssi_a = int32(array[11])
        rssi_b = int32(array[12])
        rssi_c = int32(array[13])
        noise = int32(array[14]) - 256
        agc = int32(array[15])
        antenna_sel = int32(array[16])
        # b_len = int.from_bytes(array[17:19], byteorder='little', signed=False)
        # fake_rate_n_flags = int.from_bytes(array[19:21], byteorder='little', signed=False)
        b_len = bytes2int(array[17:19], byteorder='little')
        fake_rate_n_flags = bytes2int(array[19:21], byteorder='little')

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
        # print(type(payload[0]))
        # Compute CSI from all this crap :
        csi = parse_csi(payload, Ntx, Nrx)

        # print(perm)
        # % matrix does not contain default values
        if sum(perm) != triangle[Nrx - 1]:
            print('WARN ONCE: Found CSI with Nrx=', Nrx, ' and invalid perm=[', perm, ']\n')
            continue
        else:
            # csi[:, perm, :] = csi[:, (0, 1, 2), :]
            csi[:, perm[0], :], csi[:, perm[1], :], csi[:, perm[2], :], \
                = csi[:, 0, :].copy(), csi[:, 1, :].copy(), csi[:,2,:].copy()

        # dict,and return
        bfee_dict = (
            timestamp_low,
            bfee_count,
            noise,
            Nrx,
            Ntx,
            rssi_a,
            rssi_b,
            rssi_c,
            agc,
            antenna_sel,
            perm,
            b_len,
            fake_rate_n_flags,
            calc_len,
        )
        dicts.append(bfee_dict)
        csis.append(csi)
        # break
    return dicts, csis


# uint8 to int
@jit(nopython=True, cache=True)
def bytes2int(byt, byteorder='little'):
    d = 0
    if byteorder == 'little':
        for item in byt[::-1]:
            d = (d << 8) + item
    elif byteorder == 'big':
        for item in byt[::1]:
            d = (d << 8) + item
    return d


@jit(nopython=True, cache=True)
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
                real_bin = (payload[start] >> remainder) | (payload[start + 1] << (8 - remainder)) & 0b11111111
                real = real_bin if real_bin < 128 else real_bin - 256
                imag_bin = (payload[start + 1] >> remainder) | (payload[start + 2] << (8 - remainder)) & 0b11111111
                imag = imag_bin if imag_bin < 128 else imag_bin - 256
                tmp = real + imag * 1j
                csi[k, j, i] = tmp
                index += 16
    return csi


@jit(nopython=True, cache=True)
def get_scale_csi(csis, params):
    res = []
    for i in np.arange(len(csis)):
        csi = csis[i]
        param = params[i]
        noise, Nrx, Ntx, rssi_a, rssi_b, rssi_c, agc = param[2:9]
        # Pull out csi
        # Calculate the scale factor between normalized CSI and RSSI (mW)
        csi_sq = np.multiply(csi, np.conj(csi)).real
        csi_pwr = np.sum(csi_sq, axis=0)
        csi_pwr = csi_pwr.reshape(1, csi_pwr.shape[0], -1)
        rssi_pwr = dbinv(get_total_rss(rssi_a, rssi_b, rssi_c, agc))

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
        # return ret
        res.append(ret.astype(np.complex64))
    return res


@jit(nopython=True, cache=True)
def db(X, U):
    R = 1
    if 'power'.startswith(U):
        assert X >= 0
    else:
        X = math.pow(abs(X), 2) / R

    return (10 * math.log10(X) + 300) - 300


@jit(nopython=True, cache=True)
def dbinv(x):
    return math.pow(10, x / 10)


@jit(nopython=True, cache=True)
def get_total_rss(rssi_a, rssi_b, rssi_c, agc):
    # Careful here: rssis could be zero
    rssi_mag = 0
    if rssi_a != 0:
        rssi_mag = rssi_mag + dbinv(rssi_a)
    if rssi_b != 0:
        rssi_mag = rssi_mag + dbinv(rssi_b)
    if rssi_c != 0:
        rssi_mag = rssi_mag + dbinv(rssi_c)
    return db(rssi_mag, 'power') - 44 - agc

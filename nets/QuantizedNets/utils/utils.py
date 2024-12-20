import torch
import copy
import numpy as np


def filter_state_dict_keys(sd, endswith_key_string):
    keys = [i for i in sd.keys() if i.endswith(endswith_key_string)]
    if not keys:
        raise KeyError(endswith_key_string)
    return sd[keys[0]]


def tensor_scale(input):
    scale = float(2*torch.max(torch.abs(torch.max(input)),
                              torch.abs(torch.min(input))))/127.0
    return scale


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return conv_w, conv_b


def prepack_conv2d(qweight, bias, stride, padding, groups=1):
   return torch.ops.quantized.conv2d_prepack(qweight, bias, stride=[stride, stride], dilation=[1, 1],
                                              padding=[padding, padding], groups=groups)


def prepack_linear(qweight, qbias=None):
    if qbias is not None:
        return torch.ops.quantized.linear_prepack(qweight, qbias)
    else:
        return torch.ops.quantized.linear_prepack(qweight)


def prepack_conv2d_transpose(qweight, bias, stride, padding, groups=1):
    if stride == 2:
        out_padding = 1
    else:
        out_padding = 0
    return torch.ops.quantized.conv_transpose2d_prepack(qweight, bias, stride=[stride, stride],
                                                        dilation=[1, 1], padding=[padding, padding], output_padding=[out_padding, out_padding], groups=groups)


def filter_table(resources, table, n_blocks):
    table = copy.deepcopy(table)

    max_time = resources.get_time()
    max_data = resources.get_data()
    max_memory = resources.get_memory()

    table = list(sorted(table, key=lambda x: x['time'], reverse=True))
    table = list(filter(lambda x: x['time'] <= max_time, table))
    table = list(filter(lambda x: x['data'] <= max_data, table))
    table = list(filter(lambda x: x['memory'] <= max_memory, table))

    out_configs = []
    # find subsets
    for config in table:
        is_subset = False
        for config2 in table:
            train1 = [i for i in range(0, n_blocks) if i not in config['freeze']]
            train2 = [i for i in range(0, n_blocks) if i not in config2['freeze']]

            if train1 != train2 and set(train1).issubset(set(train2)):
                    is_subset = True
                    break
        if not is_subset:
            out_configs.append(config['freeze'])
    assert out_configs, "[ERROR]: No configuration available for requested resource constraint."
    return out_configs

def filter_unit_max(unit, table):
    table = copy.deepcopy(table)
    table = list(filter(lambda x: unit == x['unit'], table))
    max_entry = np.zeros(3)
    max_entry[0] = max(table, key=lambda x: x['time'])['time']
    max_entry[1] = max(table, key=lambda x: x['data'])['data']
    max_entry[2] = max(table, key=lambda x: x['memory'])['memory']
    return max_entry

def filter_table_unit(unit,table):
    table = copy.deepcopy(table)
    out_configs = list(filter(lambda x: unit == x['unit'], table))
    out_configs = [ config['freeze'] for config in out_configs ]
    return out_configs

def get_units(table):
    table = copy.deepcopy(table)
    units = {item['unit'] for item in table}
    units_list = list(units)
    return units_list

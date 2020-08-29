
import numpy as np

# for pruning
def acc_reward(net, acc, flops,pre_ex,pre_act):
    return acc * 0.01


def acc_flops_reward(net, acc, flops,pre_ex,pre_act):
    error = (100 - acc) * 0.01
    return -error * np.log(flops)


def acc_pres_reward(net, acc, flops,pre_ex,pre_act):
    # error = (100 - acc) * 0.01
    pre_achieve = pre_act/pre_ex
    return acc * 0.01 * 0.9 + (1-abs(pre_achieve-1))*0.1

    
def acc_normalize(net, acc, flops,pre_ex,pre_act):
    acc_exp = None
    if pre_ex==0.3:
        acc_exp = 0.09
    elif pre_ex ==0.5:
        acc_exp = 0.73
    elif pre_ex==0.7:
        acc_exp = 0.91
    else:
        raise AssertionError
    # error = (100 - acc) * 0.01

    return acc* 0.01 /acc_exp

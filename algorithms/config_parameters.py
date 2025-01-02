import math

cadi_para = {"const_c": 0.8, "sigma": 0.05}

armNoiseScale = 0.1
suparmNoiseScale = 0.1
batch_size = 50
sampling = "optimal_greedy"
seeds_set = [2756048, 675510, 807110, 216-151, 9492253, 927, 218, 495, 515, 452]


def my_lambda(t):
    return 5 * int(math.log(t + 1))

bt = my_lambda



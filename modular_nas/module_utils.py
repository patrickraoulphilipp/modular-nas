from nasbench import api
import numpy as np
from modular_nas.config import *
import random

def sample_modules(num_samples):

    nasbench = api.NASBench(PATH_TO_TFRECORD)

    INPUT = 'input'
    OUTPUT = 'output'
    CONV3X3 = 'conv3x3-bn-relu'
    CONV1X1 = 'conv1x1-bn-relu'
    MAXPOOL3X3 = 'maxpool3x3'

    data = []
    modules = []
    num_bin_choices = 21
    bin_choices = [0, 1]
    num_layers = 5
    layers = [CONV3X3, CONV1X1, MAXPOOL3X3]

    while len(modules) < num_samples:
        try:
            mds = np.random.choice(bin_choices, num_bin_choices,
                                     p=[.5 for i in range(len(bin_choices))])
            ops_draws = np.random.choice(layers, num_layers,
                                     p=[1./3. for i in range(len(layers))])

            model_spec = api.ModelSpec(
              matrix=[[0, mds[0], mds[1], mds[2], mds[3], mds[4], mds[5]], # input
                      [0, 0, mds[6], mds[7], mds[8], mds[9], mds[10]],
                      [0, 0, 0, mds[11], mds[12], mds[13], mds[14]],
                      [0, 0, 0, 0, mds[15], mds[16], mds[17]],
                      [0, 0, 0, 0, 0, mds[18], mds[19]],
                      [0, 0, 0, 0, 0, 0, mds[20]],
                      [0, 0, 0, 0, 0, 0, 0]],                              # output
              ops=[INPUT,
                    ops_draws[0],
                    ops_draws[1],
                    ops_draws[2],
                    ops_draws[3],
                    ops_draws[4],
                    OUTPUT])

            data.append(nasbench.query(model_spec))  #to check if it is valid

            modules = modules + [model_spec]
        except:
            pass

    return modules, data

def generate_modules(
            num_modules=None, 
            ranked=None,
            num_samples=None
):

    modules, data = sample_modules(num_samples=num_samples)
    
    if num_modules is not None:
        return_modules = []
        return_data = []

        cache = None
        if not ranked:
            cache = random.choices(range(len(modules)),k=num_modules)
            for i in cache:
                return_modules.append(modules[i])
                return_data.append(data[i])
        else:
            accs = [m['test_accuracy'] for m in data]
            import heapq
            best = heapq.nlargest(num_modules, accs)
            return_data, return_modules = [], []
            for i, d in enumerate(data):
                if d['test_accuracy'] in best:
                    return_data = return_data + [d]
                    return_modules = return_modules + [modules[i]]

    else:
        return_modules = modules
        return_data = data

    return return_modules, return_data


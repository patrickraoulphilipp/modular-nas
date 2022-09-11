from keras.layers import Conv2D, MaxPooling2D, concatenate, BatchNormalization, Activation, add

from copy import deepcopy
import numpy as np

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

def create_model(matrix, ops, filter_size, input, iteration):
    # print("setting filter size::", filter_size)
    ops_iter = deepcopy(ops)
    del ops_iter[0]
    del ops_iter[-1]

    layers = {0: input}

    names = []
    name_counter = 0

    for l, layer in enumerate(ops_iter):
        if layer == "conv3x3-bn-relu":
            if np.count_nonzero(matrix[:, l+1]) > 1:
                adjusted_layers_indexes = [i for i in np.concatenate(np.argwhere(matrix[:, l+1] == 1)).ravel().tolist()]
                a_shapes = []
                a_dict = dict()
                for i in adjusted_layers_indexes:
                    sh = None
                    try:
                        sh = layers[i].output_shape[-1]
                    except:
                        sh = layers[i].shape[-1]
                    a_dict[i] = sh
                    a_shapes.append(int(sh))
                max_output_shape = min(a_shapes) if min(a_shapes) >= filter_size else filter_size
                if len(set(a_shapes)) > 1:
                    for i in adjusted_layers_indexes:
                        if a_dict[i] != max_output_shape:
                            layers[i] = Conv2D(max_output_shape, kernel_size=1, padding='same', name="layer_{}_{}".format(iteration, name_counter))(layers[i])
                            names.append("layer_{}_{}".format(iteration, name_counter))
                            name_counter += 1
                            layers[i] = BatchNormalization(name="layer_{}_{}".format(iteration, name_counter))(layers[i])
                            names.append("layer_{}_{}".format(iteration, name_counter))
                            name_counter += 1
                            layers[i] = Activation('relu')(layers[i])                        
                adjusted_layers = [layers[i] for i in adjusted_layers_indexes]
                layers["{}-merge".format(l + 1)] = (add(adjusted_layers))
                layers[l + 1] = Conv2D(filter_size, kernel_size=3, activation='relu', padding='same', name="layer_{}_{}".format(iteration, name_counter))(layers["{}-merge".format(l + 1)])
                names.append("layer_{}_{}".format(iteration, name_counter))
                name_counter += 1
            else:
                layers[l + 1] = Conv2D(filter_size, kernel_size=3, activation='relu', padding='same', name="layer_{}_{}".format(iteration, name_counter))(layers[int(np.argwhere(matrix[:, l+1] == 1))])
                names.append("layer_{}_{}".format(iteration, name_counter))
                name_counter += 1
        elif layer == "conv1x1-bn-relu":
            if np.count_nonzero(matrix[:, l+1]) > 1:
                adjusted_layers_indexes = [i for i in np.concatenate(np.argwhere(matrix[:, l+1] == 1)).ravel().tolist()]
                a_shapes = []
                a_dict = dict()
                for i in adjusted_layers_indexes:
                    sh = None
                    try:
                        sh = layers[i].output_shape[-1]
                    except:
                        sh = layers[i].shape[-1]
                    a_dict[i] = sh
                    a_shapes.append(int(sh))
                max_output_shape = min(a_shapes) if min(a_shapes) >= filter_size else filter_size
                if len(set(a_shapes)) > 1:
                    for i in adjusted_layers_indexes:
                        if a_dict[i] != max_output_shape:
                            layers[i] = Conv2D(max_output_shape, kernel_size=1, padding='same', name="layer_{}_{}".format(iteration, name_counter))(layers[i])
                            names.append("layer_{}_{}".format(iteration, name_counter))
                            name_counter += 1
                            layers[i] = BatchNormalization(name="layer_{}_{}".format(iteration, name_counter))(layers[i])
                            names.append("layer_{}_{}".format(iteration, name_counter))
                            name_counter += 1
                            layers[i] = Activation('relu')(layers[i])

                adjusted_layers = [layers[i] for i in adjusted_layers_indexes]
                layers["{}-merge".format(l + 1)] = (add(adjusted_layers))
                layers[l + 1] = Conv2D(filter_size, kernel_size=1, activation='relu', padding='same', name="layer_{}_{}".format(iteration, name_counter))(layers["{}-merge".format(l + 1)])
                names.append("layer_{}_{}".format(iteration, name_counter))
                name_counter += 1
            else:
                layers[l + 1] = Conv2D(filter_size, kernel_size=1, activation='relu', padding='same', name="layer_{}_{}".format(iteration, name_counter))(layers[int(np.argwhere(matrix[:, l+1] == 1))])
                names.append("layer_{}_{}".format(iteration, name_counter))
                name_counter += 1
        elif layer == "maxpool3x3":
            if np.count_nonzero(matrix[:, l+1]) > 1:
                adjusted_layers_indexes = [i for i in np.concatenate(np.argwhere(matrix[:, l+1] == 1)).ravel().tolist()]
                a_shapes = []
                a_dict = dict()
                for i in adjusted_layers_indexes:
                    sh = None
                    try:
                        sh = layers[i].output_shape[-1]
                    except:
                        sh = layers[i].shape[-1]
                    a_dict[i] = sh
                    a_shapes.append(int(sh))
                max_output_shape = min(a_shapes) if min(a_shapes) >= filter_size else filter_size
                if len(set(a_shapes)) > 1:
                    for i in adjusted_layers_indexes:
                        if a_dict[i] != max_output_shape:
                            layers[i] = Conv2D(max_output_shape, kernel_size=1, padding='same', name="layer_{}_{}".format(iteration, name_counter))(layers[i])
                            names.append("layer_{}_{}".format(iteration, name_counter))
                            name_counter += 1
                            layers[i] = BatchNormalization(name="layer_{}_{}".format(iteration, name_counter))(layers[i])
                            names.append("layer_{}_{}".format(iteration, name_counter))
                            name_counter += 1
                            layers[i] = Activation('relu')(layers[i])  
                adjusted_layers = [layers[i] for i in adjusted_layers_indexes]
                layers["{}-merge".format(l + 1)] = (add(adjusted_layers ))
                layers[l + 1] = MaxPooling2D(pool_size=(3, 3), padding='same', strides = (1,1))(layers["{}-merge".format(l + 1)])
            else:
                prior = layers[int(np.argwhere(matrix[:, l+1] == 1))]
                layers[l + 1] = MaxPooling2D(pool_size=(3, 3), padding='same', strides = (1,1))(prior)
    output = None
    p_layers = [layers[i] for i in np.concatenate(np.argwhere(matrix[:, len(ops) - 1] == 1)).ravel().tolist()]
    if len(p_layers) > 1:
        layers[len(ops) - 1] = concatenate(p_layers)

        output = layers[len(ops) - 1]
    else:
        output = p_layers[0]
    return output, names
import keras
import numpy as np
from keras.layers import Activation, BatchNormalization
from keras.layers import Conv2D
from keras.regularizers import l2

import modular_nas.keras_transform as kt

class Module:

    def __init__(
            self, 
            max_module_count, 
            min_module_count, 
            module_count,
            probability_mutation, 
            filters, 
            modules, 
            data, 
            structure=None
    ):

        self.max_module_count = max_module_count
        self.min_module_count = min_module_count
        self.module_count = module_count
        self.probability_mutation = probability_mutation
        self.filters = filters
        self.modules = modules
        self.data = data
        self.num_possible_modules = len(modules)
        self.structure = [module_count]

        if structure is None:
            self.structure = []
            for i in range(module_count):
                self.structure.append(np.random.randint(0, self.num_possible_modules))

        else:
            self.structure = structure

    def add_module(self, x, iterator_modules, itName):
        stage = iterator_modules 
        res_block = itName

        # the following case is true if the model mutation added another module
        if (iterator_modules >= len(self.structure)):
            print("new Module added")
            self.structure.append(np.random.randint(0, self.num_possible_modules))

        i = self.structure[iterator_modules]
        # print("Data i=", i, self.data[i])

        num_filters_in = self.filters

        activation = 'relu'
        batch_normalization = True
        strides = 1
        num_filters_out = num_filters_in
        if stage == 0:
            if res_block == 0:
                activation = None
                batch_normalization = False
                num_filters_out = num_filters_in * 2
        else:
            if res_block == 0:
                num_filters_out = num_filters_in * 2
                strides = 2

        y, names1 = resnet_layer(
                         inputs=x,
                         num_filters=num_filters_in,
                         kernel_size=1,
                         strides=strides,
                         activation=activation,
                         batch_normalization=batch_normalization,
                         conv_first=False,
                         iteration=iterator_modules,
                         position=1,
                         stacked=itName
                    )
        y, names11 = transformed_layer(
                            inputs=y,
                            iteration=iterator_modules,
                            stacked=itName,
                            num_filters=num_filters_in,
                            conv_first=False,
                            module_data=self.data[i]
                     )
        y, names2 = resnet_layer(
                         inputs=y,
                         num_filters=num_filters_out,
                         kernel_size=1,
                         conv_first=False,
                         iteration=iterator_modules,
                         stacked=itName,
                         position=2
                    )
        names3 = None
        if res_block == 0:
            x, names3 = resnet_layer(
                             inputs=x,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             strides=strides,
                             activation=None,
                             batch_normalization=False,
                             iteration=iterator_modules,
                             stacked=itName,
                             position=3
                         )
     
        x = keras.layers.add([x, y])
        self.filters = num_filters_out
        names = []
        names.extend(names11)
        names.extend(names1)
        names.extend(names2)
        if names3 is not None:
            names.extend(names3)
        
        return x, names, num_filters_in, i

    def mutate_architecture(
                self, 
                cache_structure
    ):
        self.structure = cache_structure

        # choice wheter a Module gets changed or the number of Modules get changed
        mut = np.random.choice(np.arange(1, 3), p=(self.probability_mutation, (1 - self.probability_mutation)))

        if (mut == 1):
            """change one number inside the structure array, so that only one Module gets changed to a new Module"""
            k = np.random.randint(0, len(self.structure))
            self.structure[k] = np.random.randint(0, self.num_possible_modules)
            print("!#!#!#!#!# changed to ", self.structure, " based on k=", k, " and max=", len(self.structure) - 1)
      
        elif (mut == 2):
            """changes number of Modules"""
            i = np.random.randint(1, 3)

            if (i == 1) and (len(self.structure) > self.min_module_count):
                self.structure.pop()
            if (i == 2) and (len(self.structure) < self.max_module_count):
                self.structure.append(np.random.randint(0, self.num_possible_modules))

        return self.structure

    def getStructure(self):
        return self.structure


def resnet_layer(
            inputs,
            num_filters=16,
            kernel_size=3,
            strides=1,
            activation='relu',
            batch_normalization=True,
            conv_first=True,
            iteration=None,
            stacked=None,
            position=None
):
    layername = "layer_conv_resnet_" + str(iteration) + "_" + str(stacked) + "_" + str(position)
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),
                  name=layername)

    did_batch = False
    batch_name = "layer_batch_resnet_" + str(iteration) + "_" + str(stacked) + "_" + str(position)
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(name=batch_name)(x)
            did_batch = True
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization(name=batch_name)(x)
            did_batch = True
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    names = [layername] if not did_batch else [layername, batch_name]

    return x, names

def transformed_layer(
            inputs,
            num_filters=16,
            iteration=None,
            stacked=None,
 #           kernel_size=3,
 #           strides=1,
            activation='relu',
            batch_normalization=True,
            conv_first=True,
            module_data=None
):
    
    did_batch = False
    batch_name = str(iteration) + "_" + str(stacked) + "_" + "batch"
    x = inputs
    if conv_first:
        x, names = kt.create_model(
                            module_data['module_adjacency'], 
                            module_data['module_operations'], 
                            num_filters, 
                            x, 
                            str(iteration) + "_" + str(stacked)
                   )
        if batch_normalization:
            x = BatchNormalization(name=batch_name)(x)
            did_batch = True
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization(name=batch_name)(x)
            did_batch = True
        if activation is not None:
            x = Activation(activation)(x)
        x, names = kt.create_model(
                            module_data['module_adjacency'], 
                            module_data['module_operations'], 
                            num_filters, 
                            x, 
                            str(iteration) + "_" + str(stacked)
                   )
    if did_batch:
      names.append(batch_name)
    return x, names

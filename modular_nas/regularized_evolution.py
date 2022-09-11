import copy
from copy import deepcopy
import collections
from errno import EPIPE
import random
import numpy as np

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
from keras import backend as K

from modular_nas.architecture import Architecture
from modular_nas.weight_sharing import do_weight_sharing
from modular_nas.config import *

# helper class to keep track of training results
class MutatedModel:
    
    def __init__(
            self,
            arch,
    ):
        self.graph = tf.get_default_graph()
        self.arch = arch

        self.accuracy = None
        self.loss = None
        self.population_number = None
        self.mutation_number = None

    def train_and_eval(
                self, 
                y_train, 
                y_test, 
                x_train, 
                x_test, 
                structure_history, 
                final=False
    ):
        with self.graph.as_default():
            if tuple(self.arch.module.structure) not in structure_history:
                self.accuracy, self.loss = self.arch.train_and_eval(
                                                    y_train, 
                                                    y_test, 
                                                    x_train, 
                                                    x_test, 
                                                    final
                                           )
            else:
                self.accuracy = structure_history[tuple(self.arch.module.structure)]

        return self.accuracy

    def random_architecture(self):
        self.arch.random_architecture()

    def mutate_arch(
                self, 
                parent_arch
    ):
        child_arch = self.arch.mutate_architecture(parent_arch)
        return child_arch


def regularized_evolution(
                cycles, 
                population_size, 
                sample_size, 
                num_classes, 
                epochs, 
                max_module_count, 
                min_module_count,
                inital_module_count, 
                probability_mutation, 
                filters, 
                verbose, 
                x_train, 
                y_train, 
                x_test, 
                y_test,
                batch_size, 
                modules, 
                data
):

    population = collections.deque()
    history = [] 

    structure_history = dict()

    population_Count = 1
    mutation_count = 0

    while len(population) < population_size:

        arch = Architecture(
                        num_classes, 
                        epochs, 
                        max_module_count, 
                        min_module_count,
                        inital_module_count, 
                        probability_mutation, 
                        filters, 
                        verbose, 
                        x_train, 
                        y_train, 
                        x_test, 
                        y_test, 
                        batch_size,
                        modules, 
                        data
                )
        model = MutatedModel(arch) #maybe need to deepcopy

        model.population_number = population_Count
        model.mutation_number = mutation_count
        model.arch.set_count(model.population_number, model.mutation_number)

        while model.accuracy is None:

            model.random_architecture()
            
            # do weight sharing if architecture is unknown and option is turned on
            model = do_weight_sharing(
                            mutated_model=model, 
                            structure_history=structure_history,
                            weight_mode=eval_weight_mode,
                            sliding_windows=sliding_windows,
                            history=history,
                            default_epochs=epochs,
                            thresholds=thresholds
                    )

            model.accuracy = model.train_and_eval(
                                        y_train, 
                                        y_test, 
                                        x_train, 
                                        x_test, 
                                        structure_history
                             )
        
        if tuple(model.arch.module.structure) not in structure_history:
            structure_history[tuple(model.arch.module.structure)] = model.accuracy
            population.append(model)
            history.append(model)
            population_Count += 1
            local_dict = dict()
            for layer in model.arch.model.layers:
                local_dict[layer.name] = model.arch.model.get_layer(layer.name).get_weights()
            model.arch.model_dict = local_dict
            K.clear_session() 

    while len(history) < cycles:

        print("# cycles: {}/{}".format(len(history), cycles))
        population_Count += 1
        mutation_count += 1

        sample = random.sample(list(population), sample_size)
        parent = max(sample, key=lambda i: i.accuracy)
        max_acc = max([m.accuracy for m in history])
        print("---> current max acc=", max_acc)

        arch = Architecture(
                num_classes, 
                epochs, 
                max_module_count, 
                min_module_count,
                inital_module_count, 
                probability_mutation, 
                filters, 
                verbose, 
                x_train, 
                y_train, 
                x_test, 
                y_test, 
                batch_size,
                modules, 
                data
        )
        child = MutatedModel(arch)

        cache_structure = copy.deepcopy(parent.arch.module.structure)
        child.mutation_number = mutation_count
        child.population_number = population_Count
        child.arch.set_count(child.population_number, child.mutation_number)
      
        while child.accuracy is None:
            print("### parent:", cache_structure)         
            child.arch = child.mutate_arch(cache_structure)
            print("########## child:", child.arch.module.structure)
           
            try:
                # do weight sharing if architecture is unknown and option is turned on
                child = do_weight_sharing(
                            mutated_model=child, 
                            structure_history=structure_history,
                            weight_mode=eval_weight_mode,
                            sliding_windows=sliding_windows,
                            history=history,
                            default_epochs=epochs,
                            thresholds=thresholds
                    )

                child.accuracy = child.train_and_eval(
                                            y_train, 
                                            y_test,
                                            x_train, 
                                            x_test, 
                                            structure_history
                                 )
            except Exception as e:
                print("Catch Exception New Model: ", e)
                arch = Architecture(
                        num_classes, 
                        epochs, 
                        max_module_count, 
                        min_module_count,
                        inital_module_count, 
                        probability_mutation, 
                        filters, 
                        verbose, 
                        x_train, 
                        y_train, 
                        x_test, 
                        y_test, 
                        batch_size,
                        modules, 
                        data
                )
             
                child = MutatedModel(arch)
                child.mutation_number = mutation_count
                child.population_number = population_Count
                child.arch.set_count(child.population_number, child.mutation_number)
        
        if tuple(child.arch.module.structure) not in structure_history:
            structure_history[tuple(child.arch.module.structure)] = child.accuracy
            population.append(child)
            history.append(child)
            population.popleft()
            local_dict = dict()
            for layer in child.arch.model.layers:
                local_dict[layer.name] = child.arch.model.get_layer(layer.name).get_weights()
            child.arch.model_dict = local_dict
            K.clear_session() 

    max_model_indice = np.argmax([m.accuracy for m in history])
    max_model = history[max_model_indice]
    max_model.arch.epochs = eval_epochs
    print(">Training best model; {}".format(max_model.arch.module.structure))
    arch = Architecture(
        num_classes, 
        epochs, 
        max_module_count, 
        min_module_count,
        inital_module_count, 
        probability_mutation, 
        filters, 
        verbose, 
        x_train, 
        y_train, 
        x_test, 
        y_test, 
        batch_size,
        modules, 
        data
)
    new_model = MutatedModel(arch)
    new_model.population_number = population_Count
    new_model.mutation_number = mutation_count
    new_model.arch.set_count(new_model.population_number, new_model.mutation_number)
    max_structure = max_model.arch.module.structure
    new_model.arch.fixed_architecture(max_structure)
    accuracy = new_model.train_and_eval(
                            y_train, 
                            y_test, 
                            x_train, 
                            x_test, 
                            [], 
                            final=True
                )
    print("---> test acc::", accuracy)

    return history

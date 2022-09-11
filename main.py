import random

from modular_nas import data_loading, module_utils, regularized_evolution
from modular_nas.config import *

if __name__ == '__main__':        
   
    random.seed(seed)
    modules, module_performance_data = module_utils.generate_modules(
                                                        num_modules=num_modules, 
                                                        ranked=eval_ranked,
                                                        num_samples=eval_num_samples
                                        )

    (x_train, y_train), (x_test, y_test) = data_loading.load_data(eval_data_set)
    eval_num_classes = data_loading.get_num_classes(eval_data_set)

    history = regularized_evolution.regularized_evolution(
                        eval_cycles,
                        eval_population_size, 
                        eval_sample_size, 
                        eval_num_classes, 
                        eval_epochs, 
                        eval_max_module_count, 
                        eval_min_modules_count, 
                        eval_inital_module_count, 
                        eval_prob_mutation, 
                        eval_filters, 
                        eval_verbose, 
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        eval_batch_size,
                        modules,
                        module_performance_data
             )
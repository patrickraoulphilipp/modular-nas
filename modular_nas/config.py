DEBUG_EVOLUTION = False
PATH_TO_TFRECORD = "/PATH/TO/nasbench_only108.tfrecord"

# weight sharing thresholds for already trained modules
thresholds = dict()
thresholds[0] = 108
thresholds[1] = 60
thresholds[2] = 22
thresholds[4] = 10
sliding_windows = 5 #for weight averaging

seed = 42
num_modules = 5
num_stages = 3
num_res_blocks = 3

eval_data_set = "cifar10"
eval_num_samples = 10
eval_population_size = 4
eval_cycles = 10
eval_batch_size = 64
eval_epochs = 108
eval_max_module_count = 4
eval_min_modules_count = 4
eval_inital_module_count = 4
eval_prob_mutation = .8
eval_sample_size = 2
eval_filters = 64
eval_verbose = 2
eval_ranked = True
eval_weight_mode = "reuse"
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from copy import deepcopy

def weight_reuse(history, new_arch):
    avged = dict()
    for m, module in enumerate(new_arch.structure):
        weights_agg = []
        counter = 0
        for h_model in history:
            try:
                if module[0] == h_model.arch.structure[m][0]:
                    for n, name in enumerate(h_model.arch.structure[m][2]):
                        if counter == 0:
                            weights_agg.append([])
                            for weight in h_model.arch.model_dict[name]:
                                weights_agg[n].append(weight)
                        else:
                            for w, weight in enumerate(h_model.arch.model_dict[name]):
                                weights_agg[n][w] = weights_agg[n][w] + weight
                    counter += 1
                    break
                else:
                    pass
            except IndexError:
                pass
        avged[m] = counter
        if counter > 0:
            weights_avg = deepcopy(weights_agg)
            for n, name in enumerate(module[2]):
                # print("---> n=", name)
                for w, weight in enumerate(weights_agg[n]):
                    weights_avg[n][w] = weight # / counter
                new_arch.model.get_layer(name).set_weights(weights_avg[n])
                new_arch.model.get_layer(name).trainable = True
    return new_arch, avged


def weight_averaging(history, new_arch, sliding_window):
    avged = dict()
    for m, module in enumerate(new_arch.structure):
        weights_agg = []
        counter = 0
        reverse_history = history.copy()
        reverse_history.reverse()
        for h_model in reverse_history:
            try:
                if module[0] == h_model.arch.structure[m][0]:
                    for n, name in enumerate(h_model.arch.structure[m][2]):
                        if counter == 0:
                            weights_agg.append([])
                            for weight in h_model.arch.model.get_layer(name).get_weights():
                                weights_agg[n].append(weight)
                        else:
                            for w, weight in enumerate(h_model.arch.model.get_layer(name).get_weights()):
                                weights_agg[n][w] = weights_agg[n][w] + weight
                    counter += 1
                    if counter == sliding_window:
                        break
                else:
                    pass
            except IndexError:
                pass
        avged[m] = counter
        if counter > 0:
            weights_avg = deepcopy(weights_agg)
            for n, name in enumerate(module[2]):
                for w, weight in enumerate(weights_agg[n]):
                    weights_avg[n][w] = weight / counter
                new_arch.model.get_layer(name).set_weights(weights_avg[n])
                new_arch.model.get_layer(name).trainable = True
    return new_arch, avged

def do_weight_sharing(
            mutated_model=None, 
            structure_history=None,
            weight_mode=None,
            sliding_windows=None,
            history=None,
            default_epochs=None,
            thresholds=None
):
    if tuple(mutated_model.arch.module.structure) not in structure_history:
       
        if weight_mode is not None:
            avged = None
            if weight_mode == "average":
                mutated_model.arch, avged = weight_averaging(history, mutated_model.arch, sliding_windows)
            elif weight_mode == "reuse":
                mutated_model.arch, avged = weight_reuse(history, mutated_model.arch)

            # set epochs for training based on number of times the modules have been seen
            set_epochs = default_epochs
            ordered_thresholds = sorted(thresholds.keys(), reverse=True)
            for threshold in ordered_thresholds:
                if all(c >= threshold for c in avged.values()):
                    set_epochs = thresholds[threshold]
                    break
            mutated_model.arch.epochs = set_epochs
          
    return mutated_model
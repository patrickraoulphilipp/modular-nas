import numpy as np
import time

from keras.models import Model
from keras.layers import (
                            Dense, 
                            Activation, 
                            Input, 
                            Conv2D,
                            GlobalAveragePooling2D, 
                            BatchNormalization
)
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from modular_nas.module import Module
from modular_nas.config import *

ref = int(round(time.time()))

class Architecture():
    def __init__(
            self, 
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
            data, 
            pop_num=None,
            mut_num=None
    ):

        self.num_classes = num_classes
        self.epochs = epochs
        self.max_module_count = max_module_count
        self.min_module_count = min_module_count
        self.module_count = inital_module_count
        self.probability_module_mutation = probability_mutation
        self.filters = filters
        self.verbose = verbose
        self.num_mutations = None
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.modules = modules
        self.data = data
        self.population_number = pop_num
        self.mutation_number = mut_num

        self.structure = []

        self.input_shape = x_train.shape[1:]

        self.model = Model()
        self.layers = dict()
        self.input = Input(shape=x_train.shape[1:])
        self.previous_module, names = resnet_layer(
                                       inputs=self.input,
                                       num_filters=self.filters,
                                       conv_first=True,
                                       iteration=-1,
                                       position=-1
                                  )

        self.structure.append([-1, filters, names])

        self.module = self.make_module()
        self.predictions = None

    def fixed_architecture(
                self, 
                f_structure
    ):
        self.module = self.make_module(structure=f_structure)
        org_filters = self.filters
        self.build_architecture()
        self.filters = org_filters

    def random_architecture(self):
        org_filters = self.filters
        self.build_architecture()
        self.filters = org_filters
        pass

    def mutate_architecture(
                self, 
                cache_structure, 
                module_count=None
    ):
        mutatedArchitecture = self.module.mutate_architecture(cache_structure)
        self.module_count = len(mutatedArchitecture)
        if module_count is not None:
            self.module_count = module_count
        self.module = self.make_module(
                                structure=mutatedArchitecture
                       )
        org_filters = self.filters
        self.build_architecture()
        self.filters = org_filters
        return self

    def build_architecture(self):
        iterator_modules = 0
        structure = []
        while iterator_modules < self.module_count:
            itName = 0
            for i in range(num_stages):
                self.previous_module, \
                layer_names, \
                num_filters, \
                module_id = self.module.add_module(
                                        self.previous_module, 
                                        iterator_modules, 
                                        itName
                            )
                structure.append([module_id, num_filters, layer_names])
                itName += 1
            iterator_modules += 1

        final_batch_layer = 'final_batch_layer'
        x = BatchNormalization(name=final_batch_layer)(self.previous_module)
        x = Activation('relu')(x)
        y = GlobalAveragePooling2D()(x)
        output_layer_name = 'final_dense'
        output_layer = Dense(self.num_classes,
                             name=output_layer_name,
                             activation='softmax',
                             kernel_initializer='he_normal')(y)

        self.model = Model(inputs=self.input, outputs=output_layer)
        self.structure.extend(structure)
        self.structure.append([len(self.structure), self.filters, [final_batch_layer, output_layer_name]])

    def set_count(
            self, 
            pop_num, 
            mut_num
    ):
        self.population_number = pop_num
        self.mutation_number = mut_num

    def set_epochs(self, epochs):
        self.epochs = epochs

    def make_module(
                self, 
                structure=None
    ):
        m = Module(
                self.max_module_count, 
                self.min_module_count, 
                self.module_count,
                self.probability_module_mutation,
                self.filters, 
                self.modules, 
                self.data, 
                structure=structure
        )

        return m  

    def train_and_eval(
                self, 
                y_train, 
                y_test, 
                x_train, 
                x_test, 
                final=False
    ):
        lr_scheduler = None
        if final is False:
            lr_scheduler = LearningRateScheduler(lr_schedule)
        else:
            lr_scheduler = LearningRateScheduler(lr_schedule_final)

        lr_reducer = ReduceLROnPlateau(
                            factor=np.sqrt(0.1),
                            cooldown=0,
                            patience=5,
                            min_lr=0.5e-6
                     )

        earlystopper = EarlyStopping(
                                monitor='train_loss', 
                                min_delta=0.001, 
                                patience=3
                       )


        model = self.model
        model.compile(
                    optimizer=Adam(learning_rate=lr_schedule(0)),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
        )
        
        # model.summary()

        model_name = "pop-{}-mut-{}-time-{}".format(
                                                self.population_number, 
                                                self.mutation_number,
                                                (int(round((time.time()))) - ref)
                                             )
        print("current model trained: ", model_name)
        tensorboard = TensorBoard(log_dir="./logs/{}".format(model_name))
        callbacks = [earlystopper, lr_reducer, lr_scheduler, tensorboard]

        # debug evolutionary algorithm by using architecture length as fitness
        if DEBUG_EVOLUTION:
            acc = np.sum(self.module.structure)
            return acc, acc
        else:
            data_augment = True
            if data_augment:
                datagen = ImageDataGenerator(
                    featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    zca_epsilon=1e-06,
                    rotation_range=0,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.,
                    zoom_range=0.,
                    channel_shift_range=0.,
                    fill_mode='nearest',
                    cval=0.,
                    horizontal_flip=True,
                    vertical_flip=False,
                    rescale=None,
                    preprocessing_function=None,
                    data_format=None,
                    validation_split=0.0)

                datagen.fit(x_train)

                model.fit_generator(
                            datagen.flow(
                                x_train, 
                                y_train, 
                                batch_size=self.batch_size
                            ),
                            validation_data=(x_test, y_test),
                            epochs=self.epochs, 
                            verbose=self.verbose,
                            callbacks=callbacks
                )

            else:
                model.fit(
                        x_train, 
                        y_train,
                        batch_size=self.batch_size, 
                        epochs=self.epochs, 
                        verbose=self.verbose,
                        validation_data=(x_test, y_test),
                        callbacks=[tensorboard],
                        shuffle=True
                )

            scores = model.evaluate(x_test, y_test, verbose=self.verbose)
            return scores[1], scores[0]

def resnet_layer(
            inputs,
            num_filters=16,
            kernel_size=3,
            strides=1,
            activation='relu',
            batch_normalization=True,
            conv_first=True,
            iteration=None,
            position=None
):
    layername = "layer_conv_resnet_" + str(iteration) + "_" + str(position)
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),
                  name=layername)

    did_batch = False
    batch_name = "first_batch"
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

def lr_schedule_final(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 105:
        lr *= 1e-3
    elif epoch > 100:
        lr *= 1e-2
    elif epoch > 85:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 58:
        lr *= 1e-2
    elif epoch > 50:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
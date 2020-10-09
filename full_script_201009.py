# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:10:42 2020

@author: admin
"""

import argparse
import os
import pathlib
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Model
import numpy as np
import pickle

# THIRD VERSION, 19/07/2020
# 3.1 MAJOR UPDATE 24/07/2020

print("Running script...")

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="local", help="path to data, should be a folder containing 3 subfolders (train, valid and test) each containing 1 subfolder for each possible class (e.g. AD and SMC)")
parser.add_argument("--output", type=str, default="local", help="path to output folder")
parser.add_argument("--step", type=str, default="test")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs for training")
parser.add_argument("--target", type=str, default="alzheimer", help="either alzheimer, cells, or patient for different classification tasks")
# Hyperparameters
parser.add_argument("--core", type=str, default="vgg16", help="inceptionv3, resnet50, vgg16")
parser.add_argument("--colormode", type=str, default="rgb_raw", help="rgb_raw (no modifications), rgb_var (rgb images, random brightness added independently to each channel), gray (black and white images), i.e. which image augmentation pipeline will be applied during training")
parser.add_argument("--globalpooling", type=str, default="avg", help="which global pooling layer will be used when creating the model, avg or max")
parser.add_argument("--optimizer", type=str, default="rmsprop", help="which optimizer to use during training")
parser.add_argument("--weighted", type=int, default=0, help="<=0 (False) or >0 (True), wether or not to use weighted loss during training")
parser.add_argument("--freeze", type=str, default="deep", help="either deep or shallow, which layers will be frozen before training")
parser.add_argument("--lr", type=float, default=1e-5)

parser.add_argument("--debug", type=int, default=0, help="1 for debug, 0 for regular")
FLAGS = parser.parse_args()

DEBUG_MODE = FLAGS.debug!=0

if FLAGS.step not in ("train", "test", "test_all_with_aug", ): # those steps will be excluded from automatic distant run
    raise Exception("Unknown mode: {}".format(FLAGS.step))

if FLAGS.core not in ("inceptionv3","resnet50","vgg16"):
    raise Exception("Unknown core model: {}".format(FLAGS.core))
    
if FLAGS.target not in ("alzheimer", "cells", "patient", "ppf", ):
    raise Exception("Unknown target: {}".format(FLAGS.target))
    
if FLAGS.colormode not in ("rgb_raw", "rgb_var", "gray", "imgaug_severe", ):
    raise Exception("Unknown colormode: {}".format(FLAGS.colormode))
    
if FLAGS.globalpooling not in ("avg", "max", ):
    raise Exception("Unknown global pooling mode: {}".format(FLAGS.globalpooling))
    
if FLAGS.optimizer not in ("rmsprop", "adam", "sgd", ):
    raise Exception("Unknown optimizer mode: {}".format(FLAGS.optimizer))
    
if FLAGS.freeze not in ("deep", "shallow", ):
    raise Exception("Unknown freeze mode: {}".format(FLAGS.freeze))
    
if FLAGS.data == "local":
    if FLAGS.target=="alzheimer":
        FLAGS.data = r"C:\Users\admin\Documents\BigDataPTA\geriatrie\dm\data\v3_alzh\data"
    elif FLAGS.target=="cells":
        FLAGS.data = r"C:\Users\admin\Documents\BigDataPTA\geriatrie\dm\data\v3_class\data"
    elif FLAGS.target=="patient":
        FLAGS.data = r"C:\Users\admin\Documents\BigDataPTA\geriatrie\dm\data\v3_patient\data"
    elif FLAGS.target=="ppf":
        FLAGS.data = r"C:\Users\admin\Documents\BigDataPTA\geriatrie\dm\data\v3_ppf\data"
if FLAGS.output == "local":
    FLAGS.output = r"C:\Users\admin\Documents\BigDataPTA\geriatrie\dm\out"

# defining initial parameters and loading the number and names of classes
IM_WIDTH = 360
IM_HEIGHT = 360
BATCH_SIZE = 32
CLASS_NAMES = os.listdir(os.path.join(FLAGS.data,"train"))
CLASS_NAMES = [CLASS_NAMES[i] for i in np.argsort(CLASS_NAMES)] # make sure alphabetical order is preserved
N_CLASSES = len(CLASS_NAMES)

print("Class names are: {}".format(CLASS_NAMES))

# %%

# Below we define the generator for feeding data & augmentation pipelines
if FLAGS.colormode == "imgaug_severe" or FLAGS.step=="augmentation_demo":
    import imgaug as ia
    from imgaug import augmenters as iaa
    import cv2
    #from imgaug import parameters as iap
    
    class ImgAugDataGenerator(tf.keras.utils.Sequence):
        'Generates data for Keras'
        def __init__(self, data_path, batch_size, image_dimensions, class_names, preprocess_function, shuffle=False, augment=False):
            #if False:
            #    data_path = os.path.join(FLAGS.data,"train")
            #    batch_size = 8
            #    image_dimensions = (360,360,)
            #    class_names = ["SNE", ]
            #    preprocess_function = preprocess_input
            #    shuffle=False
            #    augment=True
            # load images
            images_paths = [os.path.join(data_path,c,im) for c in os.listdir(data_path) for im in os.listdir(os.path.join(data_path,c))]
            # load labels
            labels = [c for c in os.listdir(data_path) for im in os.listdir(os.path.join(data_path,c))]
            # filter: keep only labels listed in class_names
            images_paths = [ip for i,ip in enumerate(images_paths) if labels[i] in class_names]
            labels = [l for l in labels if l in class_names]
            # convert labels to numpy array (one hot encode)
            labels_dict = {c:i for i,c in enumerate(class_names)}
            labels = [labels_dict[l] for l in labels]
            b = np.zeros((len(labels), max(labels)+1))
            b[np.arange(len(labels)),labels] = 1
            print("Custom imgaug generator: found {} images belonging to {} classes.".format(len(images_paths),b.shape[1]))
            #if False:
            #    labels=b
            #    dim=image_dimensions
            self.images_paths = images_paths        # array of image paths
            self.labels       = b                   # array of labels (one hot encoded)
            self.dim          = image_dimensions    # image dimensions
            self.batch_size   = batch_size          # batch size
            self.shuffle      = shuffle             # shuffle bool
            self.augment      = augment             # augment data bool
            self.class_names  = class_names
            self.preprocess_function = preprocess_function
            self.on_epoch_end()
    
        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor(len(self.images_paths) / self.batch_size))
    
        def on_epoch_end(self):
            'Updates indexes after each epoch'
            #if False:
            #    indexes = np.arange(len(images_paths))
            self.indexes = np.arange(len(self.images_paths))
            if self.shuffle:
                np.random.shuffle(self.indexes)
    
        def __getitem__(self, index):
            'Generate one batch of data'
            # selects indices of data for next batch
            indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
    
            # select data and load images
            labels = np.array([self.labels[k] for k in indexes])
            images = [cv2.imread(self.images_paths[k]) for k in indexes]
            
            # preprocess and augment data
            if self.augment == True:
                images = self.augmentor(images)
            
            images = np.array([self.preprocess_function(img) for img in images])
            return images, labels
        
        def augmentor(self, images):
            'Apply data augmentation'
            sometimes = lambda aug: iaa.Sometimes(0.5, aug)
            seq = iaa.Sequential(
                [
                    # apply the following augmenters to most images
                    iaa.Fliplr(0.5), # horizontally flip 50% of all images
                    iaa.Flipud(0.2), # vertically flip 20% of all images
                    # crop images by -5% to 10% of their height/width
                    sometimes(iaa.CropAndPad(
                        percent=(-0.05, 0.1),
                        pad_mode=ia.ALL,
                        pad_cval=(0, 255)
                    )),
                    sometimes(iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45), # rotate by -45 to +45 degrees
                        shear=(-16, 16), # shear by -16 to +16 degrees
                        order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                        cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                        mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )),
                    # execute 0 to 5 of the following (less important) augmenters per image
                    # don't execute all of them, as that would often be way too strong
                    iaa.SomeOf((0, 5),
                        [
                            sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                            iaa.OneOf([
                                iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                                iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                                iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                            ]),
                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                            # search either for all edges or for directed edges,
                            # blend the result with the original image using a blobby mask
                            iaa.SimplexNoiseAlpha(iaa.OneOf([
                                iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                            ])),
                            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                            iaa.OneOf([
                                iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                                iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                            ]),
                            iaa.Invert(0.05, per_channel=True), # invert color channels
                            iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                            iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                            # either change the brightness of the whole image (sometimes
                            # per channel) or change the brightness of subareas
                            iaa.OneOf([
                                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                iaa.FrequencyNoiseAlpha(
                                    exponent=(-4, 0),
                                    first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                    second=iaa.LinearContrast((0.5, 2.0))
                                )
                            ]),
                            iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                            iaa.Grayscale(alpha=(0.0, 1.0)),
                            sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                        ],
                        random_order=True
                    )
                ],
                random_order=True
            )
            return seq.augment_images(images)

# %%

# In this step, we'll create and initialize the selected model, and run training on the training set (subfolder train in the data path)
if FLAGS.step=="train":
    EPOCHS = FLAGS.epochs
    N_IMAGES = len([f for c in CLASS_NAMES for f in os.listdir(os.path.join(FLAGS.data,"train",c))])
    N_IMAGES_VALID = sum([len(os.listdir(os.path.join(FLAGS.data,"valid",cl))) for cl in os.listdir(os.path.join(FLAGS.data,"valid"))])
    MIN_LEARNING_RATE = FLAGS.lr*1e-2
    LEARNING_RATE_DEC_FACTOR = 10
    WORKERS_COUNT = 1
    
    # pathlib.Path(os.path.join(OUTPUT_PATH,"tb")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(FLAGS.output)).mkdir(parents=True, exist_ok=True)
    
    # CREATE THE MODEL
    if FLAGS.core == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        print("Downloading InceptionV3 model and weights...", end="")
        core_model = InceptionV3(include_top = False, input_shape = (IM_HEIGHT, IM_WIDTH, 3), pooling = FLAGS.globalpooling)
        print(" done.")
        
        if FLAGS.freeze == "shallow":
            N_UNFREEZE = 9 # dernier bloc pour Inception v3
            N_CONV = len([l for l in core_model.layers if l.name[:4]=="conv"])
            N_FREEZE = N_CONV-N_UNFREEZE
            print("Freezing {} first convolutional layers...".format(N_FREEZE), end="")
            n = 1
            for l in core_model.layers:
                if l.name[:4]=="conv" and n<=N_FREEZE:
                    l.trainable = False
                    n += 1
            print(" done.")
        elif FLAGS.freeze == "deep":
            N_UNFREEZE = 5 # dernier bloc pour Inception v3
            N_CONV = len([l for l in core_model.layers if l.name[:4]=="conv"])
            N_FREEZE = N_CONV-N_UNFREEZE
            print("Freezing {} last convolutional layers...".format(N_FREEZE), end="")
            n = 1
            for l in core_model.layers:
                if l.name[:4]=="conv":
                    if n>N_UNFREEZE:
                        l.trainable = False
                    n += 1
            print(" done.")
    elif FLAGS.core in ("vgg16",):
        if FLAGS.freeze=="shallow":
            BLOCKS_FREEZE = ("block1","block2","block3","block4",) # blocks à freezer
        elif FLAGS.freeze=="deep":
            BLOCKS_FREEZE = ("block2","block3","block4","block5") # blocks à freezer
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
        print("Downloading VGG16 model and weights...", end="")
        core_model = VGG16(include_top = False, input_shape = (IM_HEIGHT, IM_WIDTH, 3), pooling = FLAGS.globalpooling)
        print(" done.")
        
        FREEZE_LAYERS = [i for i,l in enumerate(core_model.layers) if l.name[:6] in BLOCKS_FREEZE and l.name[7:11]=="conv"]
        print("Freezing {} layers: {}...".format(len(FREEZE_LAYERS),[l.name for i,l in enumerate(core_model.layers) if i in FREEZE_LAYERS]), end="")
        for i in FREEZE_LAYERS:
            core_model.layers[i].trainable = False
        print(" done.")
    
    print("Creating model with transfer learning...", end="")
    x = core_model.output
    x = Flatten()(x)
    output = Dense(N_CLASSES, activation='softmax')(x)
    model = Model(inputs=core_model.input, outputs=output)
    
    if FLAGS.optimizer=="sgd":
        print("Setting optimizer to SGD with learning rate: {:.1e}".format(FLAGS.lr))
        opt=tf.keras.optimizers.SGD(lr=FLAGS.lr, momentum=0.9, nesterov=True)
    elif FLAGS.optimizer=="rmsprop":
        print("Setting optimizer to RMSprop with learning rate: {:.1e}".format(FLAGS.lr))
        opt=tf.keras.optimizers.RMSprop(learning_rate = FLAGS.lr)
    elif FLAGS.optimizer=="adam":
        print("Setting optimizer to Adam with learning rate: {:.1e}".format(FLAGS.lr))
        opt=tf.keras.optimizers.Adam(learning_rate = FLAGS.lr)
    else:
        raise Exception("No optimizer defined for core: {}".format(FLAGS.core))
    
    #if FLAGS.target == "alzheimer":
    #    model.compile(optimizer = opt, loss = "binary_crossentropy", metrics = ["categorical_accuracy"])
    #elif FLAGS.target == "cells" or FLAGS.target == "patient" or FLAGS.target == "ppf":
    #    model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["categorical_accuracy"])
    model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["categorical_accuracy"])
    
    print(" done.")
    model.summary()
    
    #model.save(filepath = os.path.join(OUTPUT_PATH,"models",FINGERPRINT+".h5"))
    #model.save_weights(filepath = os.path.join(OUTPUT_PATH,"models",FINGERPRINT+"_weights.h5"))
    
    # DEFINE AUGMENTATION PIPELINE AND INSTANTIATE GENERATORS FOR TRAINING AND VALIDATION SETS
    if FLAGS.colormode in ("rgb_raw", "rgb_var", "gray", ):
        if FLAGS.colormode=="rgb_var":
            # make our own preprocessing_function
            print("Changing preprocessing to include RGB variations")
            def rgb_var_preprocess_input(x, **kwargs): # should accept values between 0 and 255
                for channel in range(x.shape[-1]):
                    delta = np.random.uniform(low=-.2*255, high=.2*255, size=1)[0]
                    x[...,channel] = x[...,channel] + delta
                x[x>255]=255
                x[x<0]=0
                return(preprocess_input(x, **kwargs))
            train_preprocess_input = rgb_var_preprocess_input
            valid_preprocess_input = preprocess_input
        elif FLAGS.colormode=="gray":
            # make our own preprocessing_function
            print("Changing preprocessing to turn images to grayscale")
            def gray_preprocess_input(x, **kwargs): # should accept values between 0 and 255
                x = np.repeat(np.dot(x, [0.2989, 0.5870, 0.1140])[:,:,np.newaxis], 3, axis=2)
                x[x>255] = 255
                return(preprocess_input(x, **kwargs))
            train_preprocess_input = gray_preprocess_input
            valid_preprocess_input = gray_preprocess_input
        else:
            print("Setting default preprocessing")
            train_preprocess_input = preprocess_input
            valid_preprocess_input = preprocess_input
        
        print("Creating image generator with keras...", end="")
        train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 360,
                                                                                width_shift_range = .1,
                                                                                height_shift_range = .1,
                                                                                brightness_range = [.8, 1.2],
                                                                                shear_range = .2,
                                                                                zoom_range = .2,
                                                                                horizontal_flip = True,
                                                                                preprocessing_function = train_preprocess_input,
                                                                                fill_mode = 'nearest')
        
        train_gen = train_image_generator.flow_from_directory(directory=os.path.join(FLAGS.data,"train"),
                                                              batch_size=BATCH_SIZE,
                                                              shuffle=True,
                                                              target_size=(IM_HEIGHT, IM_WIDTH),
                                                              classes = CLASS_NAMES)
        
        #if False:
        #    for X, y in train_gen:
        #        break
        #    
        #    from matplotlib import pyplot as plt
        #    plt.figure(figsize=(8,8))
        #    for i in range(8):
        #        plt.subplot(3,3,i+1)
        #        plt.imshow((X[i]+1)/2)

        # we can't feed validation data as a generator
        # so we'll just load all images for the validation set
        valid_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = valid_preprocess_input)
        valid_gen = valid_image_generator.flow_from_directory(directory=os.path.join(FLAGS.data,"valid"),
                                                              batch_size=N_IMAGES_VALID,
                                                              shuffle=True,
                                                              target_size=(IM_HEIGHT, IM_WIDTH),
                                                              classes = CLASS_NAMES)
        for X_valid, y_valid in valid_gen:
            break
        
        print(" done.")
    
    elif FLAGS.colormode == "imgaug_severe":
        print("Creating custom image generator with imgaug...", end="")
        train_gen = ImgAugDataGenerator(data_path = os.path.join(FLAGS.data,"train"),
                                        batch_size = BATCH_SIZE,
                                        image_dimensions = (IM_HEIGHT, IM_WIDTH, ),
                                        class_names = CLASS_NAMES,
                                        preprocess_function = preprocess_input,
                                        shuffle=True,
                                        augment=False)    
        valid_gen = ImgAugDataGenerator(data_path = os.path.join(FLAGS.data,"valid"),
                                        batch_size = N_IMAGES_VALID,
                                        image_dimensions = (IM_HEIGHT, IM_WIDTH, ),
                                        class_names = CLASS_NAMES,
                                        preprocess_function = preprocess_input,
                                        shuffle=True,
                                        augment=False)    
        for X_valid, y_valid in valid_gen:
            break
        
        print(" done.")
    else:
        raise Exception("Generator for colormode '{}' not handled".format(FLAGS.colormode))
        
    if FLAGS.weighted>0:
        weights = {c:len(os.listdir(os.path.join(FLAGS.data,"train",c))) for c in CLASS_NAMES} # get n for each class
        weights = {k:max(weights.values())/v for k,v in weights.items()} # weight
        weights = [weights[c] for c in CLASS_NAMES] # turn to list
        print("Using class weights: {}".format(weights))
    else:
        print("Not using class weights")
        weights = None
    
    if DEBUG_MODE:
        N_IMAGES=BATCH_SIZE*2
        X_valid = X_valid[:(BATCH_SIZE*2),...]
        y_valid = y_valid[:(BATCH_SIZE*2),...]
        EPOCHS=1
        
    # Run training
    print("Running training with {} images, validation on {} images...".format(N_IMAGES,X_valid.shape[0]))
    
    # weights ? avec vs sans
    tmp_wght = "noweights"
    if FLAGS.weighted>0:
        tmp_wght = "weights"
    model_name = "{}_{}_{}_{}pool_{}_{}_lr{:.0e}_{}.h5".format(FLAGS.target,FLAGS.core,FLAGS.colormode,FLAGS.globalpooling,FLAGS.optimizer,FLAGS.freeze,FLAGS.lr,tmp_wght)

    history = model.fit(x = train_gen,
                        validation_data = (X_valid, y_valid, ),
                        steps_per_epoch = N_IMAGES // BATCH_SIZE,
                        class_weight = weights,
                        epochs = EPOCHS,
                        workers = WORKERS_COUNT,
                        callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(FLAGS.output,model_name), monitor='val_loss', save_best_only=True, save_weights_only=False),
                                     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=1e-2),
                                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/LEARNING_RATE_DEC_FACTOR, patience=3, min_lr=MIN_LEARNING_RATE, min_delta=1e-2)],
                        verbose = 1)
    
    # save training history
    with open(os.path.join(FLAGS.output,model_name[:-2]+"log"), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
            
    print("Done.")
    
# In the step below we'll select on architecture with defined hyperparameters
# Reload the model and run inference for train, valid and test data
# And save raw results (i.e. predictions) into numpy arrays
if FLAGS.step=="test":
    if FLAGS.core == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif FLAGS.core == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif FLAGS.core in ("vgg16",):
        from tensorflow.keras.applications.vgg16 import preprocess_input
            
    print("Loading model and weights...", end="")
    tmp_wght = "noweights"
    if FLAGS.weighted>0:
        tmp_wght = "weights"
    model_name = "{}_{}_{}_{}pool_{}_{}_lr{:.0e}_{}.h5".format(FLAGS.target,FLAGS.core,FLAGS.colormode,FLAGS.globalpooling,FLAGS.optimizer,FLAGS.freeze,FLAGS.lr,tmp_wght)
    model = tf.keras.models.load_model(filepath = os.path.join(FLAGS.output,model_name))
    print(" done.")
    
    # print(model.summary())
    
    # create subpath (easier to retrieve data)
    npys_output_path = os.path.join(FLAGS.output,"results")
    pathlib.Path(npys_output_path).mkdir(parents=True, exist_ok=True)
    
    print(model.summary())
    
    for dset in ("train", "valid", "test", ):
        if dset=="train":
            print("Analyzing training set")
        elif dset=="valid":
            print("Analyzing validation set")
        elif dset=="test":
            print("Analyzing test set")
    
        N_IMAGES = len([f for c in CLASS_NAMES for f in os.listdir(os.path.join(FLAGS.data,dset,c))])
    
        if FLAGS.colormode=="rgb_var" or FLAGS.colormode=="imgaug_severe":
            valid_preprocess_input = preprocess_input
        elif FLAGS.colormode=="gray":
            # make our own preprocessing_function
            print("Changing preprocessing to turn images to grayscale")
            def gray_preprocess_input(x, **kwargs): # should accept values between 0 and 255
                x = np.repeat(np.dot(x, [0.2989, 0.5870, 0.1140])[:,:,np.newaxis], 3, axis=2)
                x[x>255] = 255
                return(preprocess_input(x, **kwargs))
            valid_preprocess_input = gray_preprocess_input
        else:
            print("Setting default preprocessing")
            valid_preprocess_input = preprocess_input
        
        # load batch of images
        # get real y
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = valid_preprocess_input)
        gen = image_generator.flow_from_directory(directory=os.path.join(FLAGS.data,dset),
                                                              batch_size=N_IMAGES,
                                                              target_size=(IM_HEIGHT, IM_WIDTH),
                                                              shuffle=False,
                                                              classes = CLASS_NAMES)
    
        print("Loading data...", end="")
        X, y = next(gen)
        print(" done.")
    
        print("Computing predictions...", end="")
        y_ = model.predict(X)
        print(" done.")
        
        print("Saving...", end="")
        np.save(os.path.join(npys_output_path,"{}_{}_gt.npy".format(model_name[:-3],dset)), y)
        np.save(os.path.join(npys_output_path,"{}_{}_pred.npy".format(model_name[:-3],dset)), y_)
        print(" done.")
        
# %%

# Same as previous section
# But we'll load all models (i.e. with all hyperparameters combinations) trained for the same prediction
# And for each model run inference and output raw results as numpy arrays
if FLAGS.step=="test_all_with_aug":
    import re
    
    if FLAGS.core == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif FLAGS.core == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif FLAGS.core in ("vgg16",):
        from tensorflow.keras.applications.vgg16 import preprocess_input
        
    dsets = (dict(name="train"), dict(name="valid"), dict(name="test"), )
    # Load datasets
    for dset in dsets:
        if dset["name"]=="train":
            print("Loading training set...", end="")
        elif dset["name"]=="valid":
            print("Loading validation set...", end="")
        elif dset["name"]=="test":
            print("Loading test set...", end="")
    
        N_IMAGES = len([f for c in CLASS_NAMES for f in os.listdir(os.path.join(FLAGS.data,dset["name"],c))])
        dset["n_images"] = N_IMAGES
    
        # make two pipelines : one with augmentation, one without
        image_generator_raw = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = preprocess_input)
        raw_gen = image_generator_raw.flow_from_directory(directory=os.path.join(FLAGS.data,dset["name"]),
                                                          batch_size=N_IMAGES,
                                                          target_size=(IM_HEIGHT, IM_WIDTH),
                                                          shuffle=False,
                                                          classes = CLASS_NAMES)
        
        image_generator_aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 90,
                                                                              width_shift_range = .5,
                                                                              height_shift_range = .5,
                                                                              horizontal_flip = True,
                                                                              preprocessing_function = preprocess_input,
                                                                              fill_mode = 'nearest')
        aug_gen = image_generator_aug.flow_from_directory(directory=os.path.join(FLAGS.data,dset["name"]),
                                                          batch_size=N_IMAGES,
                                                          target_size=(IM_HEIGHT, IM_WIDTH),
                                                          shuffle=False,
                                                          classes = CLASS_NAMES)
    
        print(" loading data...", end="")
        X, y = next(raw_gen)
        dset["X"] = X
        dset["y"] = y
        X, y = next(aug_gen)
        dset["X_aug"] = X
        if np.any(dset["y"]!=y):
            raise Exception("Error while loading data: y's don't match")
        print(" done.")
        
    # Compute predictions for each model
    # list available models
    model_names = [f for f in os.listdir(FLAGS.output) if f[-3:]==".h5"]
    # restrict to models using current target AND preprocess (core)
    model_names = [f for f in model_names if re.search(FLAGS.core, f) is not None and re.search(FLAGS.target, f) is not None]
    
    # create subpath (easier to retrieve data)
    npys_output_path = os.path.join(FLAGS.output,"results")
    pathlib.Path(npys_output_path).mkdir(parents=True, exist_ok=True)
    
    # for each model...
    print("")
    print("Predictions", end="")
    for i,model_name in enumerate(model_names):
        model = tf.keras.models.load_model(filepath = os.path.join(FLAGS.output,model_name))
        # ... compute predictions for each dataset
        for dset in dsets:
            print("\rPredictions for {} (model {}/{})".format(dset["name"],i+1,len(dsets)), end="")
            y_ = model.predict(dset["X"])
            y_aug_ = model.predict(dset["X_aug"])
            
            np.save(os.path.join(npys_output_path,"{}_{}_gt.npy".format(model_name[:-3],dset["name"])), dset["y"])
            np.save(os.path.join(npys_output_path,"{}_{}_pred.npy".format(model_name[:-3],dset["name"])), y_)
            np.save(os.path.join(npys_output_path,"{}_{}_pred_aug.npy".format(model_name[:-3],dset["name"])), y_aug_)
            
        # clear session for next model
        tf.keras.backend.clear_session()
    
    
# %%
        
# In this step we'll make a summary of the raw results of one model (with defined hyperparameters)
# We'll reload the raw results output by the model during inference
# And compute multiple metrics (accuracy, ROC-AUC)
# Then output figure(s) and results table(s) for summarizing the model performance
if FLAGS.step == "test_summary":
    from misc_ml.viz import plotROC
    import pandas as pd
    
    tmp_wght = "noweights"
    if FLAGS.weighted>0:
        tmp_wght = "weights"
    model_name = "{}_{}_{}_{}pool_{}_{}_lr{:.0e}_{}.h5".format(FLAGS.target,FLAGS.core,FLAGS.colormode,FLAGS.globalpooling,FLAGS.optimizer,FLAGS.freeze,FLAGS.lr,tmp_wght)
    results = dict()
    for dset in ("train","valid","test", ):
        results[dset] = dict()
        # load data
        results[dset]["gt"] = np.load(os.path.join(FLAGS.output,"results","{}_{}_gt.npy".format(model_name[:-3],dset)))
        results[dset]["pred"] = np.load(os.path.join(FLAGS.output,"results","{}_{}_pred.npy".format(model_name[:-3],dset)))
        results[dset]["pred_aug"] = np.load(os.path.join(FLAGS.output,"results","{}_{}_pred_aug.npy".format(model_name[:-3],dset)))
        # # compute accuracy
        results[dset]["gt_class"] = np.argmax(results[dset]["gt"], axis=1)
        results[dset]["pred_class"] = np.argmax(results[dset]["pred"], axis=1)
        results[dset]["crosstab"] = pd.crosstab(results[dset]["gt_class"], "Pred: "+pd.Series(results[dset]["pred_class"]).astype(str))
        print("'{}' set cross table:\n{}".format(dset,results[dset]["crosstab"]))
        print("Accuracy on {} set: {:.3f}".format(dset,np.sum(results[dset]["gt_class"]==results[dset]["pred_class"])/results[dset]["pred_class"].shape[0]))
        # compute map
        #from sklearn.metrics import average_precision_score
        #
        #aps = []
        #for i in range(N_CLASSES):
        #    aps.append(average_precision_score(results[dset]["gt"][:,i], results[dset]["pred"][:,i]))
        #print("mAP: {:.1f}%".format(100*np.mean(aps)))
        
        if FLAGS.target=="alzheimer":
            # calcul kappa
            y_gt = results[dset]["gt"][:,0]
            y_pred = results[dset]["pred"][:,0]
            y_pred_aug = results[dset]["pred_aug"][:,0]
            # determine optimal threshold according to ROC curve
            from sklearn.metrics import cohen_kappa_score
            best_thres_with_kappa = []
            for thres in np.unique(y_pred):
                y_pred_bin = (y_pred>thres)*1
                cur_kappa = cohen_kappa_score(y_gt,y_pred_bin)
                if len(best_thres_with_kappa) == 0 or best_thres_with_kappa[0]<cur_kappa:
                    best_thres_with_kappa = [cur_kappa,thres]
            best_thres_with_kappa # best kappa : 0.30 for 0.726 as threshold
            best_thres_with_kappa = []
            for thres in np.unique(y_pred_aug):
                y_pred_aug_bin = (y_pred_aug>thres)*1
                cur_kappa = cohen_kappa_score(y_gt,y_pred_aug_bin)
                if len(best_thres_with_kappa) == 0 or best_thres_with_kappa[0]<cur_kappa:
                    best_thres_with_kappa = [cur_kappa,thres]
            best_thres_with_kappa # best kappa : 0.19 for 0.835 as threshold

            
    
    # Accuracy by class
    for c, cname in enumerate(CLASS_NAMES):
        dset = "test"
        #tpreds_for_c = y_train_class_[y_train_class==c]
        vpreds_for_c = results[dset]["pred_class"][results[dset]["gt_class"]==c]
        
        #tacc = np.sum(tpreds_for_c==c)/tpreds_for_c.shape[0]
        tacc = 0
        vacc = np.sum(vpreds_for_c==c)/vpreds_for_c.shape[0]
        
        bsp = "    "
        if len(cname)==3:
            bsp = "   "
        print("Class {}:{}train: {:.1f}    test: {:.1f}".format(cname,bsp,100*tacc,100*vacc))
        
        if vpreds_for_c.shape[0]>3:
            # plot ROC
            y_class_dummy = np.zeros(results[dset]["gt_class"].shape)
            y_class_dummy[results[dset]["gt_class"]==c]=1
            plotROC(y_class_dummy, results[dset]["pred"][:,c], title = "Test set ROC curve for class: {}".format(cname))
            
    # Plot learning curves
    with open(os.path.join(FLAGS.output,model_name[:-3]+".log"), 'rb') as file_pi:
        hst = pickle.load(file_pi)
        
    from matplotlib import pyplot as plt
    plt.figure(figsize=(6,6))
    
    plt.subplot(3,1,1)
    plt.plot(np.arange(1,len(hst["loss"])+1),hst["loss"], color="red", label="Training set")
    plt.plot(np.arange(1,len(hst["val_loss"])+1),hst["val_loss"], color="blue", label="Validation set")
    plt.title("Epoch loss")
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.plot(np.arange(1,len(hst["categorical_accuracy"])+1),hst["categorical_accuracy"], color="red", label="Training set")
    plt.plot(np.arange(1,len(hst["val_categorical_accuracy"])+1),hst["val_categorical_accuracy"], color="blue", label="Validation set")
    plt.title("Epoch categorical accuracy")
    plt.legend()
    
    plt.subplot(3,1,3)
    plt.plot(np.arange(1,len(hst["lr"])+1),hst["lr"], color="orange", label="Learning rate")
    plt.yscale("log")
    plt.title("Epoch learning rate")
    plt.legend()
    
    plt.show()

    ## Pour retrouver les images sur lesquelles il y a eu des erreurs
    #train_errs = np.where(y_train_class != y_train_class_)[0].tolist()
    #test_errs = np.where(y_test_class != y_test_class_)[0].tolist()
    #
    ## find files matching the errors indices
    #TRAIN_SAMPLES_PATH = [os.path.join(c,f) for c in CLASS_NAMES for f in os.listdir(os.path.join(DATA_PATH,"train",c))]
    #TEST_SAMPLES_PATH = [os.path.join(c,f) for c in CLASS_NAMES for f in os.listdir(os.path.join(DATA_PATH,"test",c))]
    #
    #[TRAIN_SAMPLES_PATH[idx] for idx in train_errs]
    #[TEST_SAMPLES_PATH[idx] for idx in test_errs]

# %%
    
# In this step we'll make a summary of the raw results of all models (with all possible hyperparameters combinations)
# trained for the same prediction
# We'll reload the raw results output by the models during inference
# And compute multiple metrics (accuracy, ROC-AUC)
# Then output a single table summarizing the performance of all those models
if FLAGS.step == "global_test_summary":
    import pandas as pd
    import re
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm
    
    WITH_AUG = True
    
    # load all available results
    models_results_files = os.listdir(os.path.join(FLAGS.output,"results"))
    
    # keep only training/ground truth (we'll deduce valid and test filenames from train file's filename)
    models_results_files = [f for f in models_results_files if re.search("train_gt", f) is not None]
    print("{} models results found".format(len(models_results_files)))
    
    # define our function to convert files into understandable data
    def numerizeFile(f):
        out = dict()
        params = [dict(name="target", values=("alzheimer","cells","patient","ppf",)),
                  dict(name="core", values=("inceptionv3","vgg16",)),
                  dict(name="colormode", values=("imgaug_severe","rgb_raw",)),
                  dict(name="globalpooling", values=("avgpool","maxpool",)),
                  dict(name="optimizer", values=("adam","rmsprop",)),
                  dict(name="weighted", values=("noweights","weights",)),
                  dict(name="freezing", values=("shallow","deep",)),
                  dict(name="lr", values=("lr1e-03","lr1e-05",))]
        for p in params:
            for s in p["values"]:
                if re.search(s, f) is not None:
                    out[p["name"]] = s
                    break
        if "lr" in out.keys():
            out["lr"] = float(out["lr"][2:])
        return out

    if FLAGS.target=="alzheimer":
        match_table = dict()
        for k in ("train","valid","test",):
            match_table[k] = pd.read_excel(os.path.join(FLAGS.data, "match_table_{}.xlsx".format(k)))
    
    results = []
    for f in tqdm(models_results_files, desc="Aggregating results"):
        # get meta data from filename
        model_meta = numerizeFile(f)
        model_meta["model"] = f[:-13]+".h5"
        # keep only if concerned by target
        if model_meta["target"]!=FLAGS.target:
            continue
        # get predictions
        train_gt = np.load(os.path.join(FLAGS.output,"results",f))
        train_pred = np.load(os.path.join(FLAGS.output,"results",re.sub("train_gt", "train_pred", f)))
        valid_gt = np.load(os.path.join(FLAGS.output,"results",re.sub("train_gt", "valid_gt", f)))
        valid_pred = np.load(os.path.join(FLAGS.output,"results",re.sub("train_gt", "valid_pred", f)))
        test_gt = np.load(os.path.join(FLAGS.output,"results",re.sub("train_gt", "test_gt", f)))
        test_pred = np.load(os.path.join(FLAGS.output,"results",re.sub("train_gt", "test_pred", f)))
        if WITH_AUG:
            train_pred_aug = np.load(os.path.join(FLAGS.output,"results",re.sub("train_gt", "train_pred_aug", f)))
            valid_pred_aug = np.load(os.path.join(FLAGS.output,"results",re.sub("train_gt", "valid_pred_aug", f)))
            test_pred_aug = np.load(os.path.join(FLAGS.output,"results",re.sub("train_gt", "test_pred_aug", f)))
        # compute metrics
        if train_gt.shape[1]==2:
            model_meta["train_auc"] = roc_auc_score(train_gt[:,1],train_pred[:,1])
            model_meta["valid_auc"] = roc_auc_score(valid_gt[:,1],valid_pred[:,1])
            model_meta["test_auc"] = roc_auc_score(test_gt[:,1],test_pred[:,1])
            results.append(model_meta)
            if WITH_AUG:
                model_meta["train_auc_aug"] = roc_auc_score(train_gt[:,1],train_pred_aug[:,1])
                model_meta["valid_auc_aug"] = roc_auc_score(valid_gt[:,1],valid_pred_aug[:,1])
                model_meta["test_auc_aug"] = roc_auc_score(test_gt[:,1],test_pred_aug[:,1])
        else:
            model_meta["train_acc"] = np.sum(np.argmax(train_gt, axis=1)==np.argmax(train_pred, axis=1))/train_gt.shape[0]
            model_meta["valid_acc"] = np.sum(np.argmax(valid_gt, axis=1)==np.argmax(valid_pred, axis=1))/valid_gt.shape[0]
            model_meta["test_acc"] = np.sum(np.argmax(test_gt, axis=1)==np.argmax(test_pred, axis=1))/test_gt.shape[0]
            if WITH_AUG:
                model_meta["train_acc_aug"] = np.sum(np.argmax(train_gt, axis=1)==np.argmax(train_pred_aug, axis=1))/train_gt.shape[0]
                model_meta["valid_acc_aug"] = np.sum(np.argmax(valid_gt, axis=1)==np.argmax(valid_pred_aug, axis=1))/valid_gt.shape[0]
                model_meta["test_acc_aug"] = np.sum(np.argmax(test_gt, axis=1)==np.argmax(test_pred_aug, axis=1))/test_gt.shape[0]
                
        if FLAGS.target=="alzheimer":
            from misc_ml.viz import plotROC
            # compute total/cumulated score
            preds = dict(train=train_pred, valid=valid_pred, test=test_pred)
            for p in ("train","valid","test",):
                mt = match_table[p].copy()
                mt.diag = mt.diag.replace({"F00.1":1,"R413":0})
                mt["score"] = preds[p][:,0]
                agg_pred = [dict(folder=folder, diag=mt.diag.loc[mt.folder==folder].iloc[0], mean_score=mt.score.loc[mt.folder==folder].mean()) for folder in mt.folder.unique()]
                model_meta["{}_mean_auc".format(p)] = roc_auc_score(np.array([a["diag"] for a in agg_pred]), np.array([a["mean_score"] for a in agg_pred]))
            results.append(model_meta)
    results_df = pd.DataFrame(results)
    
    # Check all grid search combinations are here
    len(results_df.core.unique())*len(results_df.colormode.unique())*len(results_df.globalpooling.unique())*len(results_df.optimizer.unique())*len(results_df.weighted.unique())*len(results_df.lr.unique())*len(results_df.freezing.unique())
    
    print("{} models compared".format(results_df.shape[0]))
    
    params = ("core","colormode","globalpooling","optimizer","weighted","freezing","lr", )
    [np.unique(results_df.loc[:,p], return_counts=True) for p in params]
    
    from matplotlib import pyplot as plt
    #from matplotlib.ticker import PercentFormatter
    if FLAGS.target=="alzheimer":
        color_dict = (dict(name="imgaug_severe",color="#ff0000",label="Strong augmentation"),
                      dict(name="rgb_raw",color="#0000ff",label="Soft augmentation"),)
        plt.rc('font', size=12)
        plt.figure(figsize=(6,6,))
        tmpp = plt.hist(np.stack([results_df.valid_auc.loc[results_df.colormode==c["name"]] for c in color_dict], axis=1),
                        bins=np.arange(0,1.01,.1),
                        color=[c["color"] for c in color_dict],
                        label=[c["label"] for c in color_dict])
        plt.gca().yaxis.set_ticks(range(int(np.max(tmpp[0])+1)))
        plt.gca().xaxis.set_ticks(np.arange(0,1.01,.1))
        plt.legend(prop={'size': 12})
        plt.xlabel("AUC-ROC on validation set", fontsize=14)
        plt.ylabel("Number of models", fontsize=14)
        plt.show()
        plt.savefig(os.path.join(FLAGS.output,"images","plots","alzheimer_models_val_auc_hist.png"))
    elif FLAGS.target in ("cells", "patient", ):
        color_dict = (dict(name="imgaug_severe",color="#ff0000",label="Strong augmentation"),
                      dict(name="rgb_raw",color="#0000ff",label="Soft augmentation"),)
        plt.rc('font', size=12)
        plt.figure(figsize=(6,6,))
        tmpp = plt.hist(np.stack([results_df.valid_acc.loc[results_df.colormode==c["name"]] for c in color_dict], axis=1),
                        bins=np.arange(0,1.01,.1),
                        color=[c["color"] for c in color_dict],
                        label=[c["label"] for c in color_dict])
        plt.gca().yaxis.set_ticks(np.arange(0,(1+np.max(tmpp[0])//5)*5+1,5))
        plt.gca().xaxis.set_ticks(np.arange(0,1.01,.1))
        plt.legend(prop={'size': 12})
        plt.xlabel("Accuracy on validation set", fontsize=14)
        plt.ylabel("Number of models", fontsize=14)
        plt.show()
        plt.savefig(os.path.join(FLAGS.output,"images","plots","{}_models_val_acc_hist.png".format(FLAGS.target)))
    
    results_df.to_excel(os.path.join(FLAGS.output,"{}_v2_results.xlsx".format(FLAGS.target)))
    
    
# %%
    
# In this step we'll run inference with a defined model with selected hyperparameters and task
# And perform Grad CAM in order to visualize the model's attention on the image for the output class
# we'll also apply random transformations (i.e. translation or rotation)
# to the image before running inference
if FLAGS.step == "vis_aug":
    # same as gradcam (above)
    # but we will apply random translate/rotation to img
    from matplotlib import pyplot as plt
    
    if FLAGS.core == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif FLAGS.core == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif FLAGS.core in ("vgg16",):
        from tensorflow.keras.applications.vgg16 import preprocess_input

    print("Loading model and weights...", end="")
    tmp_wght = "noweights"
    if FLAGS.weighted>0:
        tmp_wght = "weights"
    model_name = "{}_{}_{}_{}pool_{}_{}_lr{:.0e}_{}.h5".format(FLAGS.target,FLAGS.core,FLAGS.colormode,FLAGS.globalpooling,FLAGS.optimizer,FLAGS.freeze,FLAGS.lr,tmp_wght)
    model = tf.keras.models.load_model(filepath = os.path.join(FLAGS.output,model_name))
    print(" done.")
    
    dset = "train"

    N_IMAGES = len([f for c in CLASS_NAMES for f in os.listdir(os.path.join(FLAGS.data,dset,c))])

    if FLAGS.colormode=="rgb_var":
        valid_preprocess_input = preprocess_input
    elif FLAGS.colormode=="gray":
        # make our own preprocessing_function
        print("Changing preprocessing to turn images to grayscale")
        def gray_preprocess_input(x, **kwargs): # should accept values between 0 and 255
            x = np.repeat(np.dot(x, [0.2989, 0.5870, 0.1140])[:,:,np.newaxis], 3, axis=2)
            x[x>255] = 255
            return(preprocess_input(x, **kwargs))
        valid_preprocess_input = gray_preprocess_input
    else:
        print("Setting default preprocessing")
        valid_preprocess_input = preprocess_input
    
    # load batch of images
    # get real y
    y = np.load(os.path.join(FLAGS.output,"results",model_name[:-3]+"_{}_gt.npy".format(dset)))
    y_ = np.load(os.path.join(FLAGS.output,"results",model_name[:-3]+"_{}_pred.npy".format(dset)))
    
    image_generator_raw = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = valid_preprocess_input)
    raw_gen = image_generator_raw.flow_from_directory(directory=os.path.join(FLAGS.data,dset),
                                                      batch_size=N_IMAGES,
                                                      target_size=(IM_HEIGHT, IM_WIDTH),
                                                      shuffle=False,
                                                      classes = CLASS_NAMES)
    
    image_generator_aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 90,
                                                                          width_shift_range = .5,
                                                                          height_shift_range = .5,
                                                                          horizontal_flip = True,
                                                                          preprocessing_function = valid_preprocess_input,
                                                                          fill_mode = 'nearest')
    aug_gen = image_generator_aug.flow_from_directory(directory=os.path.join(FLAGS.data,dset),
                                                      batch_size=N_IMAGES,
                                                      target_size=(IM_HEIGHT, IM_WIDTH),
                                                      shuffle=False,
                                                      classes = CLASS_NAMES)
    
    X_raw,y_verif_raw = next(raw_gen)
    X_aug,y_verif_aug = next(aug_gen)
    X = (X_raw,X_aug,)
    
    if np.all(y==y_verif_raw) != True:
        raise Exception("Problem while loading images")
    if np.all(y==y_verif_aug) != True:
        raise Exception("Problem while loading images")

    # create our grad cam class
    from tensorflow.keras.models import Model
    import cv2
    class GradCAM:
        def __init__(self, model, classIdx, layerName=None):
            # store the model, the class index used to measure the class
            # activation map, and the layer to be used when visualizing
            # the class activation map
            self.model = model
            self.classIdx = classIdx
            self.layerName = layerName
            
            # if the layer name is None, attempt to automatically find
            # the target output layer
            if self.layerName is None:
                self.layerName = self.find_target_layer()

        def find_target_layer(self):
            # attempt to find the final convolutional layer in the network
            # by looping over the layers of the network in reverse order
            for layer in reversed(self.model.layers):
                # check to see if the layer has a 4D output
                if len(layer.output_shape) == 4:
                    return layer.name
            # otherwise, we could not find a 4D layer so the GradCAM
            # algorithm cannot be applied
            raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
        
        def compute_heatmap(self, image, scale = False, eps=1e-8):
            # construct our gradient model by supplying (1) the inputs
            # to our pre-trained model, (2) the output of the (presumably)
            # final 4D layer in the network, and (3) the output of the
            # softmax activations from the model
            #gradModel = Model(inputs=[model.inputs],
            #            outputs=[model.get_layer(layerName).output,
            #                     model.output])
            gradModel = Model(inputs=[self.model.inputs],
                        outputs=[self.model.get_layer(self.layerName).output,
                                 self.model.output])
            
            # record operations for automatic differentiation
            with tf.GradientTape() as tape:
                # cast the image tensor to a float-32 data type, pass the
                # image through the gradient model, and grab the loss
                # associated with the specific class index
                inputs = tf.cast(image, tf.float32)
                (convOutputs, predictions) = gradModel(inputs)
                #loss = predictions[:, classIdx]
                loss = predictions[:, self.classIdx]
            # use automatic differentiation to compute the gradients
            grads = tape.gradient(loss, convOutputs)
            
            # compute the guided gradients
            castConvOutputs = tf.cast(convOutputs > 0, "float32")
            castGrads = tf.cast(grads > 0, "float32")
            guidedGrads = castConvOutputs * castGrads * grads
            # the convolution and guided gradients have a batch dimension
            # (which we don't need) so let's grab the volume itself and
            # discard the batch
            convOutputs = convOutputs[0]
            guidedGrads = guidedGrads[0]
            
            # compute the average of the gradient values, and using them
            # as weights, compute the ponderation of the filters with
            # respect to the weights
            weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
            cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
            
            # grab the spatial dimensions of the input image and resize
            # the output class activation map to match the input image
            # dimensions
            (w, h) = (image.shape[2], image.shape[1])
            heatmap = cv2.resize(cam.numpy(), (w, h))
            # normalize the heatmap such that all values lie in the range
            # [0, 1], scale the resulting values to the range [0, 255],
            # and then convert to an unsigned 8-bit integer
            numer = heatmap - np.min(heatmap)
            denom = (heatmap.max() - heatmap.min()) + eps
            heatmap = numer / denom
            if scale:
                heatmap = heatmap/np.max(heatmap)
            heatmap = (heatmap * 255).astype("uint8")
            # return the resulting heatmap to the calling function
            return heatmap
        
        def overlay_heatmap(self, heatmap, image, raw_image, alpha=0.5, colormap=cv2.COLORMAP_JET):
            # apply the supplied color map to the heatmap and then
            # overlay the heatmap on the input image
            heatmap_ready = cv2.applyColorMap(255-heatmap, colormap)
            # image_ready = ((image[0]/2.+.5)*255).astype('uint8')
            image_ready = raw_image[0].astype('uint8')
            output_ready = cv2.addWeighted(image_ready, alpha, heatmap_ready, 1 - alpha, 0)
            # return a 2-tuple of the color mapped heatmap and the output,
            # overlaid image
            return (image_ready, heatmap_ready, output_ready)
        
    # the index of the image we want to compute a heatmap for
    only_correctpreds = False
    only_falsepreds = False
    EXPORT=True
    AUG_LABEL_TOPRIGHT = False
    how_many_for_each_class = 8
    np.random.seed(861)
    # we can use those parameters in order to visualize :
    # any random sample
    # only samples for which prediction was correct, so we can see what areas are really important for classification, and were correctly identified by the model
    # only samples for which prediction was wrong, so we can see what provokes overfitting in the model and which regions may be misleading in the images
    if only_correctpreds:
        image_indices = [ind for i in np.unique(np.argmax(y,axis=1)) for ind in np.random.choice(np.intersect1d(np.where(np.argmax(y,axis=1)==i)[0],np.where(np.argmax(y_,axis=1)==i)[0]),how_many_for_each_class,replace=False).tolist()]
    else:
        if only_falsepreds:
            image_indices = [ind for i in np.unique(np.argmax(y,axis=1)) for ind in np.random.choice(np.intersect1d(np.where(np.argmax(y,axis=1)==i)[0],np.where(np.argmax(y_,axis=1)!=i)[0]),how_many_for_each_class,replace=False).tolist()]
        else:
            image_indices = [ind for i in np.unique(np.argmax(y,axis=1)) for ind in np.random.choice(np.where(np.argmax(y,axis=1)==i)[0],how_many_for_each_class,replace=False).tolist()]
        
    image_indices = [ind for i in (3,) for ind in np.random.choice(np.intersect1d(np.where(np.argmax(y,axis=1)==i)[0],np.where(np.argmax(y_,axis=1)==i)[0]),how_many_for_each_class,replace=False).tolist()]
    
    if FLAGS.core=="vgg16":
        def restore_original_image_from_array(x, data_format='channels_last'):
            mean = [103.939, 116.779, 123.68]
        
            # Zero-center by mean pixel
            if data_format == 'channels_first':
                if x.ndim == 3:
                    x[0, :, :] += mean[0]
                    x[1, :, :] += mean[1]
                    x[2, :, :] += mean[2]
                else:
                    x[:, 0, :, :] += mean[0]
                    x[:, 1, :, :] += mean[1]
                    x[:, 2, :, :] += mean[2]
            else:
                x[..., 0] += mean[0]
                x[..., 1] += mean[1]
                x[..., 2] += mean[2]
        
            if data_format == 'channels_first':
                # 'BGR'->'RGB'
                if x.ndim == 3:
                    x = x[::-1, ...]
                else:
                    x = x[:, ::-1, ...]
            else:
                # 'BGR'->'RGB'
                x = x[..., ::-1]
        
            return x
        unpreprocess = restore_original_image_from_array
    else:
        def restore_original_image_from_array(x, data_format="channels_last"):
            return (x+1)*127.5
        unpreprocess = restore_original_image_from_array
        
    if EXPORT:
        tmp_path_out = os.path.join(FLAGS.output,"images","gradcam","raw_images_aug",model_name[:-3])
        pathlib.Path(tmp_path_out).mkdir(parents=True, exist_ok=True)
    for i in image_indices:
        images = [X[j][[i]] for j in range(len(X))]
        raw_images = [unpreprocess(image) for image in images]
        ground_truth = y[[i]]
        predictions = [model.predict(image) for image in images]
        
        ground_truth_class = np.argsort(ground_truth[0])[::-1][0]
        predicted_classes = [np.argsort(prediction[0])[::-1][0] for prediction in predictions]
        
        cams = [GradCAM(model, predicted_class) for predicted_class in predicted_classes]
        heatmaps = [cam.compute_heatmap(image, scale=True) for cam,image in zip(cams,images)]
        # resize the resulting heatmap to the original input image dimensions
        # and then overlay heatmap on top of the image
        heatmaps = [cv2.resize(heatmap, (image.shape[2], image.shape[1])) for heatmap,image in zip(heatmaps,images)]
        
        plot_data = list(map(lambda tmpl: dict(image_ready=tmpl[0],heatmap_ready=tmpl[1],output_ready=tmpl[2]), [cam.overlay_heatmap(heatmap, image, raw_image, alpha=0.5) for cam,heatmap,image,raw_image in zip(cams,heatmaps,images,raw_images)]))
        
        if FLAGS.target=="patient":
            EXPORT_CLASSES = {c:c for c in CLASS_NAMES}
        else:
            EXPORT_CLASSES = {"EO":"EO",
                              "LY":"LY",
                              "MO":"MO",
                              "SNE":"NE",
                              "F00.1":"AD",
                              "R413":"MC"}
            #EXPORT_CLASSES = {"EO":"Eosinosphil",
            #                  "LY":"Lymphocyte",
            #                  "MO":"Monocyte",
            #                  "SNE":"Neutrophil",
            #                  "F00.1":"AD",
            #                  "R413":"MC"}
        
        if EXPORT:
            plt.figure(figsize=(len(plot_data)*2,4))
            for j,dat in enumerate(plot_data):
                plt.subplot(2, len(plot_data), j+1)
                plt.imshow(dat["image_ready"])
                plt.axis("off")
                if AUG_LABEL_TOPRIGHT and j%2!=0:
                    plt.text(358,2,"Ground Truth: {}".format(EXPORT_CLASSES[CLASS_NAMES[ground_truth_class]]),
                             verticalalignment = "top", horizontalalignment = "right", color="#ffffff", fontfamily="Arial", fontsize=10, fontweight="bold",
                             bbox = dict(boxstyle='square', facecolor='black', alpha=1.0))
                else:
                    plt.text(2,2,"Ground Truth: {}".format(EXPORT_CLASSES[CLASS_NAMES[ground_truth_class]]),
                             verticalalignment = "top", color="#ffffff", fontfamily="Arial", fontsize=10, fontweight="bold",
                             bbox = dict(boxstyle='square', facecolor='black', alpha=1.0))
                plt.subplot(2, len(plot_data), j+3)
                plt.imshow(dat["output_ready"])
                plt.axis("off")
                if AUG_LABEL_TOPRIGHT and j%2!=0:
                    plt.text(358,2,"{} ({:.1f}%)".format(EXPORT_CLASSES[CLASS_NAMES[predicted_classes[j]]],100*predictions[j][0,predicted_classes[j]]),
                             verticalalignment = "top", horizontalalignment = "right", color="#ffffff", fontfamily="Arial", fontsize=10, fontweight="bold",
                             bbox = dict(boxstyle='square', facecolor='black', alpha=1.0))
                else:
                    plt.text(2,2,"{} ({:.1f}%)".format(EXPORT_CLASSES[CLASS_NAMES[predicted_classes[j]]],100*predictions[j][0,predicted_classes[j]]),
                             verticalalignment = "top", color="#ffffff", fontfamily="Arial", fontsize=10, fontweight="bold",
                             bbox = dict(boxstyle='square', facecolor='black', alpha=1.0))
            plt.tight_layout()
            plt.show()
            figname = "{}_{}.png".format(CLASS_NAMES[ground_truth_class],i)
            plt.savefig(os.path.join(tmp_path_out,figname))
        else:
            plt.figure(figsize=(8,len(plot_data)*4))
            for j,dat in enumerate(plot_data):
                plt.subplot(len(plot_data), 3, j*3+1)
                plt.imshow(dat["image_ready"])
                plt.title("True class: {} ({})".format(CLASS_NAMES[ground_truth_class],i))
                plt.subplot(len(plot_data), 3, j*3+2)
                plt.imshow(dat["heatmap_ready"])
                plt.title("Heatmap for class {}".format(CLASS_NAMES[predicted_classes[j]]))
                plt.subplot(len(plot_data), 3, j*3+3)
                plt.imshow(dat["output_ready"])
                plt.title("Confidence: {:.1f}%".format(100*predictions[j][0,predicted_classes[j]]))
            plt.show()
        
# %%
        
# finally this step is only about making a figure for demonstrating examples of
# the various image augmentation pipelines
elif FLAGS.step == "augmentation_demo":
    from matplotlib import pyplot as plt
    
    if FLAGS.core == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif FLAGS.core == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif FLAGS.core in ("vgg16",):
        from tensorflow.keras.applications.vgg16 import preprocess_input
        
    # declare igmage augmentation preprocess function
    def rgb_var_preprocess_input(x, **kwargs): # should accept values between 0 and 255
        for channel in range(x.shape[-1]):
            delta = np.random.uniform(low=-.2*255, high=.2*255, size=1)[0]
            x[...,channel] = x[...,channel] + delta
        x[x>255]=255
        x[x<0]=0
        return(preprocess_input(x, **kwargs))
    def gray_preprocess_input(x, **kwargs): # should accept values between 0 and 255
        x = np.repeat(np.dot(x, [0.2989, 0.5870, 0.1140])[:,:,np.newaxis], 3, axis=2)
        x[x>255] = 255
        return(preprocess_input(x, **kwargs))

    dset = "train"
    subset_size = 8
    CLASS_NAMES = ["LY", ]

    N_IMAGES = len([f for c in CLASS_NAMES for f in os.listdir(os.path.join(FLAGS.data,dset,c))])
    
    pipelines = [dict(name="Raw", preproc_func=preprocess_input, aug=None),
                 dict(name="Soft augmentation", preproc_func=preprocess_input, aug="keras"),
                 dict(name="Soft augmentation", preproc_func=preprocess_input, aug="keras"),
                 # dict(name="Grayscale + augmentation", preproc_func=gray_preprocess_input, aug="keras"),
                 # dict(name="Grayscale + augmentation", preproc_func=gray_preprocess_input, aug="keras"),
                 # dict(name="RGB shift + augmentation", preproc_func=rgb_var_preprocess_input, aug="keras"),
                 # dict(name="RGB shift + augmentation", preproc_func=rgb_var_preprocess_input, aug="keras"),
                 dict(name="Severe augmentation", preproc_func=preprocess_input, aug="imgaug"),
                 dict(name="Severe augmentation", preproc_func=preprocess_input, aug="imgaug")]
    
    pipelines_results = []
    for ppl in pipelines:
        
        if ppl["aug"] is None:
            generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = ppl["preproc_func"])
            gen = None
        elif ppl["aug"]=="keras":
            generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 360,
                                                                        width_shift_range = .1,
                                                                        height_shift_range = .1,
                                                                        brightness_range = [.8, 1.2],
                                                                        shear_range = .2,
                                                                        zoom_range = .2,
                                                                        horizontal_flip = True,
                                                                        preprocessing_function = ppl["preproc_func"],
                                                                        fill_mode = 'nearest')
            gen = None
        elif ppl["aug"]=="imgaug":
            gen = ImgAugDataGenerator(data_path = os.path.join(FLAGS.data,dset),
                                      batch_size = subset_size,
                                      image_dimensions = (IM_HEIGHT, IM_WIDTH, ),
                                      class_names = CLASS_NAMES,
                                      preprocess_function = ppl["preproc_func"],
                                      shuffle=False,
                                      augment=True)
        if gen is None:
            gen = generator.flow_from_directory(directory=os.path.join(FLAGS.data,dset),
                                                batch_size=subset_size,
                                                target_size=(IM_HEIGHT, IM_WIDTH),
                                                shuffle=False,
                                                classes = CLASS_NAMES)
        for X,y in gen:
            break
        pipelines_results.append(dict(name=ppl["name"], batch=X))
        
    # make figures
    for i in range(subset_size):
        concat_im_data = []
        for ppl in range(len(pipelines_results)):
            concat_im_data.append(dict(pipeline=pipelines_results[ppl]["name"],im=pipelines_results[ppl]["batch"][i]))
        plt.figure(figsize=(4*len(concat_im_data),4))
        for j in range(len(concat_im_data)):
            im = concat_im_data[j]["im"]
            im=(im+1)*127.5
            im=im.astype("uint8")
            plt.subplot(1, len(concat_im_data), j+1)
            plt.imshow(im)
            plt.tight_layout()
            #plt.title(concat_im_data[j]["pipeline"])
            plt.axis('off')
        plt.show()
        plt.tight_layout()
        filepath = r"C:\Users\admin\Documents\BigDataPTA\geriatrie\dm\out\images\aug_demo"
        filename = os.path.join(filepath,"{}.png".format(len(os.listdir(filepath))+1))
        plt.savefig(filename)
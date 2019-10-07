
#### Fine-tune InceptionV3 on a new set of classes

# The task of fine-tuning a network is to tweak the parameters of an already trained network so that it adapts to the new task at hand. As explained here, the initial layers learn very general features and as we go higher up the network, the layers tend to learn patterns more specific to the task it is being trained on. Thus, for fine-tuning, we want to keep the initial layers intact ( or freeze them ) and retrain the later layers for our task.
# Thus, fine-tuning avoids both the limitations discussed above.
#
# The amount of data required for training is not much because of two reasons. First, we are not training the entire network. Second, the part that is being trained is not trained from scratch.
# Since the parameters that need to be updated is less, the amount of time needed will also be less.
#


# Ref:
# https://github.com/fchollet/deep-learning-with-python-notebooks
# https://gist.github.com/liudanking
# https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
# https://github.com/jkjung-avt/keras-cats-dogs-tutorial/blob/master/train_inceptionresnetv2.py
# https://forums.fast.ai/t/globalaveragepooling2d-use/8358

### You might consider running this script within a virtual environment
###  like this, for example, from the command line:

## first, setup a virtualenv from shell...
# virtualenv -p python3 venv_activities
# source venv_activities/bin/activate

## with the right packages...
# pip install tensorflow
# pip install keras
# pip install opencv-python
# pip install requests
# pip install matplotlib

## then launch python(3)...
# python


import keras
# import numpy as np
import os
# import cv2

#!export HIP_VISIBLE_DEVICES=0,1 #  For 2 GPU training
os.environ['HIP_VISIBLE_DEVICES'] = '0,1'


import csv
import pandas as pd
import pathlib
import fnmatch


import ssl

### Avoid certificat error (source: https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error)
import requests
requests.packages.urllib3.disable_warnings()


import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


from keras.applications import inception_resnet_v2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt


from tensorflow.python.client import device_lib


from keras.preprocessing import image
# from keras.applications import vgg16
import numpy as np

from tensorflow.keras.utils import multi_gpu_model # Multi-GPU Training ref: https://gist.github.com/mattiavarile/223d9c13c9f1919abe9a77931a4ab6c1



default_path = '/home/alan/Dropbox/KIT/FlickrEU/deepGreen/'
os.chdir(default_path)
# photo_path = default_path + '/Photos_168_retraining'

# split utils from the web
import split_utils
from keras.applications import inception_resnet_v2


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.optimizers import Adam
from keras import metrics

from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping


img_width, img_height = 331, 331

# nb_train_samples = 210
# nb_validation_samples = 99

train_data_dir = "../LabelledData/Costa Rica/Training data_4_edited by Torben for second loop/"
# validation_data_dir = "../LabelledData/Costa Rica/FirstTraining_31Aug2019/validation/"
# train_data_dir = "../LabelledData/Korea/Korea_CameraTrapPhotos/"
#train_data_dir = "../LabelledData/Seattle/Photos_iterative_Sep2019/train/"


# sitename = "Korea"
#sitename = "Seattle"
sitename = "CostaRica"

multiGPU = False
dropout = 0.3

addingClasses = False
loadWeights = True
num_layers_train = 3


batch_size = 32 # proportional to the training sample size.. (64 did not work for Vega56 8GB, 128 did not work for Radeon7 16GB)
val_batch_size = batch_size
epochs = 100

#batch_size means the number of images used in one batch. If you have 320 images and your batch size is 32, you need 10 internal iterations go through the data set once (which is called `one epoch')
# It is set proportional to the training sample size. There are discussions but generally if you can afford, bigger is better. It

# An epoch means the whole input dataset has been used for training the network. There are some heuristics to determine the maximum epoch. Also there is a way to stop the training based on the performance (callled  `Early stopping').


num_classes = 30

# ____________________________________________________________________________________________
# None
# Found 13285 images belonging to 20 classes.
# Found 5708 images belonging to 20 classes.
# ****************
# Class #0 = Amur hedgehog
# Class #1 = Birds
# Class #2 = Car
# Class #3 = Cat
# Class #4 = Chipmunk
# Class #5 = Dog
# Class #6 = Eurasian badger
# Class #7 = Goral
# Class #8 = Human
# Class #9 = Korean hare
# Class #10 = Leopard cat
# Class #11 = Marten
# Class #12 = No animal
# Class #13 = Racoon dog
# Class #14 = Red squirrel
# Class #15 = Rodentia
# Class #16 = Roe dear
# Class #17 = Siberian weasel
# Class #18 = Water deer
# Class #19 = Wild boar
# Class #20 = unidentified
# None
# ****************
# Class #0 = Amphibians
# Class #1 = Backpacking
# Class #2 = Beach
# Class #3 = Bicycles
# Class #4 = Birds
# Class #5 = Birdwatchers
# Class #6 = Boat
# Class #7 = Bustravel
# Class #8 = Camping
# Class #9 = Canoe
# Class #10 = Canyoning
# Class #11 = Caving
# Class #12 = Climbing
# Class #13 = Coffee
# Class #14 = Cows
# Class #15 = Diving
# Class #16 = Fishing
# Class #17 = Flooding
# Class #18 = Flowers
# Class #19 = Hiking
# Class #20 = Horses
# Class #21 = Hotsprings
# Class #22 = Hunting
# Class #23 = Insects
# Class #24 = Kitesurfing
# Class #25 = Landscapes
# Class #26 = Mammals
# Class #27 = Markets
# Class #28 = Monkeys Sloths
# Class #29 = Motorcycles
# Class #30 = Paragliding
# Class #31 = Pplnoactivity
# Class #32 = Rafting
# Class #33 = Reptiles
# Class #34 = Skyboat
# Class #35 = Surfing
# Class #36 = Swimming
# Class #37 = Tourgroups
# Class #38 = Trailrunning
# Class #39 = Volcano
# Class #40 = Waterfall
# Class #41 = Whalewatching
# Class #42 = Ziplining
# Class #43 = otheractivities
# ****************
# the ratio of validation_split is 0.3
# the size of train_dir is 5365
# the size of val_dir is 2329


##### build our classifier model based on pre-trained InceptionResNetV2:


# Load the base pre-trained model

# do not include the top fully-connected layer
# 1. we don't include the top (fully connected) layers of InceptionResNetV2

model = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',input_tensor=None, input_shape=(img_width, img_height, 3))
# Freeze the layers which you don't want to train. Here I am freezing the all layers.
# i.e. freeze all InceptionV3 layers
# model.aux_logits=False


# New dataset is small and similar to original dataset:
# There is a problem of over-fitting, if we try to train the entire network. Since the data is similar to the original data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.
# So lets freeze all the layers and train only the classifier

# first: train only the top layers

# for layer in net_final.layers[:FREEZE_LAYERS]:
#     layer.trainable = False
# for layer in net_final.layers[FREEZE_LAYERS:]:
#     layer.trainable = True

x = model.output

# Now that we have set the trainable parameters of our base network, we would like to add a classifier on top of the convolutional base. We will simply add a fully connected layer followed by a softmax layer with num_classes outputs.


# https://keras.io/applications/#fine-tune-inceptionv3-on-a-new-set-of-classes
# add a global spatial average pooling layer
# https://www.quora.com/What-is-global-average-pooling
# https://arxiv.org/pdf/1312.4400.pdf
# allows you to have the input image be any size, not just a fixed size like 227x227.
# It does through taking an average of every incoming feature map.
# For example, with a 15x15x8 incoming tensor of feature maps, we take the average of each 15x15 matrix slice, giving us an 8 dimensional vector.
# We can now feed this into the fully connected layers.


# New dataset is small and similar to original dataset:
# There is a problem of over-fitting, if we try to train the entire network. Since the data is similar to the original data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.
# So lets freeze all the layers and train only the classifier
#
# # first: train only the top layers (which were randomly initialized)

#


# Notice how that the size of the matrix slices can change, for example, the input might be 32x32x8,
# and we’ll still get an n-of-classes dimensional vector as an output from the global average pooling layer.
# Adding custom Layer
# x = Flatten()(x)
x = GlobalAveragePooling2D()(x) # before dense layer
x = Dense(1024, activation='relu')(x)
# https://datascience.stackexchange.com/questions/28120/globalaveragepooling2d-in-inception-v3-example


if dropout > 0:
    # If the network is stuck at 50% accuracy, there’s no reason to do any dropout.
    # Dropout is a regularization process to avoid overfitting; when underfitting not really useful .
    x = Dropout(0.3)(x) # 30% dropout




if addingClasses:

    num_classes_prev= 16
    # A Dense (fully connected) layer which generates softmax class score for each class
    predictions_old = Dense(num_classes_prev, activation='softmax', name='softmax')(x)


    # creating the final model to train
    model_final = Model(inputs = model.input, outputs = predictions_old)

    ## adding classes
    # @todo possibly wrong? delete the old layers
    predictions_new = Dense(num_classes, activation='softmax', name='softmax')(x)
    model_final = Model(inputs = model.input, outputs = predictions_new)

else:
    # A Dense (fully connected) layer which generates softmax class score for each class

    predictions_new = Dense(num_classes, activation='softmax', name='softmax')(x)
    model_final = Model(inputs = model.input, outputs = predictions_new)


## load trained weights
if loadWeights:


    ## load previously trained weights (old class number)
    model_final.load_weights('../FlickrCNN/TrainedWeights/InceptionResnetV2_CostaRica_retrain_30classes_finetuning_iterative_first_val_acc0.79.h5')
#   model_final.load_weights('../FlickrCNN/TrainedWeights/InceptionResnetV2_Seattle_retrain_instabram_15classes_Sep2019_val_acc0.88.h5')



# We can start fine-tuning convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:


# let's visualize layer names and layer indices to see how many layers
# we should freeze:
# for i, layer in enumerate(model_final.layers):
#     print(i, layer.name)

# Fine tuning (
FREEZE_LAYERS = len(model.layers) - num_layers_train # train the newly added layers and the last few layers

for layer in model_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False



if multiGPU:
    # @todo multi gpu throws an error possibly due to version conflicts..
    model_final = multi_gpu_model(model_final, gpus=2, cpu_merge=True, cpu_relocation=False)

# compile the model (should be done *after* setting layers to non-trainable)




# Need to recompile the model for these modifications to take effect
# Compile the final model using an Adam optimizer, with a low learning rate (since we are 'fine-tuning')
#model_final.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy', 'loss', 'val_acc'])
model_final.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])

# lr: float >= 0. Learning rate.
# beta_1: float, 0 < beta < 1. Generally close to 1.
# beta_2: float, 0 < beta < 1. Generally close to 1.
# epsilon: float >= 0. Fuzz factor.If None, defaults to K.epsilon().
# decay: float >= 0. Learning rate decay over each update.
# amsgrad: boolean. Whether to apply the AMSGrad variantof this algorithm from the paper

# References:
# https://keras.io/optimizers/
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# https://medium.com/@nishantnikhil/adam-optimizer-notes-ddac4fd7218


# we can use SGD with a low learning rate
#model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=1e-4, momentum=0.9), metrics=["accuracy", "categorical_accuracy"])

#
# # https://github.com/keras-team/keras/issues/7924
#
#
# Hi @rnirdhar @ivancruzbht @dilipajm
# I am using the following codes on Keras 2.2.0. The original weights at the last layer are copied back into the new layer. Hope this can help. When model is a sequential model:
#
# from keras.layers import Dense
# import numpy as np
#
# # save the original weights
# weights_bak = model.layers[-1].get_weights()
# nb_classes = model.layers[-1].output_shape[-1]
# model.pop()
# model.add(Dense(nb_classes + 1, activation='softmax'))
# weights_new = model.layers[-1].get_weights()
#
# # copy the original weights back
# weights_new[0][:, :-1] = weights_bak[0]
# weights_new[1][:-1] = weights_bak[1]
#
# # use the average weight to init the new class.
# weights_new[0][:, -1] = np.mean(weights_bak[0], axis=1)
# weights_new[1][-1] = np.mean(weights_bak[1])
#
# model.layers[-1].set_weights(weights_new)
#
# When the model is defined by functional API:
#
# from keras.models import Model
# import numpy as np
#
# # save the original weights
# weights_bak = model.layers[-1].get_weights()
# nb_classes = model.layers[-1].output_shape[-1]
#
# model.layers.pop()
# new_layer = Dense(nb_classes + 1, activation='softmax')
# out = new_layer(model.layers[-1].output)
# inp = model.input
# model = Model(inp, out)
# weights_new = model.layers[-1].get_weights()
#
# # copy the original weights back
# weights_new[0][:, :-1] = weights_bak[0]
# weights_new[1][:-1] = weights_bak[1]
#
# # use the average weight to init the new class.
# weights_new[0][:, -1] = np.mean(weights_bak[0], axis=1)
# weights_new[1][-1] = np.mean(weights_bak[1])
#
# model.layers[-1].set_weights(weights_new)
#

print(model_final.summary())

# Save the model architecture
with open('Model/InceptionResnetV2_retrain_' + sitename + '_architecture_dropout' + '0.3' + '.json', 'w') as f:
    f.write(model_final.to_json())

validation_split = 0.3

# all data in train_dir and val_dir which are alias to original_data. (both dir is temporary directory)
# don't clear base_dir, because this directory holds on temp directory.
base_dir, train_tmp_dir, val_tmp_dir = split_utils.train_valid_split(train_data_dir, validation_split, seed=1)




# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    brightness_range = [0.3, 1],
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30)


# generator for validation data
val_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    brightness_range=[0.3, 1],
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30)


train_generator = train_datagen.flow_from_directory(
     train_tmp_dir,
     target_size = (img_height, img_width),
     batch_size = batch_size,
     class_mode = "categorical")


validation_generator = val_datagen.flow_from_directory(
    val_tmp_dir,
    target_size = (img_height, img_width),
    batch_size=val_batch_size,
    class_mode = "categorical")



# show class indices
print('****************')
for cls, idx in train_generator.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
print('****************')



nb_train_samples = train_generator.n
nb_validation_samples = validation_generator.n

print('the ratio of validation_split is {}'.format(validation_split))
print('the size of train_dir is {}'.format(nb_train_samples))
print('the size of val_dir is {}'.format(nb_validation_samples))


# Save the model according to the conditions
checkpoint = ModelCheckpoint("../FlickrCNN/TrainedWeights/InceptionResnetV2_" + sitename + "_retrain.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=5)


early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# steps per epoch depends on the batch size
steps_per_epoch = int(np.ceil(nb_train_samples / batch_size))
validation_steps_per_epoch = int(np.ceil(nb_validation_samples / batch_size))

# Tensorboard
# If printing histograms, validation_data must be provided, and cannot be a generator.
callback_tb = keras.callbacks.TensorBoard(
        log_dir = "log_dir", # tensorflow log
        histogram_freq=1,    # histogram
        # embeddings_freq=1,
        # embeddings_data=train_generator.labels,
        write_graph=True, write_images=True
    )

# callback_tb_simple = keras.callbacks.TensorBoard(
#         log_dir = "log_dir", # tensorflow log
#         # histogram_freq=1,    # histogram
#         # embeddings_freq=1,
#         # embeddings_data=train_generator.labels,
#         write_graph=True, write_images=True
#     )


# Train the model

history = model_final.fit_generator(
    train_generator,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = validation_steps_per_epoch,
    callbacks = [checkpoint, early]) # , callback_tb



# at this point, the unfreezed layers are well trained.

# Save the model
model_final.save('../FlickrCNN/TrainedWeights/InceptionResnetV2_retrain_' + sitename + '.h5')

# save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('../FlickrCNN/TrainedWeights/InceptionResnetV2_retrain_' + sitename + '.csv')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot( acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training accuracy')
plt.legend()

plt.figure()

plt.plot( loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training loss')
plt.legend()

plt.show()


# Get the filenames from the generator
fnames = validation_generator.filenames

# Get the ground truth from generator
ground_truth = validation_generator.classes

# Get the label to class mapping from the generator
label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v,k) for k, v in label2index.items())

# validation_steps_per_epoch = int(np.ceil(nb_validation_samples / batch_size))

# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator, steps = validation_generator.samples /
                                                                    validation_generator.batch_size, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

#errors = np.where(predicted_classes != ground_truth)[0]
errors = np.where(np.not_equal(predicted_classes, ground_truth[0]))

print("No of errors = {}/{}".format(len(errors),validation_generator.samples))

# Show the errors
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]

    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class])

    original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
    plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()


#### Feature extraction


# @todo feature extraction
























import numpy as np
import os

import cv2
import numpy as np
import pandas as pd
### Avoid certificat error (source: https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error)
import requests
import json

import keras_utils
requests.packages.urllib3.disable_warnings()

import fnmatch
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
from keras.applications.inception_resnet_v2 import decode_predictions

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from keras.preprocessing import image

from keras import backend as k

img_width, img_height = 299, 299

import matplotlib.pyplot as plt


import pathlib
import fnmatch

from shutil import copyfile

# default_path = '/Users/seo-b/Dropbox/KIT/FlickrEU/FlickrCNN'
default_path = '/Users/seo-b/Dropbox/KIT/FlickrEU/deepGreen'
os.chdir(default_path)

photo_path_base = '/Users/seo-b/Dropbox/KIT/FlickrEU/Costa Rica_Data/SamplePhotos/AOI_CellID002146_Golfito_NatPark_Corcovado/'
# photo_path_base = '/home/alan/Dropbox/KIT/FlickrEU/FlickrCNN/Korea_CameraTrapPhotos/Untagged'





# Found 1246 images belonging to 11 classes.
# Found 0 images belonging to 0 classes.
# ****************
# Class #0 = birds
# Class #1 = hedgehog
# Class #2 = leopardcat
# Class #3 = meles
# Class #4 = mustela
# Class #5 = noanimal
# Class #6 = nyctereutes
# Class #7 = person
# Class #8 = roedear
# Class #9 = waterdeer
# Class #10 = wildboar

# classes = ["birds", "hedgehog", "leopardcat", "meles", "mustela", "noanimal", "nyctereutes", "person", "roedear",
#            "waterdeer", "wildboar"]

# Imagenet class labels
imagenet_labels_filename = "DATA/imagenet_class_index.json"
with open(imagenet_labels_filename) as f:
    CLASS_INDEX = json.load(f)

classlabel = []
for i in range(CLASS_INDEX.__len__()):
    classlabel.append(CLASS_INDEX[str(i)][1])
classes = np.array(classlabel)


num_classes = len(classes)

top = 7 # choose top-seven classes


batch_size = 32 # number of images for one batch prediction

#Load the inception_v4_resnet_v2
model_trained = inception_resnet_v2.InceptionResNetV2(weights='imagenet')





# Load the Inception_V3

# Load the base pre-trained model
# do not include the top fully-connected layer
# model = inception_v3.InceptionV3(include_top=False,  input_shape=(img_width, img_height, 3))
# Freeze the layers which you don't want to train. Here I am freezing the all layers.


# New dataset is small and similar to original dataset:
# There is a problem of over-fitting, if we try to train the entire network. Since the data is similar to the original data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.
# So lets freeze all the layers and train only the classifier
#
# # first: train only the top layers (which were randomly initialized)
# # i.e. freeze all InceptionV3 layers
# for layer in model.layers[:]:
#     layer.trainable = False
# # Adding custom Layer
# x = model.output
# # add a global spatial average pooling layer
# x = GlobalAveragePooling2D()(x)
#
# # let's add a fully-connected layer
# x = Dense(1024, activation='relu')(x)
# # and a logistic layer -- let's say we have n classes
# predictions = Dense(num_classes, activation='softmax')(x)
#
#
# # creating the final model
# # this is the model we will train
# model_final = Model(inputs = model.input, outputs = predictions)
#
#
#
# model_final.load_weights('TrainedWeights/InceptionResnetV2_retrain_instagram_epoch150_acc0.97.h5')
#
modelname = "InceptionResnetV2"
# dataname = "Photos_338"
dataname = "FlickrCR"

# filename = 'photoid_19568808955.jpg' # granpa
filename = 'photoid_23663993529.jpg'  # bridge

foldernames = os.listdir(photo_path_base)

for foldername in foldernames:

    photo_path_aoi = photo_path_base + "/" + foldername

    # years = os.listdir(photo_path_aoi)
    #
    year = ""
    # photo_path = photo_path_aoi + "/" + year
    photo_path = photo_path_aoi

    ### Read filenames
    filenames = os.listdir(photo_path)

    filenames1 = fnmatch.filter(filenames, "*.jpg")
    filenames2 = fnmatch.filter(filenames, "*.JPG")

    filenames = filenames1 + filenames2
    n_files = len(filenames)

    # load all images into a list
    images = []
    for img in filenames:
        img = os.path.join(photo_path, img)

        # load an image in PIL format
        img = image.load_img(img, target_size=(img_width, img_height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # prepare the image (normalisation for channels)
        img_preprocessed = inception_resnet_v2.preprocess_input(img.copy())
        images.append(img_preprocessed)

    images_vstack = np.vstack(images)


    # stack up images list to pass for prediction
    predictions = model_trained.predict(images_vstack, batch_size=batch_size)
    print(predictions)

    predictions.shape

    ## top selected classes
    top_classes_idx_arr = np.argsort(predictions)[:, ::-1][:,:top]

    top_classes_arr = classes[top_classes_idx_arr]

    # create an empty array
    top_classes_probs_arr = np.empty([n_files, top])
    top_classes_probs_arr[:] = 0

    for i in range(0, n_files):
        top_classes_probs_arr[i,] = predictions[i, [top_classes_idx_arr[i,]]]


    # np.argsort(predictions)[:, ::-1][:,:top][0, :]

    # chainlink_fence', 'worm_fence', 'lakeside', 'seashore', 'stone_wall', 'cliff', 'breakwater']
    # Out[61]: array([489, 912, 975, 978, 825, 972, 460])
    top_classes_arr[0, :]
    top_classes_probs_arr[0, :]





    for filename in filenames[:]:

        fname = photo_path + "/" + filename

        if os.path.isfile(fname):

            # load an image in PIL format
            try:
                original = load_img(fname, target_size=(img_width, img_height))
            except OSError:
                print("Bad file - Try again..." + fname)
                continue

            # plt.imshow(original)
            # plt.show()

            # @todo skip readily done files.

            # convert the PIL image to a numpy array
            # IN PIL - image is in (width, height, channel)
            # In Numpy - image is in (height, width, channel)
            numpy_image = img_to_array(original)

            # Convert the image / images into batch format
            # expand_dims will add an extra dimension to the data at a particular axis
            # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
            # Thus we add the extra dimension to the axis 0.
            image_batch = np.expand_dims(numpy_image, axis=0)

            # prepare the image (normalisation for channels)
            processed_image = inception_resnet_v2.preprocess_input(image_batch.copy())

            # get the predicted probabilities for each class
            # @todo predict_on_batch
            predictions = model_trained.predict(processed_image)



            # stack up images list to pass for prediction
            images = np.vstack(images)
            predictions = model_trained.predict(images, batch_size=10)
            print(predictions)

            ## top selected classes
            top_classes_idx = np.argsort(predictions[0])[::-1][:top]
            top_classes = classes[top_classes_idx]
            top_classes_probs = predictions[0][top_classes_idx]

            # print predictions
            # dominant_feature_idx = np.argmax(predictions[0])
            dominant_feature_idx = top_classes_idx[0]

            # This is the dominant entry in the prediction vector
            dominant_output = model_trained.output[:, dominant_feature_idx]

            # convert the probabilities to class labels
            predicted_class = classes[dominant_feature_idx]
            print('Predicted:', predicted_class)

            # The is the output feature map of the `conv_7b_ac` layer,
            # the last convolutional layer in InceptionResnetV2
            last_conv_layer = model_trained.get_layer('conv_7b_ac')

            # This is the gradient of the dominant class with regard to
            # the output feature map of `conv_7b_ac`
            grads = k.gradients(dominant_output, last_conv_layer.output)[0]

            # This is a vector of shape (1536,), where each entry
            # is the mean intensity of the gradient over a specific feature map channel
            pooled_grads = k.mean(grads, axis=(0, 1, 2))

            # This function allows us to access the values of the quantities we just defined:
            # `pooled_grads` and the output feature map of `conv_7b_ac`,
            # given a sample image
            iterate = k.function([model_trained.input], [pooled_grads, last_conv_layer.output[0]])

            # These are the values of these two quantities, as Numpy arrays,
            # given our sample image of two elephants
            pooled_grads_value, conv_layer_output_value = iterate([processed_image])

            # We multiply each channel in the feature map array
            # by "how important this channel is" with regard to the elephant class
            for i in range(1536):
                conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

            # The channel-wise mean of the resulting feature map
            # is our heatmap of class activation
            heatmap = np.mean(conv_layer_output_value, axis=-1)

            # For visualization purpose, we will also normalize the heatmap between 0 and 1:

            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            plt.matshow(heatmap)
            plt.show()

            plt.imshow(original)
            plt.show()

            # We use cv2 to load the original image
            img = cv2.imread(fname)

            # We resize the heatmap to have the same size as the original image
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

            # We convert the heatmap to RGB
            heatmap = np.uint8(255 * heatmap)

            # We apply the heatmap to the original image
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # 0.4 here is a heatmap intensity factor
            superimposed_img = heatmap * 0.4 + img

            # Choose a font
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Draw the text

            for x in range(0, top):
                tags = top_classes[x] + ":" + str((top_classes_probs[x] * 100).round(2)) + "%"
                cv2.putText(superimposed_img, tags, (10, 25 * x + int(img.shape[0] / 10)), font, 0.7, (255, 255, 255),
                            2)  # ,cv2.LINE_AA)

            out_filepath = "Result/Heatmap_" + modelname + "/" + foldername + "/" + year + "/" + predicted_class
            if not (os.path.exists(out_filepath)):
                os.makedirs(out_filepath, exist_ok=False)

            # Save the image to disk
            cv2.imwrite(
                "Result/Heatmap_" + modelname + "/" + foldername + "/" + year + "/" + predicted_class + "/AttentionMap_" + predicted_class + "_" + filename,
                superimposed_img)

            out_orgfilepath = "Result/Original_" + modelname + "/" + foldername + "/" + year + "/" + predicted_class
            if not (os.path.exists(out_orgfilepath)):
                os.makedirs(out_orgfilepath, exist_ok=False)
            copyfile(fname, out_orgfilepath + '/' + filename)

    else:
        print(fname + " error?")

    for filename in filenames[:]:

        fname = photo_path + "/" + filename

        if os.path.isfile(fname):

            # load an image in PIL format
            # original = load_img(filename, target_size=(299, 299))
            try:
                original = load_img(fname, target_size=(662, 662))
            except OSError:
                print("Bad file - Try again..." + fname)
                continue

            # convert the PIL image to a numpy array
            # IN PIL - image is in (width, height, channel)
            # In Numpy - image is in (height, width, channel)
            numpy_image = img_to_array(original)

            # Convert the image / images into batch format
            # expand_dims will add an extra dimension to the data at a particular axis
            # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
            # Thus we add the extra dimension to the axis 0.
            image_batch = np.expand_dims(numpy_image, axis=0)

            # prepare the image (normalisation for channels)
            processed_image = inception_resnet_v2.preprocess_input(image_batch.copy())

            # get the predicted probabilities for each class
            predictions = model_trained.predict(processed_image)
            # print predictions
            dominant_feature_idx = np.argmax(predictions[0])

            # convert the probabilities to class labels
            predicted_class = classes[dominant_feature_idx]
            print('Predicted:', predicted_class)

            df = pd.DataFrame(predictions[0]).transpose()
            name_csv = default_path + "/Result/Tag_" + modelname + "/" + filename + "_" + predicted_class + ".csv"

            # df.to_csv(name_csv)
            header = classes
            df.columns = classes
            df.to_csv(name_csv, index=False, columns=header)

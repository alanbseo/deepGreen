import os

import numpy as np
import pandas as pd
### Avoid certificat error (source: https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error)
import requests

requests.packages.urllib3.disable_warnings()

from keras.applications import inception_resnet_v2

from keras.preprocessing import image

img_width, img_height = 331, 331
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

import fnmatch

from shutil import copyfile

default_path = '/home/alan/Dropbox/KIT/FlickrEU/deepGreen'

modelname = "InceptionResnetV2_dropout30"


#!export HIP_VISIBLE_DEVICES=0,1 #  For 2 GPU training
# os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
os.environ['HIP_VISIBLE_DEVICES'] = '0'

# dataname = "FlickrCR_Photos_All"
# model_json = "Model/InceptionResnetV2_retrain_CostaRica_architecture_dropout0.3.json"
# photo_path_base = '/DATA2TB/FlickrCR_download/Nov2019_V2_Photo/'
# out_path_base = "/DATA2TB/FlickrCR_Tagging_Nov2019/"
#
# trainedweights_name = '../FlickrCNN/TrainedWeights/InceptionResnetV2_CostaRica_retrain_30classes_finetuning_iterative_fourth_val_acc0.84.h5'
# # trainedweights_name = '../FlickrCNN/TrainedWeights/InceptionResnetV2_CostaRica_retrain_30classes_finetuning_iterative_final_val_acc0.82.h5'
# classes = ["Amphibians", "Beach", "Bicycles", "Birds", "Boat tours", "Bustravel",
#            "Canoe", "Cows", "Diving", "Dog walking", "Fishing", "Flowers",
#            "Horses", "Insects", "Landscapes", "Mammals",
#            "Miscellaneous",
#            "Monkeys Sloths", "Motorcycles", "Pplnoactivity", "Rafting", "Reptiles",
#            "Surfing",
#            "Swimming", "Trees-leafs", "Volcano", "Waterfall", "Whalewatching", "Ziplining",
#            "boat other"]

# Seattle
dataname = "FlickrSeattle_Photos_All"
model_json = "Model/InceptionResnetV2_retrain_Seattle_architecture_dropout0.3.json"
photo_path_base = '/home/alan/Dropbox/KIT/FlickrEU/UnlabelledData/Seattle/NewTestPhotos'
# photo_path_base = '/home/alan/Dropbox/KIT/FlickrEU/FlickrCNN/Seattle/FlickrSeattle_download/Photos/AOI_CellID_Merged/' # FlickrSeattle_Photos_Flickr_All/'
#
trainedweights_name = "../FlickrCNN/TrainedWeights/InceptionResnetV2_Seattle_retrain_instabram_15classes_Okt2019_val_acc0.88.h5"
# trainedweights_name = "../FlickrCNN/TrainedWeights/InceptionResnetV2_Seattle_retrain_instabram_15classes_Weighted_Nov2019_val_acc0.87.h5"
#
out_path_base = "/DATA2TB/FlickrSeattle_Tagging_Feb2020/"
#
classes = ["backpacking", "birdwatching", "boating", "camping", "fishing", "flooding", "hiking", "horseriding",
           "mtn_biking", "noactivity", "otheractivities", "pplnoactivity", "rock climbing", "swimming",
           "trailrunning"]


trainedweights_name = "../FlickrCNN/TrainedWeights/InceptionResnetV2_Seattle_retrain_instabram_15classes_Okt2019_val_acc0.88.h5"


# Saxony
#dataname = "FlickrSaxnony"
#model_json = "Model/InceptionResnetV2_retrain_Saxony_architecture_dropout0.2.json"
#photo_path_base = '/DATA2TB/FlickrEU_download/FlickrEU_Jan2019_V1_Photo_Sachsen'
 #trainedweights_name = "../FlickrCNN/TrainedWeights/InceptionResnetV2_Saxony_retrain_19classes_Dec2019_val_acc0.85_3layers.h5"
#out_path_base = "/DATA2TB/FlickrSaxony_Tagging_Dec2019_v3/"
#
# classes = ["aesthetic landscape", "climbing", "cycling", "hiking", "horseback Riding", "nature appreciation", "water-related activities",
          # "winter sports", "non-CES"]

# Class #0 = Aesthetic Landscape
# Class #1 = Climbing
# Class #2 = Cycling
# Class #3 = Hiking
# Class #4 = Horseback Riding
# Class #5 = Nature Appreciation
# Class #6 = Water-Related Activities
# Class #7 = Winter Sports
# Class #8 = non-CES

#classes = ["ball sports", "birds", "camping", "cars", "climbing", "cycling", "dog walking", "hiking", "horseback Riding", "landscape",
#          "mammals", "motocycles", "nature", "noactivity", "otheractivities", "pplnoactivity", "reptiles", "water",
#           "winter sports"]
#
# ****************
# Class #0 = ball sports
# Class #1 = birds
# Class #2 = camping
# Class #3 = cars
# Class #4 = climbing
# Class #5 = cycling
# Class #6 = dog walking
# Class #7 = hiking
# Class #8 = horseback riding
# Class #9 = landscape
# Class #10 = mammals
# Class #11 = motorcycles
# Class #12 = nature
# Class #13 = noactivity
# Class #14 = otheractivities
# Class #15 = pplnoactivity
# Class #16 = reptiles
# Class #17 = water
# Class #18 = winter sports
# ****************


# Korea
# dataname = "Korea"
# model_json = 'Model/InceptionResnetV2_retrain_Korea_architecture_dropout0.2.json'
# photo_path_base = '/DATA2TB/CameraTraps_Korea/Photos/NewCameraTrapPhotos_unlabelled_Dec2019'
#
# trainedweights_name = "../FlickrCNN/TrainedWeights/InceptionResnetV2_retrain_Korea_26classes_Dec2019_val_acc0.74.h5"
# #_Korea_21classes_Dec2019_val_acc0.65.h5"
# #
# out_path_base = "/DATA2TB/CameraTraps_Korea_Tagging2019_v5/"
# #
#
#
# classes = ["Amur hedgehog", "Bat", "Car", "Cat", "Chipmunk", "Dog", "Eurasian badger", "Eurasian Otter", "Goral", "Korean hare",
#            "Least weasel",
#             "Marten", "Musk deer",  "No animal", "Red squirrel", "Rodentia",  "Flying squirrel",  "Siberian weasel", "Birds", "Human", "Leopard cat",
#             "Racoon dog", "Roe dear", "Unidentified", "Water deer", "Wild boar"]
# #
# ****************
# Class #0 = Amur hedgehog
# Class #1 = Bat
# Class #2 = Car
# Class #3 = Cat
# Class #4 = Chipmunk
# Class #5 = Dog
# Class #6 = Eurasian badger
# Class #7 = Eurasian otter
# Class #8 = Goral
# Class #9 = Korean hare
# Class #10 = Least weasel
# Class #11 = Marten
# Class #12 = Musk deer
# Class #13 = No animal
# Class #14 = Red squirrel
# Class #15 = Rodentia
# Class #16 = Siberian flying squirrel
# Class #17 = Siberian weasel
# Class #18 = birds
# Class #19 = human
# Class #20 = leopard cat
# Class #21 = racoon dog
# Class #22 = roe dear
# Class #23 = unidentified
# Class #24 = water deer
# Class #25 = wild boar
os.chdir(default_path)

out_path = out_path_base + modelname + "/" + dataname + "/"

prediction_batch_size = 32  # to increase the speed of tagging .
# number of images for one batch prediction

top = 10  # 10  print top-n classes


# Class #0 = backpacking
# Class #1 = birdwatching
# Class #2 = boating
# Class #3 = camping
# Class #4 = fishing
# Class #5 = flooding
# Class #6 = hiking
# Class #7 = horseriding
# Class #8 = mtn_biking
# Class #9 = noactivity
# Class #10 = otheractivities
# Class #11 = pplnoactivity
# Class #12 = rock climbing
# Class #13 = swimming
# Class #14 = trailrunning


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
# classes = ["Amphibians", "Backpacking", "Beach", "Bicycles", "Birds", "Birdwatchers", "Boat", "Bustravel",
#            "Camping",
#            "Canoe", "Canyoning", "Caving", "Climbing", "Coffee", "Cows", "Diving", "Fishing", "Flooding", "Flowers",
#            "Hiking", "Horses", "Hotsprings", "Hunting", "Insects", "Kitesurfing", "Landscapes", "Mammals",
#            "Markets",
#            "Monkeys Sloths", "Motorcycles", "Paragliding", "Pplnoactivity", "Rafting", "Reptiles", "Skyboat",
#            "Surfing",
#            "Swimming", "Tourgroups", "Trailrunning", "Volcano", "Waterfall", "Whalewatching", "Ziplining",
#            "otheractivities"]
#

#
# Found 4933 images belonging to 30 classes.
# Found 2139 images belonging to 30 classes.
# ****************
# Class #0 = Amphibians
# Class #1 = Beach
# Class #2 = Bicycles
# Class #3 = Birds
# Class #4 = Boat tours
# Class #5 = Bustravel
# Class #6 = Canoe
# Class #7 = Cows
# Class #8 = Diving
# Class #9 = Dog walking
# Class #10 = Fishing
# Class #11 = Flowers
# Class #12 = Horses
# Class #13 = Insects
# Class #14 = Landscape
# Class #15 = Mammals
# Class #16 = Miscellanous
# Class #17 = Monkeys Sloths
# Class #18 = Motorcycles
# Class #19 = Pplnoactivity
# Class #20 = Rafting
# Class #21 = Reptiles
# Class #22 = Surfing
# Class #23 = Swimming
# Class #24 = Trees - leafs
# Class #25 = Volcano
# Class #26 = Waterfall
# Class #27 = Whalewatching
# Class #28 = Ziplining
# Class #29 = boat other
# ****************



classes_arr = np.array(classes)
# # Imagenet class labels
# imagenet_labels_filename = "Data/imagenet_class_index.json"
# with open(imagenet_labels_filename) as f:
#     CLASS_INDEX = json.load(f)
#
# classlabel = []
# for i in range(CLASS_INDEX.__len__()):
#     classlabel.append(CLASS_INDEX[str(i)][1])
# classes = np.array(classlabel)

num_classes = len(classes)


##### Predict

# Load the retrained CNN model

# Model reconstruction from JSON file
# with open(model_json, 'r') as f:
#    model_trained = model_from_json(f.read())



model_trained = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',input_tensor=None, input_shape=(img_width, img_height, 3))
x = model_trained.output
x = GlobalAveragePooling2D()(x) # before dense layer
x = Dense(1024, activation='relu')(x)
predictions_new = Dense(num_classes, activation='softmax', name='softmax')(x)
model_trained = Model(inputs=model_trained.input, outputs=predictions_new)

# Load weights into the new model
model_trained.load_weights(trainedweights_name)

# model_final = multi_gpu_model(model_final, gpus=2, cpu_merge=True, cpu_relocation=False)


def onlyfolders(path):
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            yield file


def onlyfiles(path):
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            yield file



from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

foldernames = os.listdir(photo_path_base)

# f_idx = 0

for f_idx in range(0, len(foldernames)):

    foldername = foldernames[f_idx]
    print(f_idx)
    print(foldername)
    photo_path_aoi = photo_path_base + "/" + foldername

    ### Read filenames
    # filenames = os.listdir(photo_path_aoi)
    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(photo_path_aoi)) for f in fn]

    if len(filenames) == 0:
        continue  # skip the folder

    filenames1 = fnmatch.filter(filenames, "*.jpg")
    filenames2 = fnmatch.filter(filenames, "*.JPG")

    filenames = filenames1 + filenames2

    filenames = filenames1 + filenames2

    base_filenames = list(map(os.path.basename, filenames))

    n_files = len(filenames)

    prediction_steps_per_epoch = int(np.ceil(n_files / prediction_batch_size))

    # load all images into a list
    batch_size_folder = min(n_files, prediction_batch_size)  # n_files can be smaller than the batch size

    for step_start_idx in range(0, n_files, batch_size_folder):

        end_idx = min(step_start_idx + batch_size_folder, n_files)

        print(step_start_idx)
        print(end_idx)

        if step_start_idx == end_idx:

            filenames_batch = [filenames[step_start_idx]]
        else:

            filenames_batch = filenames[step_start_idx:end_idx]

        bsize_tmp = min(batch_size_folder, len(filenames_batch))  # for the last batch

        images = []

        for img_name in filenames_batch:
            print(img_name)
            img_name = os.path.join(photo_path_aoi, img_name)

            # load an image in PIL format
            img = image.load_img(img_name, target_size=(img_width, img_height))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            # prepare the image (normalisation for channels)
            img_preprocessed = inception_resnet_v2.preprocess_input(img.copy())
            images.append(img_preprocessed)

        images_vstack = np.vstack(images)

        # stack up images list to pass for prediction
        predictions = model_trained.predict(images_vstack, batch_size=bsize_tmp)

        # predictions.shape

        ## top selected classes
        top_classes_idx_arr = np.argsort(predictions)[:, ::-1][:, :top]

        top_classes_arr = classes_arr[top_classes_idx_arr]
        print(top_classes_arr)

        # create an empty array
        top_classes_probs_arr = np.empty([bsize_tmp, top])
        top_classes_probs_arr[:] = 0

        for i in range(0, bsize_tmp):
            top_classes_probs_arr[i,] = predictions[i, [top_classes_idx_arr[i,]]]

        # np.argsort(predictions)[:, ::-1][:,:top][0, :]

        # chainlink_fence', 'worm_fence', 'lakeside', 'seashore', 'stone_wall', 'cliff', 'breakwater']
        # Out[61]: array([489, 912, 975, 978, 825, 972, 460])
        top_classes_arr[0, :]
        top_classes_probs_arr[0, :]

        predicted_class_v = top_classes_arr[:, 0] # top1
        predicted_class_top2_v = top_classes_arr[:, 1] # top2

        #print('Predicted:', predicted_class_v)


        # 2nd-level
        # kind of equivalent to `sapply()' in R
        def foo_get_predicted_filename(x, x2):
            # return (out_path + "Result/" + modelname + "/ClassifiedPhotos/" + foldername + "/" + x)
            return (out_path + "Result/" + "/ClassifiedPhotos/" + "/" + x + "/2ndClass_" +x2 )


        predicted_filenames = list(map(foo_get_predicted_filename, predicted_class_v, predicted_class_top2_v))
        save_folder_names = list(map(os.path.basename, predicted_filenames))

        # create necessary folders
        # for i in range(0, n_files):
        #     if not (os.path.exists(save_folder_names[i])):
        #         os.makedirs(save_folder_names[i], exist_ok=False)

        for i in range(0, bsize_tmp):

            save_folder = predicted_filenames[i]
            print(save_folder)

            if not (os.path.exists(save_folder)):
                os.makedirs(save_folder, exist_ok=False)

            copyfile(filenames_batch[i], predicted_filenames[i] + '/' + os.path.basename(filenames_batch[i]) )

        arr_tmp = pd.DataFrame(np.concatenate((top_classes_arr, top_classes_probs_arr), axis=1))

        if step_start_idx == 0:
            arr_aoi = arr_tmp
        else:
            arr_aoi = np.concatenate((arr_aoi, arr_tmp), axis=0)

    # Write csv files

    name_csv = out_path + "Result/" + modelname + "/CSV/" + foldername + ".csv"
    if not (os.path.exists(os.path.dirname(name_csv))):
        os.makedirs(os.path.dirname(name_csv), exist_ok=False)

    # Write a Pandas data frame
    df_aoi = pd.concat([pd.DataFrame(base_filenames), pd.DataFrame(arr_aoi)], axis=1)
    header = np.concatenate(
        (["Filename"], ["Top1", "Top2", "Top3", "Top4", "Top5", "Top6", "Top7", "Top8", "Top9", "Top10"],
         ["Prob1", "Prob2", "Prob3", "Prob4", "Prob5", "Prob6", "Prob7", "Prob8", "Prob9", "Prob10"]))

    df_aoi.columns = header

ll
    # @todo attention map




#
#
# for filename in filenames[:]:
#
#     fname = photo_path + "/" + filename
#
#     if os.path.isfile(fname):
#
#         # load an image in PIL format
#         try:
#             original = load_img(fname, target_size=(img_width, img_height))
#         except OSError:
#             print("Bad file - Try again..." + fname)
#             continue
#
#     # plt.imshow(original)
#     # plt.show()
#
#     # @todo skip readily done files.
#
#     # convert the PIL image to a numpy array
#     # IN PIL - image is in (width, height, channel)
#     # In Numpy - image is in (height, width, channel)
#     numpy_image = img_to_array(original)
#
#     # Convert the image / images into batch format
#     # expand_dims will add an extra dimension to the data at a particular axis
#     # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
#     # Thus we add the extra dimension to the axis 0.
#     image_batch = np.expand_dims(numpy_image, axis=0)
#
#     # prepare the image (normalisation for channels)
#     processed_image = inception_resnet_v2.preprocess_input(image_batch.copy())
#
#     # get the predicted probabilities for each class
#     # @todo predict_on_batch
#     predictions = model_trained.predict(processed_image)
#
#     # stack up images list to pass for prediction
#     images = np.vstack(images)
#     predictions = model_trained.predict(images, batch_size=10)
#     print(predictions)
#
#     ## top selected classes
#     top_classes_idx = np.argsort(predictions[0])[::-1][:top]
#     top_classes = classes[top_classes_idx]
#     top_classes_probs = predictions[0][top_classes_idx]
#
#     # print predictions
#     # dominant_feature_idx = np.argmax(predictions[0])
#     dominant_feature_idx = top_classes_idx[0]
#
#     # This is the dominant entry in the prediction vector
#     dominant_output = model_trained.output[:, dominant_feature_idx]
#
#     # convert the probabilities to class labels
#     predicted_class = classes[dominant_feature_idx]
#     print('Predicted:', predicted_class)
#
#     # The is the output feature map of the `conv_7b_ac` layer,
#     # the last convolutional layer in InceptionResnetV2
#     last_conv_layer = model_trained.get_layer('conv_7b_ac')
#
#     # This is the gradient of the dominant class with regard to
#     # the output feature map of `conv_7b_ac`
#     grads = k.gradients(dominant_output, last_conv_layer.output)[0]
#
#     # This is a vector of shape (1536,), where each entry
#     # is the mean intensity of the gradient over a specific feature map channel
#     pooled_grads = k.mean(grads, axis=(0, 1, 2))
#
#     # This function allows us to access the values of the quantities we just defined:
#     # `pooled_grads` and the output feature map of `conv_7b_ac`,
#     # given a sample image
#     iterate = k.function([model_trained.input], [pooled_grads, last_conv_layer.output[0]])
#
#     # These are the values of these two quantities, as Numpy arrays,
#     # given our sample image of two elephants
#     pooled_grads_value, conv_layer_output_value = iterate([processed_image])
#
#     # We multiply each channel in the feature map array
#     # by "how important this channel is" with regard to the elephant class
#     for i in range(1536):
#         conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
#
#     # The channel-wise mean of the resulting feature map
#     # is our heatmap of class activation
#     heatmap = np.mean(conv_layer_output_value, axis=-1)
#
#     # For visualization purpose, we will also normalize the heatmap between 0 and 1:
#
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)
#     plt.matshow(heatmap)
#     plt.show()
#
#     plt.imshow(original)
#     plt.show()
#
#     # We use cv2 to load the original image
#     img = cv2.imread(fname)
#
#     # We resize the heatmap to have the same size as the original image
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#
#     # We convert the heatmap to RGB
#     heatmap = np.uint8(255 * heatmap)
#
#     # We apply the heatmap to the original image
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#
#     # 0.4 here is a heatmap intensity factor
#     superimposed_img = heatmap * 0.4 + img
#
#     # Choose a font
#     font = cv2.FONT_HERSHEY_SIMPLEX
#
#     # Draw the text
#
#     for x in range(0, top):
#         tags = top_classes[x] + ":" + str((top_classes_probs[x] * 100).round(2)) + "%"
#         cv2.putText(superimposed_img, tags, (10, 25 * x + int(img.shape[0] / 10)), font, 0.7, (255, 255, 255),
#                     2)  # ,cv2.LINE_AA)
#
#     out_filepath = "Result/Heatmap_" + modelname + "/" + foldername + "/" + year + "/" + predicted_class
#     if not (os.path.exists(out_filepath)):
#         os.makedirs(out_filepath, exist_ok=False)
#
#     # Save the image to disk
#     cv2.imwrite(
#         "Result/Heatmap_" + modelname + "/" + foldername + "/" + year + "/" + predicted_class + "/AttentionMap_" + predicted_class + "_" + filename,
#         superimposed_img)
#
#     out_orgfilepath = "Result/Original_" + modelname + "/" + foldername + "/" + year + "/" + predicted_class
#     if not (os.path.exists(out_orgfilepath)):
#         os.makedirs(out_orgfilepath, exist_ok=False)
#     copyfile(fname, out_orgfilepath + '/' + filename)
#
#     else:
#     print(fname + " error?")
#
#     for filename in filenames[:]:
#
#     fname = photo_path + "/" + filename
#
#     if os.path.isfile(fname):
#
#     # load an image in PIL format
#     # original = load_img(filename, target_size=(299, 299))
#     try:
#         original = load_img(fname, target_size=(662, 662))
#     except OSError:
#         print("Bad file - Try again..." + fname)
#         continue
#
#     # convert the PIL image to a numpy array
#     # IN PIL - image is in (width, height, channel)
#     # In Numpy - image is in (height, width, channel)
#     numpy_image = img_to_array(original)
#
#     # Convert the image / images into batch format
#     # expand_dims will add an extra dimension to the data at a particular axis
#     # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
#     # Thus we add the extra dimension to the axis 0.
#     image_batch = np.expand_dims(numpy_image, axis=0)
#
#     # prepare the image (normalisation for channels)
#     processed_image = inception_resnet_v2.preprocess_input(image_batch.copy())
#
#     # get the predicted probabilities for each class
#     predictions = model_trained.predict(processed_image)
#     # print predictions
#     dominant_feature_idx = np.argmax(predictions[0])
#
#     # convert the probabilities to class labels
#     predicted_class = classes[dominant_feature_idx]
#     print('Predicted:', predicted_class)
#
#     df = pd.DataFrame(predictions[0]).transpose()
#     name_csv = default_path + "/Result/Tag_" + modelname + "/" + filename + "_" + predicted_class + ".csv"
#
#     # df.to_csv(name_csv)
#     header = classes
#     df.columns = classes
#     df.to_csv(name_csv, index=False, columns=header)
#
# for filename in filenames[:]:
#
#     fname = photo_path + "/" + filename
#
#     if os.path.isfile(fname):
#
#         # load an image in PIL format
#         try:
#             original = load_img(fname, target_size=(img_width, img_height))
#         except OSError:
#             print("Bad file - Try again..." + fname)
#             continue
#
#         # plt.imshow(original)
#         # plt.show()
#
#         # @todo skip readily done files.
#
#         # convert the PIL image to a numpy array
#         # IN PIL - image is in (width, height, channel)
#         # In Numpy - image is in (height, width, channel)
#         numpy_image = img_to_array(original)
#
#         # Convert the image / images into batch format
#         # expand_dims will add an extra dimension to the data at a particular axis
#         # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
#         # Thus we add the extra dimension to the axis 0.
#         image_batch = np.expand_dims(numpy_image, axis=0)
#
#         # prepare the image (normalisation for channels)
#         processed_image = inception_resnet_v2.preprocess_input(image_batch.copy())
#
#         # get the predicted probabilities for each class
#         # @todo predict_on_batch
#         predictions = model_trained.predict(processed_image)
#
#         # print predictions
#         dominant_feature_idx = np.argmax(predictions[0])
#
#         # This is the dominant entry in the prediction vector
#         dominant_output = model_trained.output[:, dominant_feature_idx]
#
#         # convert the probabilities to class labels
#         predicted_class = classes[dominant_feature_idx]
#         print('Predicted:', predicted_class)
#
#         # The is the output feature map of the `conv_7b_ac` layer,
#         # the last convolutional layer in InceptionResnetV2
#         last_conv_layer = model_trained.get_layer('conv_7b_ac')
#
#         # This is the gradient of the dominant class with regard to
#         # the output feature map of `conv_7b_ac`
#         grads = k.gradients(dominant_output, last_conv_layer.output)[0]
#
#         # This is a vector of shape (1536,), where each entry
#         # is the mean intensity of the gradient over a specific feature map channel
#         pooled_grads = k.mean(grads, axis=(0, 1, 2))
#
#         # This function allows us to access the values of the quantities we just defined:
#         # `pooled_grads` and the output feature map of `conv_7b_ac`,
#         # given a sample image
#         iterate = k.function([model_trained.input], [pooled_grads, last_conv_layer.output[0]])
#
#         # These are the values of these two quantities, as Numpy arrays,
#         # given our sample image of two elephants
#         pooled_grads_value, conv_layer_output_value = iterate([processed_image])
#
#         # We multiply each channel in the feature map array
#         # by "how important this channel is" with regard to the elephant class
#         for i in range(1536):
#             conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
#
#         # The channel-wise mean of the resulting feature map
#         # is our heatmap of class activation
#         heatmap = np.mean(conv_layer_output_value, axis=-1)
#
#         # For visualization purpose, we will also normalize the heatmap between 0 and 1:
#
#         heatmap = np.maximum(heatmap, 0)
#         heatmap /= np.max(heatmap)
#         # plt.matshow(heatmap)
#         # plt.show()
#
#         # We use cv2 to load the original image
#         img = cv2.imread(fname)
#
#         # We resize the heatmap to have the same size as the original image
#         heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#
#         # We convert the heatmap to RGB
#         heatmap = np.uint8(255 * heatmap)
#
#         # We apply the heatmap to the original image
#         heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#
#         # 0.4 here is a heatmap intensity factor
#         superimposed_img = heatmap * 0.4 + img
#
#         # Choose a font
#         font = cv2.FONT_HERSHEY_COMPLEX
#
#         # Draw the text
#
#         for x in range(0, num_classes):
#             tags = classes[x] + ":" + str((predictions[0][x] * 100).round(2)) + "%"
#             cv2.putText(superimposed_img, tags, (10, 25 * x + int(img.shape[0] / 10)), font, 0.5, (255, 255, 255),
#                         2)  # ,cv2.LINE_AA)
#
#         out_filepath = out_path + "Result/Heatmap_" + modelname + "/" + predicted_class
#         if not (os.path.exists(out_filepath)):
#             os.makedirs(out_filepath,
#                         exist_ok=False)
#
#         # Save the image to disk
#         cv2.imwrite(out_filepath + "/AttentionMap_" + predicted_class + "_" + filename, superimposed_img)
#
#         copyfile(fname, out_filepath + '/' + filename)
#
#     else:
#         print(fname + " error?")
#
# for filename in filenames[:]:
#
#     fname = photo_path + "/" + filename
#
#     if os.path.isfile(fname):
#
#         # load an image in PIL format
#         # original = load_img(filename, target_size=(299, 299))
#         try:
#             original = load_img(fname, target_size=(662, 662))
#         except OSError:
#             print("Bad file - Try again..." + fname)
#             continue
#
#         # convert the PIL image to a numpy array
#         # IN PIL - image is in (width, height, channel)
#         # In Numpy - image is in (height, width, channel)
#         numpy_image = img_to_array(original)
#
#         # Convert the image / images into batch format
#         # expand_dims will add an extra dimension to the data at a particular axis
#         # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
#         # Thus we add the extra dimension to the axis 0.
#         image_batch = np.expand_dims(numpy_image, axis=0)
#
#         # prepare the image (normalisation for channels)
#         processed_image = inception_resnet_v2.preprocess_input(image_batch.copy())
#
#         # get the predicted probabilities for each class
#         predictions = model_trained.predict(processed_image)
#         # print predictions
#         dominant_feature_idx = np.argmax(predictions[0])
#
#         # convert the probabilities to class labels
#         predicted_class = classes[dominant_feature_idx]
#         print('Predicted:', predicted_class)
#
#         df = pd.DataFrame(predictions[0]).transpose()
#         name_csv = default_path + "/Result/Tag_" + modelname + "/" + filename + "_" + predicted_class + ".csv"
#
#         # df.to_csv(name_csv)
#         header = classes
#         df.columns = classes
#         df.to_csv(name_csv, index=False, columns=header)
#
# # https://fairyonice.github.io/Visualization%20of%20Filters%20with%20Keras.html
# layer_output = layer_dict["predictions"].output
#
# out_index = [65, 18, 92, 721]
# for i in out_index:
#     visualizer = VisualizeImageMaximizeFmap(pic_shape=(224, 224, 3))
#     images = []
#     probs = []
#     myprob = 0
#     n_alg = 0
#     while (myprob < 0.9):
#         myimage = visualizer.find_images_for_layer(input_img, layer_output, [i],
#                                                    picorig=True, n_iter=20)
#         y_pred = model.predict(myimage[0][0]).flatten()
#         myprob = y_pred[i]
#         n_alg += 1
#
#     print("The total number of times the gradient ascent needs to run: {}".format(n_alg))
#
#     argimage = {"prediction": [myimage]}
#     print("{} probability:".format(classlabel[i])),
#     print("{:4.3}".format(myprob)),
#
#     visualizer.plot_images_wrapper(argimage, n_row=1, scale=4)
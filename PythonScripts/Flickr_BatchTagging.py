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
# dataname = "FlickrSeattle_Photos_All"
# model_json = "Model/InceptionResnetV2_retrain_Seattle_architecture_dropout0.3.json"
# photo_path_base = '/home/alan/Dropbox/KIT/FlickrEU/UnlabelledData/Seattle/NewTestPhotos'
# # photo_path_base = '/home/alan/Dropbox/KIT/FlickrEU/FlickrCNN/Seattle/FlickrSeattle_download/Photos/AOI_CellID_Merged/' # FlickrSeattle_Photos_Flickr_All/'
# #
# # trainedweights_name = "../FlickrCNN/TrainedWeights/InceptionResnetV2_Seattle_retrain_instabram_15classes_Okt2019_val_acc0.88.h5"
# # trainedweights_name = "../FlickrCNN/TrainedWeights/InceptionResnetV2_Seattle_retrain_instabram_15classes_Weighted_Nov2019_val_acc0.87.h5"
# #
# out_path_base = "/DATA2TB/FlickrSeattle_Tagging_Feb2020/"
# #
# classes = ["backpacking", "birdwatching", "boating", "camping", "fishing", "flooding", "hiking", "horseriding",
#            "mtn_biking", "noactivity", "otheractivities", "pplnoactivity", "rock climbing", "swimming",
#            "trailrunning"]
#
#
# trainedweights_name = "../FlickrCNN/TrainedWeights/InceptionResnetV2_Seattle_retrain_instabram_15classes_Okt2019_val_acc0.88.h5"


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
# Class #7 = hiking+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
dataname = "Korea"
model_json = 'Model/InceptionResnetV2_retrain_Korea_architecture_dropout0.3.json'
photo_path_base = '/DATA10TB/CameraTraps_Korea/Photos/'

trainedweights_name = "../TrainedWeights/InceptionResnetV2_retrain_Korea_27classes_Okt2020_val_acc0.78.h5"
#_Korea_21classes_Dec2019_val_acc0.65.h5"
#
out_path_base = "/DATA10TB/CameraTraps_Korea/Tagging2020_v1/"
#


classes = ["Amur hedgehog", "Asian badger", "Bat", "Car", "Cat", "Chipmunk", "Dog", "Eurasian badger", "Eurasian Otter", "Goral", "Korean hare",
           "Least weasel",
            "Marten", "Musk deer",  "No animal", "Red squirrel", "Rodentia",  "Flying squirrel",  "Siberian weasel", "Birds", "Human", "Leopard cat",
            "Racoon dog", "Roe dear", "Unidentified", "Water deer", "Wild boar"]
#
# ****************
# Class #0 = Amur hedgehog
# Class #1 = Asian badger
# Class #2 = Bat
# Class #3 = Car
# Class #4 = Cat
# Class #5 = Chipmunk
# Class #6 = Dog
# Class #7 = Eurasian badger
# Class #8 = Eurasian otter
# Class #9 = Goral
# Class #10 = Korean hare
# Class #11 = Least weasel
# Class #12 = Marten
# Class #13 = Musk deer
# Class #14 = No animal
# Class #15 = Red squirrel
# Class #16 = Rodentia
# Class #17 = Siberian flying squirrel
# Class #18 = Siberian weasel
# Class #19 = birds
# Class #20 = human
# Class #21 = leopard cat
# Class #22 = racoon dog
# Class #23 = roe dear
# Class #24 = unidentified
# Class #25 = water deer
# Class #26 = wild boar
os.chdir(default_path)

out_path = out_path_base + modelname + "/" + dataname + "/"

prediction_batch_size = 511  # to increase the speed of tagging .
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
            return (out_path + "Result/" + "/ClassifiedPhotos/" + "/" + foldername + "/" + x + "/2ndClass_" +x2 )


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
    df_aoi.to_csv(name_csv, index=False, columns=header)


    # @todo attention map

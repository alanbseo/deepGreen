import os
import json

import numpy as np
import pandas as pd

from keras.applications import inception_resnet_v2

from keras.preprocessing import image

img_width, img_height = 331, 331
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

import fnmatch

from shutil import copyfile

import PIL
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # read broken images

# default_path = '/home/alan/Dropbox/KIT/FlickrEU/deepGreen'
# default_path = '/Users/seo-b/Dropbox/KIT/FlickrEU/deepGreen'
default_path = '/pd/data/crafty/deepGreen'
modelname = "InceptionResnetV2_dropout30"

# copy jpg files
toCopyFile = False

# !export HIP_VISIBLE_DEVICES=0,1 #  For 2 GPU training
# os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
# os.environ['HIP_VISIBLE_DEVICES'] = '0'


# EU
dataname = "EU"

modelname = "InceptionResnetV2"

photo_path_base = "/pd/data/crafty/FlickrEU_DOWNLOAD_14May2018/May2018_V1_Photo/"
out_path_base = "/pd/data/crafty/FlickrEU_result/Tagging_EU/"

os.chdir(default_path)

out_path = out_path_base + modelname + "/" + dataname + "/"

# number of images for one batch prediction
prediction_batch_size = 1024

top = 10  # print top-n classes

img_width = img_height = 299
model_trained = inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None,
                                                      input_shape=(img_width, img_height, 3))

# Imagenet class labels
imagenet_labels_filename = "Data/imagenet_class_index.json"
with open(imagenet_labels_filename) as f:
    CLASS_INDEX = json.load(f)
#
classes = []
for i in range(CLASS_INDEX.__len__()):
    classes.append(CLASS_INDEX[str(i)][1])

classes_arr = np.array(classes)

num_classes = len(classes)

##### Predict

# Load the retrained CNN model

# Model reconstruction from JSON file
# with open(model_json, 'r') as f:
#    model_trained = model_from_json(f.read())

# model_final = multi_gpu_model(model_final, gpus=2, cpu_merge=True, cpu_relocation=False)


# def onlyfolders(path):
#     for file in os.listdir(path):
#         if os.path.isdir(os.path.join(path, file)):
#             yield file
#
#
# def onlyfiles(path):
#     for file in os.listdir(path):
#         if os.path.isdir(os.path.join(path, file)):
#             yield file


# list only folder names
foldernames = [d for d in os.listdir(photo_path_base) if os.path.isdir(os.path.join(photo_path_base, d))]

f_idx = 0

for f_idx in (range(0, len(foldernames))):

    foldername = foldernames[f_idx]
    print(f_idx)
    print(foldername)
    photo_path_aoi = os.path.join(photo_path_base, foldername)

    for (root, subdirs, files) in os.walk(photo_path_aoi):
        print('--\nroot = ' + root)
        # print(files)
        # print(subdirs)

        # csv output file
        name_csv = out_path + "Result/" + modelname + "/CSV/" + os.path.relpath(root, photo_path_base) + ".csv"
        if os.path.exists(name_csv):
            continue  # skip the folder if there is already the output csv file


        ### Read filenames
        filenames_raw = os.listdir(root)
        # print(filenames_raw)

        # filenames_raw = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(photo_path_aoi)) for f in fn]

        filenames1 = fnmatch.filter(filenames_raw, "*.jpg")
        filenames2 = fnmatch.filter(filenames_raw, "*.JPG")

        filenames = filenames1 + filenames2

        n_files = len(filenames)

        if n_files == 0:
            print('skip')
            continue  # skip the folder if there is no image


        if not (os.path.exists(os.path.dirname(name_csv))):
            os.makedirs(os.path.dirname(name_csv), exist_ok=True)

        # base filenames
        base_filenames = list(map(os.path.basename, filenames))

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
                # print(img_name)
                img_name = os.path.join(photo_path_aoi, root, img_name)

                # load an image in PIL format
                try:
                    img = image.load_img(img_name, target_size=(img_width, img_height))
                except:
                    img = PIL.Image.new(mode="RGB", size=(img_width, img_height))

                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)

                # prepare the image (normalisation for channels)
                img_preprocessed = inception_resnet_v2.preprocess_input(img.copy())
                images.append(img_preprocessed)

            # vstack for batch tagging
            images_vstack = np.vstack(images)

            # stack up images list to pass for prediction
            predictions = model_trained.predict(images_vstack, batch_size=bsize_tmp)

            # predictions.shape

            ## top selected classes
            top_classes_idx_arr = np.argsort(predictions)[:, ::-1][:, :top]

            top_classes_arr = classes_arr[top_classes_idx_arr]
            # print(top_classes_arr)

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

            predicted_class_v = top_classes_arr[:, 0]  # top1
            #predicted_class_top2_v = top_classes_arr[:, 1]  # top2


            # print('Predicted:', predicted_class_v)

            # 2nd-level
            # kind of equivalent to `sapply()' in R
            # def foo_get_predicted_filename(x):
            #     return (out_path + "Result/" + modelname + "/ClassifiedPhotos/" + os.path.relpath(root,
            #                                                                                       photo_path_base) + "/" + x)


            # predicted_filenames = list(map(foo_get_predicted_filename, predicted_class_v))

            arr_tmp = pd.DataFrame(np.concatenate((top_classes_arr, top_classes_probs_arr), axis=1))

            if step_start_idx == 0:
                arr_aoi = arr_tmp
            else:
                arr_aoi = np.concatenate((arr_aoi, arr_tmp), axis=0)


            # save_folder_names = list(map(os.path.basename, predicted_filenames))

            # create necessary folders
            # for i in range(0, n_files):
            #     if not (os.path.exists(save_folder_names[i])):
            #         os.makedirs(save_folder_names[i], exist_ok=False)
            # if (toCopyFile):
            #     for i in range(0, bsize_tmp):
            #
            #         save_folder = predicted_filenames[i]
            #         print(save_folder)
            #
            #         if not (os.path.exists(save_folder)):
            #             os.makedirs(save_folder, exist_ok=False)
            #             copyfile(filenames_batch[i], predicted_filenames[i] + '/' + os.path.basename(filenames_batch[i]))

        # Write csv files

        # Write a Pandas data frame
        df_aoi = pd.concat([pd.DataFrame(base_filenames), pd.DataFrame(arr_aoi)], axis=1)
        header = np.concatenate(
            (["Filename"], ["Top1", "Top2", "Top3", "Top4", "Top5", "Top6", "Top7", "Top8", "Top9", "Top10"],
             ["Prob1", "Prob2", "Prob3", "Prob4", "Prob5", "Prob6", "Prob7", "Prob8", "Prob9", "Prob10"]))

        df_aoi.columns = header
        df_aoi.to_csv(name_csv, index=False, columns=header)

        # @todo attention map

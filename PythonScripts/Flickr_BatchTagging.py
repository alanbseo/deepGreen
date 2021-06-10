import os

import numpy as np
import pandas as pd

from keras.applications import inception_resnet_v2

from keras.preprocessing import image

img_width, img_height = 331, 331
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

import fnmatch

from shutil import copyfile


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # read broken images
ImageFile.ERRORS



default_path = '/home/alan/Dropbox/KIT/FlickrEU/deepGreen'
#default_path = '/Users/seo-b/Dropbox/KIT/FlickrEU/deepGreen'

modelname = "InceptionResnetV2_dropout30"

# copy jpg files
toCopyFile = False

#!export HIP_VISIBLE_DEVICES=0,1 #  For 2 GPU training
# os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
# os.environ['HIP_VISIBLE_DEVICES'] = '0'


# Korea
dataname = "Korea"
model_json = 'Model/InceptionResnetV2_retrain_Korea_architecture_dropout0.3.json'
photo_path_base = '/home/alan/Dropbox/Korea Marten/ESP EU 2021/Camera trap/'
out_path_base = "/home/alan/Dropbox/Korea Marten/ESP EU 2021/Camera trap result v1/"

trainedweights_name = "../TrainedWeights/InceptionResnetV2_retrain_Korea_27classes_Okt2020_val_acc0.78.h5"
 #
# out_path_base = "/DATA10TB/CameraTraps_Korea/Tagging2021_Eco_v1/"
classes = ["Amur hedgehog", "Asian badger", "Bat", "Car", "Cat", "Chipmunk", "Dog", "Eurasian badger", "Eurasian Otter", "Goral", "Korean hare",
          "Least weasel",
           "Marten", "Musk deer",  "No animal", "Red squirrel", "Rodentia",  "Flying squirrel",  "Siberian weasel", "Birds", "Human", "Leopard cat",

           "Racoon dog", "Roe dear", "Unidentified", "Water deer", "Wild boar"]

# model_trained = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',input_tensor=None, input_shape=(img_width, img_height, 3))
# x = model_trained.output
# x = GlobalAveragePooling2D()(x) # before dense layer
# x = Dense(1024, activation='relu')(x)
# predictions_new = Dense(num_classes, activation='softmax', name='softmax')(x)
# model_trained = Model(inputs=model_trained.input, outputs=predictions_new)
#
# # Load weights into the new model
# model_trained.load_weights(trainedweights_name)


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

# EU
# modelname = "InceptionResnetV2"
#
# out_path_base = "/Users/seo-b/Downloads/Tagging_Bayern"


os.chdir(default_path)

out_path = out_path_base + modelname + "/" + dataname + "/"

prediction_batch_size = 512  # to increase the speed of tagging .
# number of images for one batch prediction

top = 10  # 10  print top-n classes

# img_width = img_height = 299
# model_trained = inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet',input_tensor=None, input_shape=(img_width, img_height, 3))


classes_arr = np.array(classes)


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
foldernames=[d for d in os.listdir(photo_path_base) if os.path.isdir(os.path.join(photo_path_base, d))]

f_idx = 6


# walk_dir = photo_path_base
# print('walk_dir = ' + walk_dir)

# If your current working directory may change during script execution, it's recommended to
# immediately convert program arguments to an absolute path. Then the variable root below will
# be an absolute path as well. Example:
# walk_dir = os.path.abspath(walk_dir)
# print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))
#
# for root, subdirs, files in os.walk(photo_path_base):
#     print('--\nroot = ' + root)
#     # list_file_path = os.path.join(root, 'my-directory-list.txt')
#     # print('list_file_path = ' + list_file_path)
#     #
#     # with open(list_file_path, 'wb') as list_file:
#     for subdir in subdirs:
#         print('\t- subdirectory ' + subdir)
#
#     for filename in files:
#         file_path = os.path.join(root, filename)
#
#         print('\t- file %s (full path: %s)' % (filename, file_path))
#
#             # with open(file_path, 'rb') as f:
#             #     f_content = f.read()
#             #     list_file.write(('The file %s contains:\n' % filename).encode('utf-8'))
#             #     list_file.write(f_content)
#             #     list_file.write(b'\n')


f_idx = 5


for f_idx in range(0, len(foldernames)):

    foldername = foldernames[f_idx]
    print(f_idx)
    print(foldername)
    photo_path_aoi = os.path.join(photo_path_base, foldername)

    for (root, subdirs, files) in os.walk(photo_path_aoi):
        print('--\nroot = ' + root)
        # print(files)
        # print(subdirs)

        ### Read filenames
        filenames_raw = os.listdir(root)
        # print(filenames_raw)

        # filenames_raw = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(photo_path_aoi)) for f in fn]

        filenames1 = fnmatch.filter(filenames_raw, "*.jpg")
        filenames2 = fnmatch.filter(filenames_raw, "*.JPG")

        filenames = filenames1 + filenames2

        n_files = len(filenames)

        if n_files == 0:
            continue  # skip the folder if there is no image

        # csv output file
        name_csv = out_path + "Result/" + "CSV/" + os.path.relpath(root, photo_path_base)  + ".csv"
        if os.path.exists(name_csv):
            continue    # skip the folder if there is already the output csv file

        if not (os.path.exists(os.path.dirname(name_csv))):
            os.makedirs(os.path.dirname(name_csv), exist_ok=False)




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
                img = image.load_img(img_name, target_size=(img_width, img_height))
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
            #print(top_classes_arr)

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
                return (out_path + "Result/" + "ClassifiedPhotos/" + os.path.relpath(root, photo_path_base) + "/" + x)
                # return (out_path + "Result/" + "/ClassifiedPhotos/" + "/" + foldername + "/" + x + "/2ndClass_" +x2 )


            predicted_filenames = list(map(foo_get_predicted_filename, predicted_class_v, predicted_class_top2_v))
            save_folder_names = list(map(os.path.basename, predicted_filenames))


            arr_tmp = pd.DataFrame(np.concatenate((top_classes_arr, top_classes_probs_arr), axis=1))

            if step_start_idx == 0:
                arr_aoi = arr_tmp
            else:
                arr_aoi = np.concatenate((arr_aoi, arr_tmp), axis=0)

            # create necessary folders
            # for i in range(0, n_files):
            #     if not (os.path.exists(save_folder_names[i])):
            #         os.makedirs(save_folder_names[i], exist_ok=False)
            if (toCopyFile):
                for i in range(0, bsize_tmp):

                    save_folder = predicted_filenames[i]
                    print(save_folder)

                    if not (os.path.exists(save_folder)):
                        os.makedirs(save_folder, exist_ok=False)
                        copyfile(filenames_batch[i], predicted_filenames[i] + '/' + os.path.basename(filenames_batch[i]))



        # Write csv files


        # Write a Pandas data frame
        df_aoi = pd.concat([pd.DataFrame(base_filenames), pd.DataFrame(arr_aoi)], axis=1)
        header = np.concatenate(
            (["Filename"], ["Top1", "Top2", "Top3", "Top4", "Top5", "Top6", "Top7", "Top8", "Top9", "Top10"],
             ["Prob1", "Prob2", "Prob3", "Prob4", "Prob5", "Prob6", "Prob7", "Prob8", "Prob9", "Prob10"]))

        df_aoi.columns = header
        df_aoi.to_csv(name_csv, index=False, columns=header)


        # @todo attention map

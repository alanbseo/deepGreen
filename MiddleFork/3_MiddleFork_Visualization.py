import os

import numpy as np
import pandas as pd

from keras.applications import inception_resnet_v2

from keras.preprocessing import image

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

import fnmatch

from shutil import copyfile

img_width, img_height = 331, 331

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from tensorflow.python.keras import backend as k

# only mac
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


home_path = '/Users/seo-b/'
default_path = home_path + 'Dropbox/KIT/FlickrEU/deepGreen'

#!export HIP_VISIBLE_DEVICES=0,1 #  For 2 GPU training
# os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
os.environ['HIP_VISIBLE_DEVICES'] = '0'

# Seattle Middle Fork
model_json = "Model/InceptionResnetV2_retrain_Seattle_architecture_dropout0.3.json"

dataname = "MiddleFork"
photo_path_base = home_path + 'Dropbox/KIT/FlickrEU/Seattle/Seattle_TaggedData_BigSize/All images/MiddleFork_AllPhotos/' # Middle Fork

#dataname = "MountainLoop"
#photo_path_base = home_path + 'Dropbox/KIT/FlickrEU/Seattle/Seattle_TaggedData_BigSize/All images/MountainLoop_AllPhotos/' # Mountain Loop

# out_path_base = "/DATA10TB/FlickrSeattle_Tagging_Feb2021/"
# out_path_base = "/SSDSATA1TB/FlickrTagging/"
out_path_base = home_path + "Downloads/FlickrTagging/"

classes = ["backpacking", "birdwatching", "boating", "camping", "fishing", "flooding", "hiking", "horseriding",
           "mtn_biking", "noactivity", "otheractivities", "pplnoactivity", "rock climbing", "swimming",
           "trailrunning"]


modelname = "InceptionResnetV2_dropout30_noweighting"
trainedweights_name = "../Seattle/Seattle_TaggedData_BigSize/ModelAndTrained weights/InceptionResnetV2_Seattle_retrain_instagram_15classes_Okt2019_val_acc0.88.h5"

#modelname = "InceptionResnetV2_dropout30_weighting"
#trainedweights_name = "../Seattle/Seattle_TaggedData_BigSize/ModelAndTrained weights/InceptionResnetV2_Seattle_retrain_instagram_15classes_Weighted_Dec2020_val_acc0.87_redone_for_traininghistory.h5"

os.chdir(default_path)

out_path = out_path_base + modelname + "/" + dataname + "/"
# out_path = out_path_base +  "/" + dataname + "/"

prediction_batch_size = 8  # to increase the speed of tagging .
# num+ber of images for one batch prediction

top = 10  # print top-n classes


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



# Load the retrained CNN model

model_trained = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',input_tensor=None, input_shape=(img_width, img_height, 3))
x = model_trained.output
x = GlobalAveragePooling2D()(x) # before dense layer
x = Dense(1024, activation='relu')(x)
predictions_new = Dense(num_classes, activation='softmax', name='softmax')(x)
model_trained = Model(inputs=model_trained.input, outputs=predictions_new)

# Load weights into the new model
model_trained.load_weights(trainedweights_name)


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
            return (out_path + "/" + "" + foldername + "/" + x)
            #return (out_path + "Result/" + "ClassifiedPhotos/" + "/" + x + "/2ndClass_" +x2 )


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

    name_csv = out_path + "/" + "/CSV/" + foldername + ".csv"
    if not (os.path.exists(os.path.dirname(name_csv))):
        os.makedirs(os.path.dirname(name_csv), exist_ok=False)

    # Write a Pandas data frame
    df_aoi = pd.concat([pd.DataFrame(base_filenames), pd.DataFrame(arr_aoi)], axis=1)
    header = np.concatenate(
        (["Filename"], ["Top1", "Top2", "Top3", "Top4", "Top5", "Top6", "Top7", "Top8", "Top9", "Top10"],
         ["Prob1", "Prob2", "Prob3", "Prob4", "Prob5", "Prob6", "Prob7", "Prob8", "Prob9", "Prob10"]))

    df_aoi.columns = header
    df_aoi.to_csv(name_csv, index=False, columns=header)



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

        # print predictions
        dominant_feature_idx = np.argmax(predictions[0])

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
        # plt.matshow(heatmap)
        # plt.show()

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
        font = cv2.FONT_HERSHEY_COMPLEX

        # Draw the text

        for x in range(0, num_classes):
            tags = classes[x] + ":" + str((predictions[0][x] * 100).round(2)) + "%"
            cv2.putText(superimposed_img, tags, (10, 25 * x + int(img.shape[0] / 10)), font, 0.5, (255, 255, 255),
                        2)  # ,cv2.LINE_AA)

        out_filepath = out_path + "Result/Heatmap_" + modelname + "/" + predicted_class
        if not (os.path.exists(out_filepath)):
            os.makedirs(out_filepath,
                        exist_ok=False)

        # Save the image to disk
        cv2.imwrite(out_filepath + "/AttentionMap_" + predicted_class + "_" + filename, superimposed_img)

        copyfile(fname, out_filepath + '/' + filename)

    else:
        print(fname + " error?")




#
#
# # # https://fairyonice.github.io/Visualization%20of%20Filters%20with%20Keras.html
# # layer_output = layer_dict["predictions"].output
# #
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

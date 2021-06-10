# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)
import datetime

import numpy as np
import torch
import torchvision.models as models
from torchvision import datasets
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import pandas as pd

import PIL
from PIL import ImageFile
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True  # read broken images

import loadcaffe

# EU
dataname = "EU28"
# the architecture to use
arch = 'resnet18'
# arch = 'resnet50'
model_name = 'places1365_' + arch


# number of workers
num_workers = 8
# number of images for one batch prediction
prediction_batch_size = 2048
# number of tags to print
top_n = 10



# KEAL
default_path = '/pd/data/crafty/deepGreen'
photo_path_base = "/pd/data/crafty/FlickrEU_DOWNLOAD_14May2018/May2018_V1_Photo/"
out_path_base = "/pd/data/crafty/FlickrEU_result/Places_EU/"

# # X470
# default_path = '/home/alan/Dropbox/KIT/FlickrEU/deepGreen'
# # photo_path_base = "/home/alan/Downloads/Bayern/Flickr_Aug2018_V2_Photo_Bayern/"
# photo_path_base = "/home/alan/nfs_keal_pd/FlickrEU_DOWNLOAD_14May2018/May2018_V1_Photo/"
# # out_path_base = "/home/alan/Dropbox/KIT/FlickrEU/LabelledData/Places_EU/"
# out_path_base = "/home/alan/nfs_keal_pd/FlickrEU_result/Places_EU/"
# # prediction_batch_size = 1024

# X570
default_path = '/home/alan/Dropbox/KIT/FlickrEU/deepGreen'
photo_path_base = "/home/alan/Dropbox/KIT/FlickrEU/FlickrEU_download/SamplePhotos/"
# photo_path_base = "/DATA10TB/FlickrEU_download/Bayern/Flickr_Aug2018_V2_Photo_Bayern/"
out_path_base = "/home/alan/Dropbox/KIT/FlickrEU/LabelledData/Test/"




os.chdir(default_path)

out_path = out_path_base + model_name + "/" + dataname + "/"




model = loadcaffe.load('deploy_alexnet_places365.prototxt', 'alexnet_places365.caffemodel', 'cudnn')

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()















# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# # load the class label
classfile_name = 'Data/categories_places365.txt'
# if not os.access(file_name, os.W_OK):
#     synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
#     os.system('wget ' + synset_url)
classes_l = list()
with open(classfile_name) as class_file:
    for line in class_file:
        classes_l.append(line.strip().split(' ')[0][3:])
classes = tuple(classes_l)
classes_arr = np.array(classes_l)


# kind of equivalent to `sapply()' in R
def foo_get_filenames(x):
    return (os.path.basename(x[0]))

def foo_get_year(x):
    return (os.path.basename(os.path.dirname(x[0])))


# Ignore broken images
# https://discuss.pytorch.org/t/questions-about-dataloader-and-dataset/806/2
# https://github.com/pytorch/pytorch/issues/1137
class ImageFolderEX(datasets.ImageFolder):

    __init__ = ImageFolder.__init__

    def __getitem__(self, index):
        path, label = self.imgs[index]
        try:
            return super(ImageFolderEX, self).__getitem__(index)
        except Exception as e:
            print(e)
            return None   #returns none (or handling error?)
        return [img, label]


# Filter the None values in the collate_fn()

def mycollate_fn(batch):
    batch = list(filter(lambda x : x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)



# list only folder names
foldernames = [d for d in os.listdir(photo_path_base) if os.path.isdir(os.path.join(photo_path_base, d))]

# f_idx = 0

for f_idx in reversed(range(0, 25000)):

    foldername = foldernames[f_idx]
    print(f_idx)
    print(foldername)
    photo_path_aoi = os.path.join(photo_path_base, foldername)

    for (root, subdirs, files) in os.walk(photo_path_aoi):

        if len(subdirs) == 0:
            continue # skip if it does not have a subdir


        # csv output file
        name_csv = out_path + "Result/" + "/CSV/" + os.path.relpath(root, photo_path_base) + ".csv"
        if os.path.exists(name_csv):
            print("skips as it is done already")
            continue  # skip the folder if there is already the output csv file


        print('--\nroot = ' + root)
        print(subdirs)


        # pytorch dataset and dataloader
        # dataset = {'predict' : datasets.ImageFolder(photo_path_aoi, centre_crop)}
        dataset = {'predict' : ImageFolderEX(photo_path_aoi, centre_crop)}


        # get filenames from the dataset
        base_filenames = list(map(foo_get_filenames, dataset['predict'].imgs))
        years = list(map(foo_get_year, dataset['predict'].imgs))

        n_files = len(base_filenames)
        print("n_files=" + str(n_files))


        if n_files == 0:
            print("skips as there is no image")
            continue  # skip the folder if there is no image


        # # check files
        # fcnt = 0
        # try:
        #     for idx, (data, image) in enumerate(dataset['predict']): fcnt += 1
        #     print("all image files are valid")
        # except: # PIL.UnidentifiedImageError
        #     print("skips because of faulty image files")
        #     # continue # skip the folder temporarily



        # parallelised
        batch_cnt = 1

        prediction_batch_size_tmp = min(prediction_batch_size, n_files)
        # prediction_batch_size_tmp = 1
        # dataloader = {'predict': torch.utils.data.DataLoader(dataset['predict'], batch_size = prediction_batch_size_tmp, shuffle=False, num_workers=num_workers)} # fails when broken images

        # Pass the collate_fn() to the DataLoader()
        dataloader = {'predict': torch.utils.data.DataLoader(dataset['predict'], batch_size = prediction_batch_size_tmp, shuffle=False, num_workers=num_workers, collate_fn = mycollate_fn)}

        # print("n_files=" + str(len(dataloader['predict'].dataset.imgs)))


        print("batch size = " + str(prediction_batch_size_tmp))

        try:
            for inputs, labels in dataloader['predict']:

                print("batch:" + str(batch_cnt))
                # print(inputs)
                logit = model.forward(inputs)
                h_x = F.softmax(logit, 1).data.squeeze()
                # probs, idx = h_x.sort(0, True) # does not work for batch tagging
                predictions = h_x.numpy()

                print("predictions " + str(predictions.shape))

                # https://stackoverflow.com/questions/12575421/convert-a-1d-array-to-a-2d-array-in-numpy
                if len(predictions.shape) ==1:
                    print("convert-a-1d-array-to-a-2d-array")
                    predictions = np.reshape(predictions, (-1, predictions.size))

                ## top selected classes
                top_classes_idx_arr = np.argsort(predictions)[:, ::-1][:, :top_n]

                top_classes_arr = classes_arr[top_classes_idx_arr]
                # print(top_classes_arr)

                bsize_tmp = labels.size().numel()

                # create an empty array
                top_classes_probs_arr = np.empty([bsize_tmp, top_n])
                top_classes_probs_arr[:] = 0

                for i in range(0, bsize_tmp):
                    top_classes_probs_arr[i,] = predictions[i, [top_classes_idx_arr[i,]]]

                top_classes_arr[0, :]
                top_classes_probs_arr[0, :]

                predicted_class_v = top_classes_arr[:, 0]  # top1


                arr_tmp = pd.DataFrame(np.concatenate((top_classes_arr, top_classes_probs_arr), axis=1))

                if batch_cnt == 1:
                    arr_aoi = arr_tmp.to_numpy()
                else:
                    arr_aoi = np.concatenate((arr_aoi, arr_tmp), axis=0)
                # increase count
                batch_cnt+=1
        except Exception as e:
            print(e)
            print("skips this folder")
            continue # your handling code

        if not (os.path.exists(os.path.dirname(name_csv))):
            os.makedirs(os.path.dirname(name_csv), exist_ok=True)


        # Write a Pandas data frame to a csv file
        df_aoi = pd.concat([pd.DataFrame(base_filenames), pd.DataFrame(years), pd.DataFrame(arr_aoi)], axis=1)
        header = np.concatenate(
            (["Filename"], ["Year"],["Top1", "Top2", "Top3", "Top4", "Top5", "Top6", "Top7", "Top8", "Top9", "Top10"],
             ["Prob1", "Prob2", "Prob3", "Prob4", "Prob5", "Prob6", "Prob7", "Prob8", "Prob9", "Prob10"]))

        df_aoi.columns = header
        df_aoi.to_csv(name_csv, index=False, columns=header)

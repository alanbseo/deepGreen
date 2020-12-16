##### Sample tagged photos for evaluation 
 
# @TODO use random number generators 
# @TODO fix the folder names

# folder <- "~/Dropbox/KIT/FlickrEU/DATA2TB_LINK/Flickr15Oct2019/backpacking/"
# folder_base <- "/DATA10TB/FlickrSeattle_Result_Finetuned/Heatmap_InceptionResnetV2_finetuning/"
folder_base <- "~/Dropbox/KIT/FlickrEU/Seattle/Seattle_TaggedData_BigSize/TaggedResult_Nov2019_Middlefork/ClassifiedPhotos/" 
folder_vali = "~/Dropbox/KIT/FlickrEU/FlickrCNN/Seattle/Manual evaluation/New18kPhotos/Sample_eval/"# new 18000 photos outside middlefork
library(stringr)

dirs = list.dirs(folder_base, recursive = F, full.names = F)
min_samples_size = 5 
max_samples_size = 100

sampling_rate = 0.2 # 20% 
summed = 0 


for (dir_idx in 1:length(dirs)) { 
    folder_class =  dirs[dir_idx]
    
    files_short <- list.files(paste0(folder_base,"/", folder_class), pattern = "*.jpg", full.names= F, recursive = T)
    files_full = list.files(paste0(folder_base,"/", folder_class), include.dirs = F, pattern = "*.jpg", full.names= T, recursive = T)
    
    sample_size = max(floor(length(files_short)* sampling_rate), min_samples_size)
    sample_size = min(sample_size, floor(length(files_short)), max_samples_size) 
    
    set.seed(dir_idx + 1978 ) # for reproducibility
    sampled_idx = sample(1:length(files_short), size = sample_size)
    
    
    second_classes =  unlist(str_extract_all(files_short[sampled_idx], pattern = "(?<=2ndClass\\_)[a-zA-Z\\_]*"))
    file_paths = files_short[sampled_idx]
    
    file_names= str_extract(file_paths, pattern = "(?<=\\/)[a-z0-9.]*")
    
    vali_df = data.frame(Filename = file_names, Top1 = folder_class, Top2 = second_classes, Top1YN = NA, Top2YN = NA)
    
    vali_dir = paste0(folder_vali, "/", folder_class)
    if (!dir.exists(vali_dir)) { 
        dir.create(vali_dir, recursive = T) 
    }
    # write.xlsx(vali_df, file = paste0(folder_vali, folder_class, "_samples.xlsx"), row.names = F)
    # 
    # file.copy(from = file.path(files_full[sampled_idx]), to = file.path(paste0(vali_dir, "/", file_names)), overwrite = T)
    summed = summed + nrow(vali_df)
}

print(summed)







# Manually evaluated data (used the sampled photos)
dt_evaluated = read.xlsx("Manual evaluation_MiddleFork/Manual evalutation_MiddleFork_10July2020_n724_withTop2_corrected.xlsx", 1)
colnames(dt_evaluated)


## training photo ids 

all_img_names = basename(list.files("../../UnlabelledData/Seattle/FlickrSeattle_AllPhotos/", recursive = T, full.names = F, include.dirs = F))

traing_img_names = basename(list.files("Training images/Photos_iterative_Sep2019/train/", recursive = T, full.names = F, include.dirs = F))








set.seed(1978)
files <- list.files(paste0(folder_base, "backpacking/"), pattern = "*.jpg", full.names= F, recursive = T)
sample(1:length(files), length(files)*0.1)
files[sample(1:length(files), length(files)*0.1)]

folder.birdwatching <- paste0(folder_base, "birdwatching/")
files.birdwatching <- list.files(folder.birdwatching, pattern = "*.jpg", full.names= F, recursive = T)
sample(1:length(files.birdwatching), length(files.birdwatching)*0.1)
files.birdwatching[sample(1:length(files.birdwatching), length(files.birdwatching)*0.1)]

folder.boating <- paste0(folder_base, "boating/")
files.boating <- list.files(folder.boating, pattern = "*.jpg", full.names= F, recursive = T) # 18
sample(1:length(files.boating), 5)
files.boating[sample(1:length(files.boating), 5)]

folder.camping <- paste0(folder_base, "camping/")
files.camping <- list.files(folder.camping, pattern = "*.jpg", full.names= F, recursive = T)
sample(1:length(files.camping), length(files.camping)*0.1)
files.camping[sample(1:length(files.camping), length(files.camping)*0.1)]


folder.fishing <- paste0(folder_base, "fishing/")
files.fishing <- list.files(folder.fishing, pattern = "*.jpg", full.names= F, recursive = T)
sample(1:length(files.fishing), max(length(files.fishing)*0.1, 1))
files.fishing[sample(1:length(files.fishing), max(length(files.fishing)*0.1, 1))]

folder.flooding <- paste0(folder_base, "flooding/")
files.flooding <- list.files(folder.flooding, pattern = "*.jpg", full.names= F, recursive = T)
sample(1:length(files.flooding), 5)
files.flooding[sample(1:length(files.flooding), 5)]




folder.hiking <- paste0(folder_base, "hiking/")
files.hiking <- list.files(folder.hiking, pattern = "*.jpg", full.names= F, recursive = T)
set.seed(1986)
random.sample <- sample(1:length(files.hiking), 300)
# print(random.sample)
random.hiking <- files.hiking[random.sample]
random.hiking_df = data.frame(filename = random.hiking)
write.xlsx(random.hiking_df, file = "hiking.samples.xlsx")
filenames_tmp = read.xlsx( "hiking.samples.xlsx", 1)
random.hiking_readfromxls = filenames_tmp$filename
dataFiles <- dir(folder.hiking, files.hiking, ignore.case = TRUE, all.files = F)
file.copy(file.path(folder.hiking, random.hiking_readfromxls), to = file.path("~/Dropbox/KIT/FlickrEU/FlickrCNN/Seattle/Manual evaluation/Sample_eval/hiking/", random.hiking_readfromxls), overwrite = T)


folder.horseriding <- paste0(folder_base, "horseriding/")
files.horseriding <- list.files(folder.horseriding, pattern = "*.jpg", full.names= F)
random.sample <- sample(1:length(files.horseriding), 5)
files.horseriding[random.sample]
#write.xlsx(files.horseriding[random.sample], file = "horseriding.samples.xlsx")


folder.mtn_biking <- paste0(folder_base, "mtn_biking/")
files.mtn_biking <- list.files(folder.mtn_biking, pattern = "*.jpg", full.names= F)
set.seed(1986)
random.sample <- sample(1:length(files.mtn_biking), floor(length(files.mtn_biking)*0.1))
# print(random.sample)
random.biking <- files.mtn_biking[random.sample]
random.biking_df = data.frame(filename = random.biking)
write.xlsx(random.biking_df, file = "mtn_biking.samples.xlsx")
filenames_tmp = read.xlsx( "mtn_biking.samples.xlsx", 1)
random.biking_readfromxls = filenames_tmp$filename
file.copy(file.path(folder.mtn_biking, random.biking_readfromxls), to = file.path("~/Dropbox/KIT/FlickrEU/FlickrCNN/Seattle/Manual evaluation/Sample_eval/mtn_biking/",random.biking_readfromxls), overwrite = T)


folder.noactivity <- paste0(folder_base, "noactivity/")
files.noactivity <- list.files(folder.noactivity, pattern = "*.jpg", full.names= F)
set.seed(1986)
random.sample <- sample(1:length(files.noactivity), 300)
# print(random.sample)
random.noactivity <- files.noactivity[random.sample]
random.noactivity_df = data.frame(filename = random.noactivity)
write.xlsx(random.noactivity_df, file = "noactivity.samples.xlsx")
filenames_tmp = read.xlsx( "noactivity.samples.xlsx", 1)
random.noactivity_readfromxls = filenames_tmp$filename
file.copy(file.path(folder.noactivity, random.noactivity_readfromxls), to = file.path("~/Dropbox/KIT/FlickrEU/FlickrCNN/Seattle/Manual evaluation/Sample_eval/noactivity/", random.noactivity_readfromxls), overwrite = T)


folder.otheractivities <- paste0(folder_base, "otheractivities/")
files.otheractivities <- list.files(folder.otheractivities, pattern = "*.jpg", full.names= F)
set.seed(1986)
random.sample <- sample(1:length(files.otheractivities), floor(length(files.otheractivities)*0.1))
# print(random.sample)
random.otheractivities <- files.otheractivities[random.sample]
random.otheractivities_df = data.frame(filename = random.otheractivities)
write.xlsx(random.otheractivities_df, file = "otheractivities.samples.xlsx")
filenames_tmp = read.xlsx( "otheractivities.samples.xlsx", 1)
random.otheractivities_readfromxls = filenames_tmp$filename
file.copy(file.path(folder.otheractivities, random.otheractivities_readfromxls), to = file.path("~/Dropbox/KIT/FlickrEU/FlickrCNN/Seattle/Manual evaluation/Sample_eval/otheractivities/", random.otheractivities_readfromxls), overwrite = T)


folder.pplnoactivity <- paste0(folder_base, "pplnoactivity/")
files.pplnoactivity <- list.files(folder.pplnoactivity, pattern = "*.jpg", full.names= F)
random.sample <- sample(1:length(files.pplnoactivity), 5)
files.pplnoactivity[random.sample]
write.xlsx(files.pplnoactivity[random.sample], file = "pplnoactivity.samples.xlsx")


folder.rockclimbing <- paste0(folder_base, "rock climbing/")
files.rockclimbing <- list.files(folder.rockclimbing, pattern = "*.jpg", full.names= F)
set.seed(1986)
random.sample <- sample(1:length(files.rockclimbing), floor(length(files.rockclimbing)*0.1))
# print(random.sample)
random.rockclimbing <- files.rockclimbing[random.sample]
random.rockclimbing_df = data.frame(filename = random.rockclimbing)
write.xlsx(random.rockclimbing_df, file = "rockclimbing.samples.xlsx")
filenames_tmp = read.xlsx( "rockclimbing.samples.xlsx", 1)
random.rockclimbing_readfromxls = filenames_tmp$filename
file.copy(file.path(folder.rockclimbing, random.rockclimbing_readfromxls), to = file.path("~/Dropbox/KIT/FlickrEU/FlickrCNN/Seattle/Manual evaluation/Sample_eval/rockclimbing/", random.rockclimbing_readfromxls), overwrite = T)

folder.swimming <- paste0(folder_base, "swimming/")
files.swimming <- list.files(folder.swimming, pattern = "*.jpg", full.names= F)
set.seed(1986)
random.sample <- sample(1:length(files.swimming), floor(length(files.swimming)*0.1))
# print(random.sample)
random.swimming <- files.swimming[random.sample]
random.swimming_df = data.frame(filename = random.swimming)
write.xlsx(random.swimming_df, file = "swimming.samples.xlsx")
filenames_tmp = read.xlsx( "swimming.samples.xlsx", 1)
random.swimming_readfromxls = filenames_tmp$filename
file.copy(file.path(folder.swimming, random.swimming_readfromxls), to = file.path("~/Dropbox/KIT/FlickrEU/FlickrCNN/Seattle/Manual evaluation/Sample_eval/swimming/", random.swimming_readfromxls), overwrite = T)



######################################################################################
eval.data <- read.xlsx("Manual evalutation_MiddleFork.xlsx")
aggregate(eval.data$result, by=list(eval.data$evaluation), FUN=summary)



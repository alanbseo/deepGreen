library(openxlsx)
library(stringr) 
library(corrplot)
library(latticeExtra)


# Location of the dropbox shared folder
dropbox_location = "~/Dropbox/KIT/FlickrEU/Seattle/Seattle_TaggedData_BigSize/"

path_seattle = paste0(dropbox_location)
setwd(path_seattle)



#### Training ACC

trn_df = read.csv("ModelAndTrained weights/Flickr_MF_training_performance.csv")
summary(trn_df)

pdf("Output/FlickrMiddleFork_training.pdf", width = 8, height = 6)
par(mfrow=c(1,2))
plot(trn_df$val_accuracy, type="l", ylim = c(0,1), col = "blue", xlab="Epoch", ylab="Accuracy")
lines(trn_df$categorical_accuracy, type="l", col = "green")
# abline(h=0.71, col="red", lty=2)
# abline(h=0.60, col="purple", lty=2)
# legend("bottomright", legend=c("Training acc.", "Validation acc.", "Validation (MF)", "Validation (MT)"), col = c(3,4, "red", "purple"), lty=c(1,1,2,2), bty="n")
legend("bottomright", legend=c("Training acc.", "Validation acc."), col = c(3,4), lty=c(1,1), bty="n")

plot(trn_df$val_loss, type="l", ylim = c(0,5), col = "blue", xlab="Epoch", ylab="Loss")
lines(trn_df$loss, type="l", col = "green")
legend("bottomright", legend=c("Training loss", "Validation loss"), col = c(3,4), lty=1, bty="n")
dev.off()









# ## manual eval 
# dt = readxl::read_excel("Manual evaluation_MiddleFork/Manual evalutation_MiddleFork_18Feb2020_n724.xlsx", 1)
# dt$YN = factor(dt$evaluation, levels = c("y", "n", "x"))
# dt$YN [dt$YN == "x"] = "y"
# dt$YN 
# 
# dt_tab_l = tapply(dt$YN , dt$result, FUN = table)
# 
# dt_tab = t(do.call("rbind", dt_tab_l))
# 
# 
# 
# dt_tab_rel = apply(dt_tab, MARGIN = 2, FUN = function(x) x / sum(x))
# over_acc =round( sum(dt_tab[1,]) / sum(dt_tab), 3)
# 
# pdf("Output/FlickrMiddleFork_manualevaluation.pdf", width = 12, height = 6)
# par(mfrow=c(1,2), mar=c(6,12,4,4))
# barplot(dt_tab, beside=T, horiz=T, main = "# of photos correctly/incorrectly classifed", las=2, names = paste0(colnames(dt_tab), " (n=", colSums(dt_tab), ")"),  sub = paste0("Overall acc.=", over_acc), col = c("red", "green", "black"))
# acc = formatC( dt_tab_rel[1,],digits = 2)
# legend("topright", legend = c("Incorrect", "Correct"), col = c( "green", "red"), pch=15, bty="n")
# barplot(dt_tab_rel, beside=T,horiz = T, main = "% of photos correctly/incorrectly classifed", las=2, names = paste0(colnames(dt_tab_rel), " (Acc.=", acc, ")" ), col = c("red", "green", "black"))
# legend("topright", legend = c("Incorrect", "Correct"), col = c( "green", "red"), pch=15, bty="n")
# 
# 
# dev.off()



# Activity classes
classes = c("backpacking", "birdwatching", "boating", "camping", "fishing", "flooding", "hiking", "horseriding", "mtn_biking", "noactivity", "otheractivities", "pplnoactivity", "rock climbing", "swimming", "trailrunning")

classes_fullnames = c("Backpacking", "Bird watching", "Boating", "Camping", "Fishing", "Flooding", "Hiking", "Horse riding", "Mountain biking", "No activity", "Other activities", "People no activity", "Rock climbing", "Swimming", "Trail running")

n_classes = length(classes)



### imagenet photos


## Flickr photos
all_img_names = basename(list.files("All images/FlickrSeattle_AllPhotos/", recursive = T, full.names = F, include.dirs = F))

traing_img_names = basename(list.files("Training images/Photos_iterative_Sep2019/train/", recursive = T, full.names = F, include.dirs = F))

length(traing_img_names)
length(unique(traing_img_names))
str(traing_img_names)


training_img_byclass = sapply(classes, FUN = function(x) list.files(paste0("Training images/Photos_iterative_Sep2019/train/", x), recursive = T, full.names = F, include.dirs = F))

## number of training samples per class 
n_of_samples = sapply(training_img_byclass, length)


training_img_external_byclass = sapply(training_img_byclass, FUN = function(x) x[!str_detect(x, pattern = "^photoid")])
training_img_internal_byclass = sapply(training_img_byclass, FUN = function(x) x[str_detect(x, pattern = "^photoid")])

                                       
n_of_internal_samples = sapply(training_img_internal_byclass, length)
n_of_external_samples = sapply(training_img_external_byclass, length)

n_of_samples - (n_of_internal_samples + n_of_external_samples)

n_samples_df = (cbind( n_of_internal_samples, n_of_external_samples, n_of_samples))

rownames(n_samples_df) = classes_fullnames

library(xtable) 
xtable(rbind(n_samples_df, colSums(n_samples_df)), digits = 0)


write.xlsx(n_samples_df, file = "Output/Number_of_training_samples.xlsx", row.names=T)

barplot((t(n_samples_df[,1:2])), main = "Number of training images", las=2, beside=T)
barplot(log(n_of_samples, 10), main = "Log10 Number of training images", las=2)



 

## predicted tags
dt_MF = read.csv("TaggedResult_Nov2019_Middlefork/CSV/FlickrSeattle_AllPhotos.csv")
str(dt_MF)

dt_MF$site = "MF"
dt_MT = read.csv("TaggedResult_Feb2020_Mountainloop/CSV/Photos.csv")
dt_MT$site = "MT"

dt = rbind(dt_MF, dt_MT)

tags_dt = dt[,2:11]
probs_dt = as.matrix( dt[,12:21])

# length(table(unlist(tags_dt)))
library(abind)
library(plyr)

tags_df1 = sapply(tags_dt, as.character)
dim(tags_df1) <- dim(tags_df1)[1] * dim(tags_df1)[2] 
str(tags_df1)
dim(probs_dt) = dim(probs_dt)[1] * dim(probs_dt)[2] 
predictions_df = data.frame(Tag = tags_df1, Prob = probs_dt)
dim(predictions_df)
 
head(predictions_df)



predictions_v = tapply(predictions_df$Prob, predictions_df$Tag, FUN = c)  

# boxplot(predictions_v)


tags_dt = sapply(dt[,2:11], FUN = as.character)
probs_dt = dt[,12:21]

classidx_dt  = t(sapply(1:nrow(tags_dt) , FUN = function(x) match(classes, tags_dt[x,]  )))

classprob_dt = t(sapply(1:nrow(classidx_dt), FUN = function(x) sapply(classidx_dt[x,], FUN = function(x2) ifelse(is.na(x2), yes = 0, no = probs_dt[x,x2]))))

str(classprob_dt)

colnames(classprob_dt) = classes_fullnames


# corrplot(cor(classprob_dt[1:14839,], method = "pearson"), type = "upper",  method = "circle", addCoefasPercent = T)
pdf("Output/AFig_MFML_PearsonCorrelationbetweenclasses_n33189.pdf", width = 12, height = 12)

corrplot(cor(classprob_dt, method = "pearson"), type = "upper",  method = "ellipse", is.corr = T, diag = F, outline = F, addCoef.col = "black", number.cex = 0.8)
# corrplot(cor(classprob_dt[1:14839,], method = "pearson"), type = "upper",  method = "number", is.corr = T)
# corrplot(cor(classprob_dt, method = "pearson"), type = "upper",  method = "ellipse", is.corr = T)
# corrplot(cor(classprob_dt, method = "pearson"), type = "upper",  method = "number", is.corr = T, diag = F, outline = T)
# levelplot( (cor(classprob_dt, method = "pearson")))

dev.off()

# corrplot(cor(classprob_dt[1:14839,], method = "pearson"), type = "upper",  method = "circle", addCoefasPercent = T)
pdf("Output/AFig_MFML_SpearmanCorrelationbetweenclasses_n33189.pdf", width = 12, height = 12)

corrplot(cor(classprob_dt, method = "spearman"), type = "upper",  method = "ellipse", is.corr = T, diag = F, outline = F, addCoef.col = "black", number.cex = 0.8)
# corrplot(cor(classprob_dt[1:14839,], method = "pearson"), type = "upper",  method = "number", is.corr = T)
# corrplot(cor(classprob_dt, method = "pearson"), type = "upper",  method = "ellipse", is.corr = T)
# corrplot(cor(classprob_dt, method = "pearson"), type = "upper",  method = "number", is.corr = T, diag = F, outline = T)
# levelplot( (cor(classprob_dt, method = "pearson")))

dev.off()





tags_v = unlist(tags_dt[,1])
probs_v = unlist(probs_dt[, 1])



  


# boxplot(probs_v[tags_v == "fishing"])
mf_idx = dt$site == "MF"
mt_idx = dt$site == "MT"

# prob_avg = tapply(probs_v[mf_idx], INDEX = tags_v[mf_idx], FUN = mean, na.rm=T)

prob_l_mf = tapply(probs_v[mf_idx], INDEX = tags_v[mf_idx], FUN = c, na.rm=T)[classes]
prob_l_mt = tapply(probs_v[mt_idx], INDEX = tags_v[mt_idx], FUN = c, na.rm=T)[classes]


names(prob_l_mf) = classes_fullnames
names(prob_l_mt) = classes_fullnames

pdf("Output/AFig_ClassifierConfidence.pdf", width = 18, height = 10)

par(mfrow=c(1,2), mar=c(4,4,4,4), oma=c(4,4,4,4))
boxplot(prob_l_mf, main = "Middle Fork", las=2, ylab = "Confidence of the classifier")
boxplot(prob_l_mt, main = "Mountain Loop", las=2, ylab = "Confidence of the classifier")

dev.off()



# 
# # tagged images used not used in the training 
# 
# dt_evaluated_valid  = dt_evaluated[!(dt_evaluated$photo_id %in% traing_img_names), ] 
# 
# tagged_img_byclass = tapply(dt_evaluated_valid$photo_id, INDEX = dt_evaluated_valid$Top1_new, FUN = c)
# 




# # prediction accuracy
# 
# perf1 = read.csv("../TrainedWeights/InceptionResnetV2_Seattle_retrain_instabram_16classes_finetuning_iterative_final_val_acc0.88_performance_first?.csv")
# perf2 = read.csv("../TrainedWeights/InceptionResnetV2_Seattle_retrain_instabram_16classes_finetuning_iterative_final_val_acc0.88_performance.csv")
# 
# model_perf = rbind(perf2, perf1)
# model_perf$X = NULL
# 
# plot(121:144, model_perf$val_acc, type="l", ylim=c(0.5,1), col = "blue", ylab = "Classification Accuracy", xlab = "Training Epoch")
# lines(121:144, model_perf$acc, type="l", col = "red")
# legend("bottomright", legend = c("Training acc.", "Validation acc. (70/30)"), col = c("red", "blue"), lty=1, bty="n")
# legend("bottomleft", legend = c("N of training samples=5365", "N of validation samples=2329"), lty=0, bty="n")
# 



# barplot(prob_avg[-11])

# plot(as.numeric(tag_top1), prob_avg)
# tag_top1 = table(tags_dt$Top1)
# 
# barplot(tag_top1, ylab = "N of photos", las=2, main = "N of photos classified for the activity classes")
# barplot(log(tag_top1[], base = 10), ylab = "Log n", las=2,  main = "Log10 N of photos classified for the activity classes")
# 
# barplot(colMeans(probs_dt), ylim=c(0,1), main = "Dist. of Prediction uncertainty per rank", las=2 )







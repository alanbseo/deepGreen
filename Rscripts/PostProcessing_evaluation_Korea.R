library(latticeExtra)
library(corrplot)

setwd("~/Dropbox/KIT/FlickrEU/Korea/")

## predicted tags
csv_files = list.files("CSV", pattern = "\\.csv$", full.names = T)

dt = do.call("rbind", lapply(csv_files, FUN = read.csv))

str(dt)
cnames = names(table(unlist( dt[,2:11])))
n_species = length(cnames)
tags_dt = sapply(dt[,2:11], FUN = function(x) factor(x, levels = cnames))
probs_dt = dt[,12:21]

length(table(unlist(tags_dt)))

# probs_all = matrix(NA, nrow =nrow(tags_dt), ncol = n_species)
#   
probs_all_df = t(sapply(1:nrow(probs_dt), FUN = function(x) {
  occurred = match(cnames, tags_dt[x,])
  # occurred = occurred[!is.na(occurred)]
  probs = sapply(occurred, FUN = function(x2) probs_dt[x, x2])
  probs[sapply(probs, is.null)] <- 0
  unlist(probs)
}))

colnames(probs_all_df) = cnames

corrplot(cor(probs_all_df), type = "full", method = "ellipse")

probs_v = tapply(unlist(probs_dt), INDEX =  unlist(data.frame(tags_dt)), FUN = c)

boxplot(probs_v, las=2)



tag_top1 = table(tags_dt[,1])
barplot(tag_top1)


prob_top1_l = tapply(probs_dt[,1], tags_dt[,1], FUN = c)
boxplot(prob_top1_l)
# 
 

 

 


# boxplot(probs_v[tags_v == "fishing"])

prob_avg = tapply(probs_v, INDEX = tags_v, FUN = mean, na.rm=T)

prob_l = tapply(probs_v, INDEX = tags_v, FUN = c, na.rm=T)




## number of training samples per class 

classes = list.files("../../LabelledData/Seattle/Photos_iterative_Aug2019/train/")

n_of_sampls = sapply(classes, function(x) length(list.files(paste0("../../LabelledData/Seattle/Photos_iterative_Aug2019/train/", x))))

n_of_sampls = n_of_sampls[ names(prob_avg)]
barplot(n_of_sampls, main = "Number of training images", las=2)
barplot(log(n_of_sampls, 10), main = "Log10 Number of training images", las=2)





# barplot(prob_avg[-11])

# plot(as.numeric(tag_top1), prob_avg)

barplot(tag_top1, ylab = "N of photos", las=2, main = "N of photos classified for the activity classes")
barplot(log(tag_top1[], base = 10), ylab = "Log n", las=2,  main = "Log10 N of photos classified for the activity classes")

barplot(colMeans(probs_dt), ylim=c(0,1), main = "Dist. of Prediction uncertainty per rank", las=2 )

boxplot(prob_l, main = "Dist. of Prediction uncertainty per class", las=2, ylab = "Confidence of the classifier")



# prediction accuracy

perf1 = read.csv("../TrainedWeights/InceptionResnetV2_Seattle_retrain_instabram_16classes_finetuning_iterative_final_val_acc0.88_performance_first?.csv")
perf2 = read.csv("../TrainedWeights/InceptionResnetV2_Seattle_retrain_instabram_16classes_finetuning_iterative_final_val_acc0.88_performance.csv")

model_perf = rbind(perf2, perf1)
model_perf$X = NULL

plot(121:144, model_perf$val_acc, type="l", ylim=c(0.5,1), col = "blue", ylab = "Classification Accuracy", xlab = "Training Epoch")
lines(121:144, model_perf$acc, type="l", col = "red")
legend("bottomright", legend = c("Training acc.", "Validation acc. (70/30)"), col = c("red", "blue"), lty=1, bty="n")
legend("bottomleft", legend = c("N of training samples=5365", "N of validation samples=2329"), lty=0, bty="n")




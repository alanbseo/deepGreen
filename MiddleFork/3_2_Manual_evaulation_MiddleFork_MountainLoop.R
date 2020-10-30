
#backpacking: 97
#birdwatching: 52
#boating: 18
#camping: 81
#fishing: 2
#flooding: 9
#hiking: 3616
#horseriding: 22
#mtn_biking: 140
#noactivity: 10106
#otheractivities:307
#pplnoactivity: 17
#rockclimbing: 198
#swimming: 174
library(parallel)
library(doMC)
library(openxlsx)
# library(xlsx)
library(rJava)
library(caret)
library(gplots)
library(stringr)
library(gridExtra)

# Location of the dropbox shared folder
dropbox_location = "~/Dropbox/KIT/FlickrEU/Seattle/Seattle_TaggedData_BigSize/"

path_seattle = paste0(dropbox_location)
setwd(path_seattle)

classification<- read.csv("Manual evaluation_MiddleFork/classification.csv", sep= ",", header = T)
str(classification)
classification <- as.data.frame(classification)
sum(classification[,2]) # 14839


# Activity classes
classes = c("backpacking", "birdwatching", "boating", "camping", "fishing", "flooding", "hiking", "horseriding", "mtn_biking", "noactivity", "otheractivities", "pplnoactivity", "rock climbing", "swimming", "trailrunning")

n_classes = length(classes)

# library(wesanderson)
# names(wes_palettes)
# col_classes = rich.colors(n_classes)
# cols = wes_palette("GrandBudapest1", n_classes, type = c("continuous"))


c25 <- c(
  "dodgerblue2", "#E31A1C", # red
  "green4",
  "#6A3D9A", # purple
  "#FF7F00", # orange
  "black", "gold1",
  "skyblue2", "#FB9A99", # lt pink
  "palegreen2",
  "#CAB2D6", # lt purple
  "#FDBF6F", # lt orange
  "gray70", "khaki2",
  "maroon", "orchid1", "deeppink1", "blue1", "steelblue4",
  "darkturquoise", "green1", "yellow4", "yellow3",
  "darkorange4", "brown"
)
# pie(rep(1, 25), col = c25)
col_classes = c25[1:n_classes]

#####
# setwd("~/Dropbox/KIT/FlickrEU/FlickrCNN/Seattle/")
# list.files()[1]


# 
# 
# if (FALSE) { 
#   ### To reselect the middle fork photos and add Top2 labels
#   
#   
#   dt_evaluated = read.xlsx("Manual evaluation/Manual evalutation_MiddleFork_18Feb2020_n724.xlsx", 1)
#   
#   dt_evaluated$photo_id
#   
#   dt_csvdata = read.csv("Seattle_TaggedData_BigSize/FlickrSeattle_Tagging_Nov2019_Middlefork/CSV/FlickrSeattle_AllPhotos.csv")
#   dt_csvdata$Filename = as.character(dt_csvdata$Filename)
#   
#   dt_evaluated_metadata = dt_csvdata[match(dt_evaluated$photo_id, dt_csvdata$Filename),]
#   
#   
#   colnames(dt_evaluated)[2:3] = c("Top1", "Top1YN")
#   dt_res = cbind(dt_evaluated, Top2 = as.character(dt_evaluated_metadata$Top2), Top2YN = NA)
#   dt_res = dt_res[, c("photo_id", "Top1", "Top2", "Top1YN", "Top2YN", "note")]
#   # replace Top1 
#   
#   table(dt_evaluated$result, dt_evaluated_metadata$Top1)
#   
#   dt_res$Top1 =  as.character(dt_evaluated_metadata$Top1)
#   dt_res$Corrected = ifelse(dt_evaluated$Top1 !=  dt_evaluated_metadata$Top1, yes = "Corrected", no = "")
#   write.xlsx(dt_res, file = "Manual evaluation/Manual evalutation_MiddleFork_10July2020_n724_withTop2_corrected.xlsx")
# }

# Manually evaluated data (used the sampled photos)
dt_evaluated = read.xlsx("Manual evaluation_MiddleFork/Manual evalutation_MiddleFork_10July2020_n724_withTop2_corrected.xlsx", 1)
colnames(dt_evaluated)





# Evaluation function 
evaluateClassification <- function(pred_in, obs_in) {
  
  res = confusionMatrix( data = pred_in,  reference = obs_in, mode="everything")
  cm <- as.matrix(res)
  n = sum(cm) # number of instances
  nc = nrow(cm) # number of classes
  rowsums = apply(cm, 1, sum) # number of instances per class
  colsums = apply(cm, 2, sum) # number of predictions per class
  diag = diag(cm)  # number of correctly classified instances per class 
  
  precision = diag / colsums 
  recall = diag / rowsums 
  f1 = 2 * precision * recall / (precision + recall) 
  
  print(" ************ Confusion Matrix ************")
  print(cm)
  print(" ************ Diag ************")
  print(diag)
  print(" ************ Precision/Recall/F1 ************")
  print(data.frame(precision, recall, f1)) 
  # After that, you are able to find the macro F1.
  
  macroPrecision = mean(precision, na.rm=T)
  macroRecall = mean(recall, na.rm=T)
  macroF1 = mean(f1, na.rm=T)
  
  
  print(" ************ Macro Precision/Recall/F1 ************")
  res$macro =  data.frame(macroPrecision, macroRecall, macroF1)
  print(res$macro )
  
  return(res)
}


 


dt_evaluated$Top1YN =   str_trim(dt_evaluated$Top1YN)
dt_evaluated$Top2YN =   str_trim(dt_evaluated$Top2YN)

dt_evaluated$TrueClass = ifelse(dt_evaluated$Top1YN=="y", dt_evaluated$Top1_new, no = NA)  # fill by Top1
dt_evaluated$TrueClass[is.na(  dt_evaluated$TrueClass )] = ifelse(dt_evaluated[ is.na(dt_evaluated$TrueClass),]$Top2YN == "y", dt_evaluated[  is.na(dt_evaluated$TrueClass),"Top2"], no = NA) # and Top2

dt_evaluated$TrueClass[is.na(  dt_evaluated$TrueClass )] =  dt_evaluated$Class[is.na(  dt_evaluated$TrueClass )]
dt_evaluated$TrueClass[is.na(  dt_evaluated$TrueClass )]

dt_evaluated = dt_evaluated[!is.na(  dt_evaluated$TrueClass ),]


pred_in = factor(dt_evaluated$Top1_new, classes)
obs_in = factor(dt_evaluated$TrueClass, classes)
table(obs_in)


result_Top1 <- evaluateClassification(pred_in, obs_in)
result_Top2 <- evaluateClassification( factor(dt_evaluated$Top2, classes), obs_in)



pdf("Output/ClassificationAccuracy_MiddleFork.pdf", width = 18, height = 10)
par(mfrow=c(1,1))

plot(0, col = "white", axes=F,  main = "Confusion matrix (Top1); reference in columns and predictions in rows", xlab=NA, ylab=NA)
grid.table(result_Top1$table)
plot(0, col = "white",  axes=F, main = "Confusion matrix (Top2); reference in columns and predictions in rows", xlab=NA, ylab=NA)
grid.table(result_Top2$table)
par(mfrow=c(1,2), mar=c(4,4,4,4), oma=c(10,0,0,0))

d1 = data.frame(c(result_Top1$overall, result_Top1$macro))
barplot(as.matrix(d1), beside=T, las=2, ylim=c(0,1), main = "Overall classification accuracy in the Middle Fork site (based on Top1)")
d2 = as.matrix(data.frame(c(result_Top2$overall, result_Top2$macro)))

barplot(d2, beside=T, las=2, ylim=c(0,1), main ="Overall classification accuracy in the Middle Fork site (Based on Top2)")

par(mfrow=c(1,1))

barplot(result_Top1$byClass, beside=T, las=2, col = col_classes, ylim=c(0,1), main = "Accuracy by class (based on Top1)")
legend("bottomleft", legend = classes, fill = col_classes, bg="white", cex=0.8)
barplot((result_Top2$byClass), beside=T, las=2, col = col_classes, ylim=c(0,1), main ="Accuracy by  (based on Top2)")
legend("bottomleft", legend = classes, fill = col_classes, bg="white", cex=0.8)


dev.off()




### Evaluate the new mountain loop site 

  
# c_idx = 1 


resobs_l = foreach(c_idx = 1:length(classes), .errorhandling = "stop") %do% { 
  
  cls_tmp = classes[c_idx] 
  dt = read.xlsx(paste0("Manual Evaluation_MountainLoop/Mountain Loop/", cls_tmp, "_samples.xlsx"))
  # str(dt)
  if(ncol(dt)<6) { 
    dt$Note = NA 
  } else { 
    colnames(dt)[6] = "Note"
    
  }
  
  dt$Top1YN =   str_trim(dt$Top1YN)
  dt$Top2YN =   str_trim(dt$Top2YN)
  
  dt$TrueClass = ifelse(dt$Top1YN=="y", dt$Top1, no = NA)  # fill by Top1
  dt$TrueClass[is.na(  dt$TrueClass )] = ifelse(dt[ is.na(dt$TrueClass),]$Top2YN == "y", dt[  is.na(dt$TrueClass),"Top2"], no = NA) # and Top2
  
  dt$TrueClass[is.na(  dt$TrueClass )]  = dt[ is.na(dt$TrueClass),]$Note # fill by Note 
  dt$TrueClass = as.character(factor(dt$TrueClas, levels = classes)) # and factor using the classes. Non-class notes are removed here. 
  
  dt$Top1and2 = ifelse(dt$Top1YN=="n" & dt$Top2YN == "y", yes = dt$Top2, no = dt$Top1) # Top2 only if Top1 is wrong but Top2 is right.   
  
  
  
  
  
  # dt$Top1
  # dt$TrueClass
  
  # table(res1)
  # table(res2)
  
  return(dt[, c("Filename", "TrueClass", "Top1", "Top2", "Top1and2", "Top1YN", "Top2YN", "Note")])
  
}




dt_tmp = do.call("rbind", resobs_l) # 494 
dt_tmp = dt_tmp[!is.na(dt_tmp$TrueClass),] # 491  

table(dt_tmp$Top1YN, dt_tmp$Top2YN)
table(dt_tmp$Top1, dt_tmp$Top1and2)

#  library(survey)
# Tag = c("A", "A", "A", "B", "B", "C")
# Weight = c(0.1, 0.1, 1, 0.5, 0.5, 0.7)
# df_tmp = data.frame(id = 1:length(Tag), Tag, Weight)
# dclus1 <- svydesign(id=~id, weights=~Weight, data=df_tmp, Tag=~Tag)
# Tag_weighted <- svytable(~Tag, dclus1)
# Tag_weighted
# 

dt_csvdata = read.csv("TaggedResult_Feb2020_Mountainloop/CSV/Photos.csv")
dt_csvdata$Filename = as.character(dt_csvdata$Filename)


dt_mtloop_tag  = unlist( sapply(1:10, FUN = function(x) as.character(dt_csvdata[, paste0("Top", x)])))
dt_mtloop_prob  = unlist(sapply(1:10, FUN = function(x) as.numeric(dt_csvdata[, paste0("Prob", x)])))

length(dt_mtloop_tag)
length(dt_mtloop_prob)

weighedsum_mtloop = tapply(dt_mtloop_prob, INDEX = dt_mtloop_tag, FUN = sum )


naivesum_mtloop = table(as.character(dt_csvdata$Top1))
naivesumTop12_mtloop = table(c(as.character(dt_csvdata$Top1), as.character( dt_csvdata$Top2)))

wt_prop = (weighedsum_mtloop / sum(weighedsum_mtloop))
nv_prop  = (naivesum_mtloop / sum(naivesum_mtloop))
nv12_prop  = (naivesumTop12_mtloop / sum(naivesumTop12_mtloop))

prop_tb = rbind( nv_prop, nv12_prop, wt_prop)*100
rownames(prop_tb) = c("Top1", "Top1+2", "Weighted")


pdf("Output/PercentPredictedClasses_MountainLoop.pdf", width = 18, height = 10)

par(mfrow=c(1,1))
barplot(prop_tb , beside=T, col = c("red", "blue", "green"), ylim=c(0,100), ylab = "Fraction of the class (%)", main = paste0( "Mountain Loop (n=", nrow(dt_csvdata), ")"))
legend("topright", legend = c("Top1 only", "Top1 + Top2", "Weighted (Top1 ~ 10)"), col = c("red", "blue", "green"), pch=15)
plot(0, col = "white",  axes=F, main = "Proportion (%)", xlab=NA, ylab=NA)
grid.table(round(prop_tb, 2))
dev.off()





pred_in = factor(dt_tmp$Top1, classes)
obs_in = factor(dt_tmp$TrueClass, classes)



result_Top1 <- evaluateClassification(pred_in, obs_in)
result_Top2 <- evaluateClassification( factor(dt_tmp$Top2, classes), obs_in)

writeLines(result_Top1, con = file("Output/ClassificationAccuracy_MountainLoop_Top1.txt", open = "rw"))
writeLines(result_Top2, con = file("Output/ClassificationAccuracy_MountainLoop_Top2.txt", open = "rw"))





 
pdf("Output/ClassificationAccuracy_MountainLoop.pdf", width = 18, height = 10)
par(mfrow=c(1,1))

plot(0, col = "white", axes=F,  main = "Confusion matrix (Top1); reference in columns and predictions in rows", xlab=NA, ylab=NA)
grid.table(result_Top1$table)
plot(0, col = "white",  axes=F, main = "Confusion matrix (Top2); reference in columns and predictions in rows", xlab=NA, ylab=NA)
grid.table(result_Top2$table)
par(mfrow=c(1,2), mar=c(4,4,4,4), oma=c(10,0,0,0))

d1 = data.frame(c(result_Top1$overall, result_Top1$macro))
barplot(as.matrix(d1), beside=T, las=2, ylim=c(0,1), main = "Overall classification accuracy in the Mountain Loop site (based on Top1)")
d2 = as.matrix(data.frame(c(result_Top2$overall, result_Top2$macro)))

barplot(d2, beside=T, las=2, ylim=c(0,1), main ="Overall classification accuracy in the Mountain Loop site (Based on Top2)")

par(mfrow=c(1,1))

barplot(result_Top1$byClass, beside=T, las=2, col = col_classes, ylim=c(0,1), main = "Accuracy by class (based on Top1)")
legend("bottomleft", legend = classes, fill = col_classes, bg="white", cex=0.8)
barplot((result_Top2$byClass), beside=T, las=2, col = col_classes, ylim=c(0,1), main ="Accuracy by  (based on Top2)")
legend("bottomleft", legend = classes, fill = col_classes, bg="white", cex=0.8)


dev.off()


 





# 
#  
# 
# library(DescTools)
# 
# # The entropy quantifies the expected value of the information contained in a vector. 
# # The mutual information is a quantity that measures the mutual dependence of the two random variables.
# 
# 
# # NOT RUN {
# Entropy(as.matrix(rep(1/8, 8)))
# 
# # http://r.789695.n4.nabble.com/entropy-package-how-to-compute-mutual-information-td4385339.html
# x <- as.factor(c("a","b","a","c","b","c")) 
# y <- as.factor(c("b","a","a","c","c","b")) 
# 
# Entropy(table(x), base=exp(1))
# Entropy(table(y), base=exp(1))
# Entropy(x, y, base=exp(1))
# 
# # Mutual information is 
# Entropy(table(x), base=exp(1)) + Entropy(table(y), base=exp(1)) - Entropy(x, y, base=exp(1))
# MutInf(x, y, base=exp(1))
# 
# Entropy(table(x)) + Entropy(table(y)) - Entropy(x, y)
# MutInf(x, y, base=2)
# 
# # http://en.wikipedia.org/wiki/Cluster_labeling
# tab <- matrix(c(60,10000,200,500000), nrow=2, byrow=TRUE)
# MutInf(tab, base=2) 
# 
# d.frm <- Untable(as.table(tab))
# str(d.frm)
# MutInf(d.frm[,1], d.frm[,2])
# 
# table(d.frm[,1], d.frm[,2])
# 
# MutInf(table(d.frm[,1], d.frm[,2]))


# Ranking mutual information can help to describe clusters
#
#   r.mi <- MutInf(x, grp)
#   attributes(r.mi)$dimnames <- attributes(tab)$dimnames
# 
#   # calculating ranks of mutual information
#   r.mi_r <- apply( -r.mi, 2, rank, na.last=TRUE )
#   # show only first 6 ranks
#   r.mi_r6 <- ifelse( r.mi_r < 7, r.mi_r, NA) 
#   attributes(r.mi_r6)$dimnames <- attributes(tab)$dimnames
#   r.mi_r6
# }


 
 

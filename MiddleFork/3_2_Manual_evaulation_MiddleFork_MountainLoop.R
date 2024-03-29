
library(parallel)
library(doSNOW)
library(openxlsx)
# library(xlsx)
# library(rJava)
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

classes_fullnames = c("Backpacking", "Bird watching", "Boating", "Camping", "Fishing", "Flooding", "Hiking", "Horse riding", "Mountain biking", "No activity", "Other activities", "People no activity", "Rock climbing", "Swimming", "Trail running")


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
dt_evaluated_old = read.xlsx("Manual evaluation_MiddleFork/Manual evalutation_MiddleFork_10July2020_n724_withTop2_corrected.xlsx", 1)
colnames(dt_evaluated_old)


## training photo ids 

all_img_names = basename(list.files("../../UnlabelledData/Seattle/FlickrSeattle_AllPhotos/", recursive = T, full.names = F, include.dirs = F))

traing_img_names = basename(list.files("Training images/Photos_iterative_Sep2019/train/", recursive = T, full.names = F, include.dirs = F))



table(all_img_names %in% traing_img_names)

length(traing_img_names)

table(dt_evaluated_old$photo_id %in% traing_img_names)


dt_evaluated_old = dt_evaluated_old[(!dt_evaluated_old$photo_id %in% traing_img_names),]

# str(dt_evaluated)


### new eval (Dec 2020)
dt_evaluated_new = read.xlsx("Manual evaluation_MiddleFork/Manual evalutation_MiddleFork_16Dec2020_n379_withTop2_newphotos.xlsx", 1)



dt_evaluated = rbind(dt_evaluated_old, dt_evaluated_new)

table(dt_evaluated$photo_id %in% traing_img_names)




 







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

 



dt_evaluated = dt_evaluated[!is.na(  dt_evaluated$TrueClass ),]


# result_Top2 <- evaluateClassification( factor(dt_evaluated$Top2, classes), obs_in)


pred_woflood_in = dt_evaluated$Top1_new
pred_woflood_in[pred_woflood_in=="flooding"] = "noactivity"
obs_woflood_in= dt_evaluated$TrueClass
obs_woflood_in[obs_woflood_in=="flooding"] = "noactivity"



pred_woflood_in = factor(pred_woflood_in, classes[-6])
obs_woflood_in = factor(obs_woflood_in, classes[-6])

table(is.na(obs_woflood_in))

# Top1 + Top2
pred_woflood_Top2_in_org = dt_evaluated$Top2
pred_woflood_Top2_in_org[pred_woflood_Top2_in_org=="flooding"] = "noactivity"


pred_woflood_top2_in = pred_woflood_in

table((pred_woflood_in!=obs_woflood_in))

top1and2_idx = (pred_woflood_in!=obs_woflood_in) & (pred_woflood_Top2_in_org==obs_woflood_in)


pred_woflood_top2_in[top1and2_idx] = pred_woflood_Top2_in_org[top1and2_idx]

length(obs_woflood_in)
table(obs_woflood_in)

# without flooding
result_Top1 <- evaluateClassification(pred_woflood_in, obs_woflood_in)
result_Top2 <- evaluateClassification(pred_woflood_top2_in, obs_woflood_in)

confmat = result_Top1$table

rownames(confmat) = colnames(confmat) = classes_fullnames[-6]


### eval without minor classes 

minor_idx = !(pred_woflood_in %in% c("trailrunning", "fishing", "horse"))

pred_min = pred_woflood_in[minor_idx]
obs_min = obs_woflood_in[minor_idx]

pred_min2 = factor(pred_min)
obs_min2 = factor(obs_min, levels = levels(factor(pred_min)))
result_Top1_wominor <- evaluateClassification(pred_min2, obs_min2)


mean(result_Top1$byClass[, "Balanced Accuracy"], na.rm=T)

mean(result_Top1$byClass[, "Balanced Accuracy"], na.rm=T)
result_Top1$overall
result_Top1$macro



library(xtable)
xtable(confmat)


col <- function(x){
  if(x>100) { 
    paste("\\textcolor{red}{", formatC(x, dig=0, format="f"), "}")
  } else if (x>10) {
    paste("\\textcolor{blue}{", formatC(x, dig=0, format="f"), "}")
  } else if (x>0) {
    paste("\\textcolor{black}{", formatC(x, dig=0, format="f"), "}")
  } else {
    paste("\\textcolor{gray}{", formatC(x, dig=0, format="f"), "}")
  }
}

confmat_color <- apply(confmat, MARGIN = 1:2, col)


rownames(confmat_color) = colnames(confmat_color) = classes_fullnames[-6]

confmat_out = rbind( cbind(confmat_color, Total=rowSums(confmat)), c(colSums(confmat), ""))


print(xtable(confmat_out),sanitize.text.function = function(x){x})


 
 


library(Thermimage)
library(viridis)
library(RColorBrewer)

library(latticeExtra)


confmat_MF = result_Top1$table
confmat_MF = t(flip.matrix(t(confmat_MF)))

rownames(confmat_MF) =   (classes_fullnames[-6]) 
colnames(confmat_MF) =  rev(classes_fullnames[-6])
 
confmat_MF_df = data.frame(expand.grid(x=1:14, y=1:14), value = as.numeric(confmat_MF), lb = as.numeric(confmat_MF))

## Applied to the example data in my other answer, this will produce
## an identical plot


confmat_MF_plot = confmat_MF           
confmat_MF_plot[confmat_MF_plot>125] = 125
Obj <- 
  levelplot((confmat_MF_plot), xlab="Prediction", ylab="Reference", col.regions = c(rep("white", 1), colorRampPalette(brewer.pal("YlGnBu", n = 9))(32)), at=c(-1,0, seq(1, 130, length=30)), scales=list(x=list( rot=90)))+ xyplot(y ~ x, data = confmat_MF_df,
                                                                                                                                                                                           panel = function(y, x, ...) {
                                                                                                                                                                                             ltext(x = x, y = y, labels = confmat_MF_df$lb, cex = 1, font = 1,fontfamily = "HersheySans")
                                                                                                                                                                                           })


print(Obj)

pdf("Output/Fig_ConfusionMatrix_MF.pdf", width = 8, height = 8)

print(Obj)
dev.off()


 






writeLines(capture.output(print(result_Top1)), con = ("Output/ClassificationAccuracy_MiddleFork_Top1.txt"))
# writeLines(capture.output(print(result_Top2)), con = ("Output/ClassificationAccuracy_MiddleFork_Top2.txt"))



selected_metrics = c("Balanced Accuracy","F1", "Precision", "Recall", "Sensitivity", "Specificity")

# pdf("Output/ClassificationAccuracy_MiddleFork.pdf", width = 18, height = 10)
# 
# par(mfrow=c(1,1))
# 
# plot(0, col = "white", axes=F,  main = "Confusion matrix (Top1); reference in columns and predictions in rows", xlab=NA, ylab=NA)
# grid.table(result_Top1$table)
# plot(0, col = "white",  axes=F, main = "Confusion matrix (Top2); reference in columns and predictions in rows", xlab=NA, ylab=NA)
# grid.table(result_Top2$table)
# par(mfrow=c(1,2), mar=c(4,4,4,4), oma=c(10,1,1,1))
# 
# d1 = data.frame(c(result_Top1$overall, result_Top1$macro))
# barplot(as.matrix(d1), beside=T, las=2, ylim=c(0,1), main = "Overall classification accuracy in the Middle Fork site (based on Top1)")
# d2 = as.matrix(data.frame(c(result_Top2$overall, result_Top2$macro)))
# 
# barplot(d2[c("Accuracy", "F1", "macroPrecision", "macroRecall")], beside=T, las=2, ylim=c(0,1), main ="Overall classification accuracy in the Middle Fork site (Based on Top2)")
# dev.off()


res_top1_MF = data.frame(result_Top1$byClass[,rev(selected_metrics)])
rownames(res_top1_MF) = classes_fullnames[-6]
write.xlsx(res_top1_MF, file = "Summary tables and figures (working on it)/Table_MF_perfomance.xlsx", row.names=T)

balacc = res_top1_MF$Balanced.Accuracy
balacc[is.na(balacc)] = 0

f1_MF = res_top1_MF$F1
f1_MF[is.na(f1_MF)] = 0

ord_MF = order(f1_MF, decreasing = T)
res_top1_MF_ord = res_top1_MF[ord_MF,]
colnames(res_top1_MF_ord)[5:6] = c("F1-score", "Balanced Accuracy")

pdf("Output/ClassificationAccuracy_MiddleFork_Top1.pdf", width = 18, height = 14)

par(mfrow=c(1,2), mar=c(4,2,4,0), oma=c(0,14,4,0))

barplot(as.matrix(res_top1_MF_ord), beside=T, las=1, col =  (col_classes[-6][ord_MF]),  main ="", horiz=T, cex.names = 2, cex.axis=1.5)
axis(1, at = seq(0.1, 0.9, 0.2), labels = rep("", 5))

plot.new()
legend("bottomleft",  legend = rev(classes[-6][ord_MF]), fill = rev(col_classes[-6][ord_MF]), bg="white",   cex=2)

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



 

pred_woflood_ML_in = dt_tmp$Top1
pred_woflood_ML_in[pred_woflood_ML_in=="flooding"] = "noactivity"

obs_woflood_ML_in= dt_tmp$TrueClass
obs_woflood_ML_in[obs_woflood_ML_in=="flooding"] = "noactivity"

pred_woflood_ML_in = factor(pred_woflood_ML_in, classes[-6])
obs_woflood_ML_in = factor(obs_woflood_ML_in, classes[-6])

 


length(obs_woflood_ML_in)
table(obs_woflood_ML_in)

# without flooding
result_ML_Top1 <- evaluateClassification(pred_woflood_ML_in, obs_woflood_ML_in)




# Top1 + Top2
pred_woflood_Top2_ML_in_org = dt_tmp$Top2
pred_woflood_Top2_ML_in_org[pred_woflood_Top2_ML_in_org=="flooding"] = "noactivity"


pred_woflood_top2_ML_in = pred_woflood_ML_in

table((pred_woflood_ML_in!=obs_woflood_ML_in))

top1and2_ML_idx = (pred_woflood_ML_in!=obs_woflood_ML_in) & (pred_woflood_Top2_ML_in_org==obs_woflood_ML_in)


pred_woflood_top2_ML_in[top1and2_ML_idx] = pred_woflood_Top2_ML_in_org[top1and2_ML_idx]

 
# without flooding
result_ML_Top2 <- evaluateClassification(pred_woflood_top2_ML_in, obs_woflood_ML_in)






 

confmat_ML = result_ML_Top1$table
confmat_ML = t(flip.matrix(t(confmat_ML)))

rownames(confmat_ML) =   (classes_fullnames[-6]) 
colnames(confmat_ML) =  rev(classes_fullnames[-6])
 


confmat_df = data.frame(expand.grid(x=1:14, y=1:14), value = as.numeric(confmat_ML), lb = as.numeric(confmat_ML))

## Applied to the example data in my other answer, this will produce
## an identical plot
 
  
confmat_ML_plot = confmat_ML

confmat_ML_plot[confmat_ML_plot>55] = 55

Obj <- 
  levelplot((confmat_ML_plot), xlab="Prediction", ylab="Reference", col.regions = c(rep("white", 1), colorRampPalette(brewer.pal("YlGnBu", n = 9))(32)), at=c(-1,0, seq(1, 60, length=30)), scales=list(x=list( rot=90)))+xyplot(y ~ x, data = confmat_df, panel = function(y, x, ...) {
           ltext(x = x, y = y, labels = confmat_df$lb, cex = 1, font = 1,fontfamily = "HersheySans")
         })

print(Obj)

pdf("Output/Fig_ConfusionMatrix_ML.pdf", width = 8, height = 8)

print(Obj)
dev.off()

 

























confmat_ML_color <- apply(confmat_ML, MARGIN = 1:2, col)


rownames(confmat_ML_color) = colnames(confmat_ML_color) = classes_fullnames[-6]

confmat_ML_out = rbind( cbind(confmat_ML_color, Total=rowSums(confmat_ML)), c(colSums(confmat_ML), ""))


print(xtable(confmat_ML_out),sanitize.text.function = function(x){x})




 


result_ML_Top1_all <- evaluateClassification(pred_woflood_ML_in, obs_woflood_ML_in)
# result_Top2 <- evaluateClassification( factor(dt_tmp$Top2, classes), obs_in)
writeLines(capture.output(print(result_ML_Top1_all)), con = ("Output/ClassificationAccuracy_MountainLoop_Top1.txt"))
# writeLines(capture.output(print(result_Top2)), con = ("Output/ClassificationAccuracy_MountainLoop_Top2.txt"))



 
res_ML_top1 = data.frame(result_ML_Top1_all$byClass[,rev(selected_metrics)])
rownames(res_ML_top1) = classes_fullnames[-6]
write.xlsx(res_ML_top1, file = "Summary tables and figures (working on it)/Table_ML_perfomance.xlsx", row.names=T)

balacc = res_ML_top1$Balanced.Accuracy
balacc[is.na(balacc)] = 0

f1_ML = res_ML_top1$F1
f1_ML[is.na(f1_ML)] = 0

# ord_ML = order(f1_ML, decreasing = T)
res_ML_top1_ord = res_ML_top1[ord_MF,]
colnames(res_ML_top1)[5:6] = c("F1-score", "Balanced Accuracy")

pdf("Output/Fig4_ClassificationAccuracy_Top1_oldversion.pdf", width = 16, height = 8)

par(mfrow=c(2,2), mar=c(4,4,1,1), oma=c(0,1,4,0), xpd=NA)


toplot = as.matrix(res_top1_MF_ord[, 5:6])
toplot[is.na(toplot)] = 0 
barplot(toplot, beside=T, las=1, col =  (col_classes[-6][ord_MF]),  main ="", horiz=F, cex.names = 1.5, cex.axis=1.5)
mtext(text = "(a)", side = 3, outer = F, adj = 0, padj =-1, at = -3.2, cex=2.5)
text( x=c(13,28), y=0.05, labels="n.d.", cex=1.2) # side = 3, outer = F, adj = 0, padj =-1, at = -3.2, cex=2.5)

plot.new()
# legend("topleft",  legend = rev(classes[-6][ord_MF]), fill = rev(col_classes[-6][ord_MF]), bg="white",   cex=2)

toplot_ml = as.matrix(res_ML_top1_ord[, 5:6])
toplot_ml[is.na(toplot_ml)] = 0 

barplot(toplot_ml, beside=T, las=1, col =  (col_classes[-6][ord_MF]),  main ="", horiz=F, cex.names = 1.5, cex.axis=1.5)
mtext(text = "(b)", side = 3, outer = F, adj=0, at=-3.2, padj=-1, cex=2.5)
text( x=c(14,29), y=0.05, labels="n.d.", cex=1.2) # side = 3, outer = F, adj = 0, padj =-1, at = -3.2, cex=2.5)

plot.new()
legend("bottomleft",  legend =  (classes[-6][ord_MF]), fill =  (col_classes[-6][ord_MF]), bg="white",   cex=2)


dev.off()

pdf("Output/Fig4_ClassificationAccuracy_Top1.pdf", width = 8, height = 6)

par(mfrow=c(1,1), mar=c(8,5,1,4), oma=c(0,1,4,0), xpd=NA)


toplot = as.matrix(rbind(res_top1_MF_ord[, 5], res_ML_top1_ord[, 5]))
toplot[is.na(toplot)] = 0 
col_tmp = viridis(2)

barplot(toplot, beside=T, las=2, col = col_tmp,  main ="", horiz=F, cex.names = 1.5, cex.axis=1.2, cex.lab=1.2, ylab = "F1-Score")
# axis(2, # , at = (1:14)*3 -1 , labels = classes[-6][ord_MF], srt=45, las=2, line = NA, lwd = 0)
text(y=rep(-0.03, 14), x= (1:14)*3 -1, labels = classes[-6][ord_MF], srt=60, adj=1)
# text( x=c(13,28), y=0.05, labels="n.d.", cex=1.2) # side = 3, outer = F, adj = 0, padj =-1, at = -3.2, cex=2.5)

legend("topright",  legend =  c("Middle Fork", "Mountain Loop"), fill =  col_tmp, bg="white", cex=1.2, inset = c(0.25, 1.3), ncol = 2)


dev.off()



n_samples_df = read.xlsx("Output/Number_of_training_samples.xlsx")[-6,] # exclude flooding

# plot(n_samples_df$n_of_internal_samples, result_Top1$byClass[,"Precision"])

par(mfrow=c(1,1))

plot(n_samples_df[, "n_of_samples"], result_Top1$byClass[,"Precision"])
plot(n_samples_df[, "n_of_samples"], result_Top1$byClass[,"Recall"])
plot(n_samples_df[, "n_of_samples"], result_Top1$byClass[,"Specificity"])
plot(n_samples_df[, "n_of_samples"], result_Top1$byClass[,"Balanced Accuracy"])

plot(n_samples_df[, "n_of_samples"], result_Top1$byClass[,"Sensitivity"])


# Middle Fork

nona_idx = !(is.na(result_Top1$byClass[,"F1"]))

cf_1 = cor.test(n_samples_df[nona_idx, "n_of_samples"], result_Top1$byClass[nona_idx,"F1"])
cf_1 = cor.test(n_samples_df[ , "n_of_samples"], result_Top1$byClass[ ,"F1"])

cor.test(n_samples_df[ , "n_of_samples"], result_Top1$byClass[ ,"F1"])
cor.test(n_samples_df[ , "n_of_samples"], result_Top1$byClass[ ,"F1"], method = "spearman")

cf_2 = cor.test(n_samples_df[, "n_of_internal_samples"], result_Top1$byClass[,"F1"])
cf_3 = cor.test(n_samples_df[, "n_of_external_samples"], result_Top1$byClass[,"F1"])


cf_1 = cor.test(n_samples_df[, "n_of_samples"], result_Top1$byClass[,"F1"])



cb_1 = cor.test(n_samples_df[, "n_of_samples"], result_Top1$byClass[,"Balanced Accuracy"])
cb_2 = cor.test(n_samples_df[, "n_of_internal_samples"], result_Top1$byClass[,"Balanced Accuracy"])
cb_3 = cor.test(n_samples_df[, "n_of_external_samples"], result_Top1$byClass[,"Balanced Accuracy"])


c_l = list(cf_1, cf_2, cf_3, cb_1, cb_2, cb_3)

c_df = data.frame(t(sapply(c_l, FUN = function(x) c(x$estimate, x$conf.int, x$p.value))))
rownames(c_df)= c("F1_all", "F1_int", "F1_ext", "BA_all", "BA_int", "BA_ext")
colnames(c_df) = c("r", "r_025", "r_975", "p")

c_df = c_df[1:3,]


pdf("Output/AFig_Correlation_NumberOfSamples_F1.pdf", width = 8, height = 6)

plotCI(x = 1:3, y = c_df$r, ui = c_df$r_975, li = c_df$r_025, ylab= "Pearson's r", xlab="", labels = "", main = "Correlation between # of samples and F1-Score (MF)", axes=F, pch=20, cex=1)
axis(side=1, at=1:3, labels = c("All (p=0.41)", "Internal (p=0.34)", "External (p=0.88)"))
axis(side = 2)

abline(a = 0, b = 0, lty=2)

dev.off()

# plot(n_samples_df[], median_prob_mf_v)
# 
# dt = data.frame(Res = result_Top1$byClass[,"Specificity"], SampleNo = n_of_samples[-6], Prob = median_prob_mf_v)
# 
# lm1 = lm(Res ~ ., data = dt)
# anova(lm1)



emptycol = rep(NA, nrow(res_top1_MF))


print(xtable(cbind(res_top1_MF[,c(5,6,4:1)], res_ML_top1[,c(5,6,4:1)])), NA.string ="\\multicolumn{1}{r}{-}")

# ### Recall and precision
# ord_ML = order(f1_ML, decreasing = T)

# res_ML_top1_etc = data.frame(result_ML_Top1_all$byClass[,c("Specificity", "Sensitivity", "Precision", "Recall")])
# 
# res_ML_top1_ord = res_ML_top1[ord_MF,]
# colnames(res_ML_top1)[5:6] = c("F1-score", "Balanced Accuracy")
# 
# pdf("Output/Fig4_ClassificationAccuracy_Top1.pdf", width = 16, height = 8)
# 
# par(mfrow=c(2,2), mar=c(4,4,1,1), oma=c(0,1,4,0), xpd=NA)
# 
# barplot(as.matrix(res_top1_MF_ord[, 5:6]), beside=T, las=1, col =  (col_classes[-6][ord_MF]),  main ="", horiz=F, cex.names = 1.5, cex.axis=1.5)
# mtext(text = "(a)", side = 3, outer = F, adj = 0, padj =-1, at = -3.2, cex=2.5)
# plot.new()
# # legend("topleft",  legend = rev(classes[-6][ord_MF]), fill = rev(col_classes[-6][ord_MF]), bg="white",   cex=2)
# 
# barplot(as.matrix(res_ML_top1_ord[, 5:6]), beside=T, las=1, col =  (col_classes[-6][ord_MF]),  main ="", horiz=F, cex.names = 1.5, cex.axis=1.5)
# mtext(text = "(b)", side = 3, outer = F, adj=0, at=-3.2, padj=-1, cex=2.5)
# plot.new()
# legend("bottomleft",  legend =  (classes[-6][ord_MF]), fill =  (col_classes[-6][ord_MF]), bg="white",   cex=2)
# 
# 
# dev.off()
# 
# 
# 
# 
# 



 




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





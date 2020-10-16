setwd("~/Dropbox/KIT/FlickrEU/FlickrCNN/Seattle/Survey/")
data <- read.csv("survey_activities_data.csv", sep = ";")

head(data)
str(data)
colSums(data[,5:21], na.rm = T)
rank(colSums(data[,5:21], na.rm = T))
sort(colSums(data[,5:21], na.rm = T), decreasing = T)
colSums(data[,5:21], na.rm = T)/nrow(data) * 100
sort((colSums(data[,5:21], na.rm = T)/nrow(data) * 100), decreasing = T)

rowSums(data[,5:21], na.rm = T) 
max(rowSums(data[,5:21], na.rm = T)) # 9
table(rowSums(data[,5:21], na.rm = T))
#  0   1   2   3   4   5   6   7   9 
# 48 293 136  63  32  10   7   2   4 
nrow(data) # 595



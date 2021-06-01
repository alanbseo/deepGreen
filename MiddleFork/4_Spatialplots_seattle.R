library(rgdal)
library(httr)
library(RCurl)
library(rjson)
# library(jsonlite)
library(raster)
library(rgdal)
library(rgeos)
library(doMC)
library(openxlsx)
library(stringr)

library(gplots)
library(RColorBrewer)
library(randomcoloR)


library(OpenStreetMap)
# install.packages("osmar")
# library(osmar)

# # library(ggplot2)
 library(latticeExtra)

library(corrplot)
library(reshape2)

library(survey)

# Location of the dropbox shared folder
dropbox_location = "~/Dropbox/KIT/FlickrEU/Seattle/Seattle_TaggedData_BigSize/"

path_seattle = paste0(dropbox_location)
setwd(path_seattle)
 
workdir = "./" # "../FlickrCNN/Seattle/"
gisdir = "GIS and input data/"
# save.image(paste0(workdir, savedir, "/Flickr_CR_workspace_metadata_download_17Aug2019.RData"))
# load(paste0(workdir, savedir, "/Flickr_CR_workspace_metadata_download_17Aug2019_2.RData"))


# locations 
path_data =  ""




n.thread <- detectCores() # 1 
proj4.LL <- "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs +towgs84=0,0,0"
proj4.DHDN <- "+proj=tmerc +lat_0=0 +lon_0=12 +k=1 +x_0=4500000 +y_0=0 +ellps=bessel +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs" # epsg:31468

# Login credentials
# api.key <- "" # An API key for Flickr API
# api.secret <- "" # Not used
# apikey_con = file("Flickr_API_KEY.txt", open = "w")
# writeLines(api.key, con = apikey_con)
# close(apikey_con)

# apikey_con = file("Flickr_API_KEY.txt", open = "r")
# readLines(apikey_con)
# close(apikey_con)
#  



## Read coordinates 

doMiddleFork = FALSE 
doMountainLook = TRUE

if (doMiddleFork) { 
    
    
    aoi.poly.in = readOGR( dsn = paste0(workdir, gisdir), layer = "middlefork_AOI")  
    n.points <- length(aoi.poly.in)
    
    plot(aoi.poly.in)
    
    
    aoi_coords = read.csv("GIS and input data/mbsmiddlefork_flickrimages_20191027.csv")
    aoi_sp = SpatialPoints(data.frame(x=aoi_coords$longitude, y = aoi_coords$latitude), proj4string = crs(proj4.LL))
    plot(aoi_sp, add=T)
    names(aoi_coords)[1] = "PhotoID"
    
    ### Read predicted tags 
    path_predictedtags = paste0("TaggedResult_Nov2019_Middlefork/CSV/")
    csv_name = "FlickrSeattle_AllPhotos.csv"
    csv_dt = read.csv(paste0(path_predictedtags, csv_name))
    csv_dt$PhotoID = str_extract(csv_dt$Filename, pattern = "(?<=photoid\\_)[0-9]*")
    
    str(csv_dt)
    
    tag_dt_m = as.matrix(csv_dt[,2:11])
    prob_dt_m = as.matrix(csv_dt[,12:21])
   
    # merge flooding and no activity 
    tag_dt_m[tag_dt_m=="flooding"] = "noactivity"
     
    prob_dt2_l = lapply(1:nrow(tag_dt_m), FUN = function(x) sort(tapply(prob_dt_m[x,], INDEX = tag_dt_m[x,], FUN = sum ), decreasing = T))
    
    tag_dt2_m = t(sapply(prob_dt2_l, FUN = function(x) names(x)[1:9]))
    prob_dt2_m = t(sapply(prob_dt2_l, FUN = function(x) (x)[1:9]))
    
    csv_dt_updated = data.frame(Filename = csv_dt$Filename, tag_dt2_m, prob_dt2_m, PhotoID = csv_dt$PhotoID)
    
    colnames(csv_dt_updated)[2:10] = paste0("Top", 1:9)
    
    colnames(csv_dt_updated)[11:19] = paste0("Prob", 1:9)
    
    table(csv_dt$Top1)
    
    table(csv_dt_updated$Top1)
    summary(csv_dt_updated$Prob1)
    
    aoi_coords_merged = merge(aoi_coords, csv_dt_updated, by.all=PhotoID, all=T) # sometimes no tags in CSV? 
    
    warning(nrow(aoi_coords_merged) == nrow(aoi_coords))
    warning(nrow(aoi_coords) == nrow(csv_dt))
    
    aoi_spdf = SpatialPointsDataFrame(coords = aoi_sp, data = aoi_coords_merged)
    
    writeOGR(aoi_spdf, dsn = "Output/", layer = "FlickrMiddlefork_predicted", driver="ESRI Shapefile", overwrite_layer = T)
    
    plot(aoi_spdf, col = aoi_spdf$Top1)
    
    
} else if (doMountainLook) {
    # The pics are from a second area in the MBS forest, in a corridor around the "Mountain Loop Hwy".
    aoi_coords = read.csv("GIS and input data/mbsmtloop_flickrimages_20200213.csv") 

    aoi_sp = SpatialPoints(data.frame(x=aoi_coords$longitude, y = aoi_coords$latitude), proj4string = crs(proj4.LL))
    plot(aoi_sp, add=T)
    names(aoi_coords)[1] = "PhotoID"
    
    ### Read predicted tags 
    path_predictedtags = paste0("TaggedResult_Feb2020_Mountainloop/CSV/")
    csv_name = "Photos.csv"
    csv_dt = read.csv(paste0(path_predictedtags, csv_name))
    csv_dt$PhotoID = str_extract(csv_dt$Filename, pattern = "[0-9]*")
    
    
    
    
    tag_dt_m = as.matrix(csv_dt[,2:11])
    prob_dt_m = as.matrix(csv_dt[,12:21])
    
    # merge flooding and no activity 
    tag_dt_m[tag_dt_m=="flooding"] = "noactivity"
    
    prob_dt2_l = lapply(1:nrow(tag_dt_m), FUN = function(x) sort(tapply(prob_dt_m[x,], INDEX = tag_dt_m[x,], FUN = sum ), decreasing = T))
    
    tag_dt2_m = t(sapply(prob_dt2_l, FUN = function(x) names(x)[1:9]))
    prob_dt2_m = t(sapply(prob_dt2_l, FUN = function(x) (x)[1:9]))
    
    csv_dt_updated = data.frame(Filename = csv_dt$Filename, tag_dt2_m, prob_dt2_m, PhotoID = csv_dt$PhotoID)
    
    colnames(csv_dt_updated)[2:10] = paste0("Top", 1:9)
    
    colnames(csv_dt_updated)[11:19] = paste0("Prob", 1:9)
    
    table(csv_dt$Top1)
    table(csv_dt_updated$Top1)
    summary(csv_dt_updated$Prob1)
    
    
     
    
    aoi.poly.in = rgeos::gConvexHull(aoi_sp)
    plot(aoi.poly.in)
    plot(aoi_sp, add=T)
    
    aoi_coords_merged = merge(aoi_coords, csv_dt_updated, by.all=PhotoID, all=T) # sometimes no tags in CSV? 
    
    warning(nrow(aoi_coords_merged) == nrow(aoi_coords))
    warning(nrow(aoi_coords) == nrow(csv_dt))
    
    aoi_spdf = SpatialPointsDataFrame(coords = aoi_sp, data = aoi_coords_merged)
    
    writeOGR(aoi_spdf, dsn = "Output/", layer = "FlickrMBSMtLoop_predicted", driver="ESRI Shapefile", overwrite = T)
    
    
    plot(aoi_spdf, col = aoi_spdf$Top1)
}


 

aoi_ext = extent(aoi.poly.in)
aoi_ul = c(aoi_ext[c(4, 2)])
aoi_lr = c(aoi_ext[c(3, 1)])

map <- openmap(aoi_ul, lowerRight = aoi_lr, zoom = 12, type="stamen-terrain") # bing
map_rs <- raster(map)
map_rs = projectRaster(map_rs, crs = proj4.LL)
# plotRGB(map_rs)

Top1_YN =  (aoi_spdf$Prob1 > 0.5)

# table(aoi_spdf$Top1)

tag_labels = c("backpacking", "birdwatching", "boating", "camping", "fishing", "flooding", "hiking", "horseriding",
               "mtn_biking", "noactivity", "otheractivities", "pplnoactivity", "rock climbing", "swimming",
               "trailrunning")


## Top1 plot  (opaquness by prob)
top1_fac = factor(aoi_spdf$Top1, levels = tag_labels)
top1_levels = levels(top1_fac)
top1_int = as.numeric(top1_fac)

# n <- 15
# qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
# col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))
# 
# 
# qual_col_pals <- distinctColorPalette(n)
# 
# pie(rep(1,n), col=sample(col_vector, n))
# 
# myCol = c("pink1", "violet", "mediumpurple1", "slateblue1", "purple", "purple3",
#           "turquoise2", "skyblue", "steelblue", "blue2", "navyblue",
#           "orange", "tomato", "coral2", "palevioletred", "violetred", "red2",
#           "springgreen2", "yellowgreen", "palegreen4",
#           "wheat2", "tan", "tan2", "tan3", "brown",
#           "grey70", "grey50", "grey30")



pal_top1 = c(rich.colors(7), brewer.pal(8, "Set3"))#, rainbow(7))
# pie(rep(1,length(pal_top1)), col=pal_top1)

col_top1 = pal_top1[top1_int]
col_top1_YN = sapply(1:length(Top1_YN), FUN = function(x) ifelse(Top1_YN[x], col_top1[x], NA))

col_top1_woNA = sapply(1:length(Top1_YN), FUN = function(x) ifelse(top1_int[x] != 10, col_top1[x], NA))

# col_top1_opaque = scales::alpha("red", alpha = aoi_spdf$Prob1 )
# cex_top1 = 2 * aoi_spdf$Prob1



# survey_shp = readOGR("../FlickrCNN/Seattle/Survey/survey_destinations/survey_destinations.shp") # old 
survey_shp = readOGR("../FlickrCNN/Seattle/Survey/survey_activities_areas/survey_activities_areas.shp") # old
# 

survey_names = levels(survey_shp$AreaNam)
length(unique(as.numeric(survey_shp$AreaNam)))
pal_survey = distinctColorPalette(length(survey_names))
col_survey = pal_survey[as.numeric(survey_shp$AreaNam)]

t_idx= 1 

 pdf(paste0("../DATA2TB_LINK/Flickr15Oct2019/FlickrMiddlefork_Plot_Survey_Top", t_idx, ".pdf"), width = 20, height = 10)

## Top1-3 plot  (opaquness by prob)
tmp_fac = factor(aoi_spdf[[5 + t_idx]], levels = tag_labels)
tmp_int = as.numeric(tmp_fac)
col_tmp = pal_top1[tmp_int]
col_tmp_woNA = sapply(1:length(Top1_YN), FUN = function(x) ifelse(tmp_int[x] != 10, col_tmp[x], NA))



par(mfrow=c(1,2), mar = c(4,4,4,4), oma=c(2,2,2,2))
plot(survey_shp)
# plotRGB(map_rs,  main = paste0("Predicted Top", t_idx, " activity"), add=T)
plot(survey_shp, add=T, col = col_survey)

plot(aoi_spdf, col = col_tmp, pch=tmp_int, add=T, cex=1)
plot(aoi.poly.in, add=T)



plot.new()
legend("top", title =  paste0("Predicted Top", t_idx, " activity"), legend =  top1_levels, col = pal_top1, pch=1:15, ncol=2)
legend("bottom", title =  paste0("Survey locations"), legend =  survey_names, col = pal_survey, pch=15, ncol=2,  cex=0.8)


dev.off()




### weighted map 

## Top1-3 plot  (opaquness by prob)
tmp_fac = factor(aoi_spdf[[5 + t_idx]], levels = tag_labels)
tmp_int = as.numeric(tmp_fac)
col_tmp = pal_top1[tmp_int]
col_tmp_woNA = sapply(1:length(Top1_YN), FUN = function(x) ifelse(tmp_int[x] != 10, col_tmp[x], NA))

map_survey_over_l = over(survey_shp, aoi_spdf, returnList = T)




map_1min = projectRaster(map_rs[[1]], res=1/60, crs = proj4string(map_rs)) # 1 arcminute
map_1min = setValues(map_1min, 1)
map_1min_sp = rasterToPolygons(map_1min)
map_1min_sp$layer.1[] = NA
names(map_1min_sp) = "Intensity"

map_over_l = (over(map_1min_sp, aoi_spdf, returnList = T))





x = map_over_l[[48]]
# library(survey)

.densitymap = function(x) { 
    
    if (is.null(x)) { 
        return(  rep(NA, length(tag_labels)))
    }
    
    if (nrow(x)==0) { 
        return(  rep(NA, length(tag_labels)))
    }
    # print(nrow(x))
    
    tg = (unlist(x[, paste0("Top", 1:10)]))
    wt = unlist(x[, paste0("Prob", 1:10)])
    wt[is.na(wt)] = 0 
    
    df = data.frame(id = 1:length(tg), tg, wt) 
    
    
    
    dclus1 <- svydesign(id=~id, weights=~wt, data=df, tg=~tg)    
    (tbl <- svytable(~tg, dclus1))
    # barplot(tbl)
    # barplot(table(tg))
    
    data.frame((tbl)[tag_labels])[,2]
    
}

library(survey)
Tag = c("A", "A", "A", "B", "B", "C")
Weight = c(0.1, 0.1, 1, 0.5, 0.5, 0.7)
df_tmp = data.frame(id = 1:length(Tag), Tag, Weight)
dclus1 <- svydesign(id=~id, weights=~Weight, data=df_tmp, Tag=~Tag)
Tag_weighted <- svytable(~Tag, dclus1)
Tag_weighted
# 
# 
tapply(Weight, INDEX = Tag, FUN = sum )
# 






map_density = data.frame(t(sapply(map_survey_over_l[], FUN = function(x) .densitymap(x))))
map_density
colnames(map_density) = tag_labels

map_density[map_density==NA] = 0

map_density_spdf = SpatialPolygonsDataFrame(survey_shp, data = map_density )
map_density_spdf_wo_na = map_density_spdf
map_density_spdf_wo_na$noactivity = NULL 
spplot(map_density_spdf, col.regions = rev(topo.colors(30)), col="transparent")
spplot(map_density_spdf_wo_na, col.regions = rev(topo.colors(30)), col="transparent")

writeOGR(map_density_spdf, dsn = ".", layer = "Middlefork_tags_surveyarea_weightedcounts", driver="ESRI Shapefile")

log_map_density = log(map_density+1) 
map_density_spdf_log = SpatialPolygonsDataFrame(survey_shp, data = log_map_density )

spplot(map_density_spdf_log, col.regions = rev(topo.colors(30)), col="transparent")







par(mfrow=c(2,2), mar = c(4,4,4,4), oma=c(2,2,2,2))
plotRGB(map_rs,  main = paste0("Predicted Top", t_idx, " activity"))
plot(aoi_spdf, col = col_tmp, pch=tmp_int, add=T, cex=1)
plot(aoi.poly.in, add=T)
plot.new()
legend("center", title =  paste0("Predicted Top", t_idx, " activity"), legend =  top1_levels, col = pal_top1, pch=1:15, ncol=2)







sitename = "FlickrMiddlefork"
sitename = "FlickrMBSMtLoop"


for (t_idx in 1:3) { 
    
    
    pdf(paste0("../DATA2TB_LINK/Flickr15Oct2019/",sitename,"_Plot_Top", t_idx, ".pdf"), width = 20, height = 20)
    
    ## Top1-3 plot  (opaquness by prob)
    tmp_fac = factor(aoi_spdf[[5 + t_idx]], levels = tag_labels)
    tmp_int = as.numeric(tmp_fac)
    col_tmp = pal_top1[tmp_int]
    col_tmp_woNA = sapply(1:length(Top1_YN), FUN = function(x) ifelse(tmp_int[x] != 10, col_tmp[x], NA))
    
    
    par(mfrow=c(2,2), mar = c(4,4,4,4), oma=c(2,2,2,2))
    plotRGB(map_rs,  main = paste0("Predicted Top", t_idx, " activity"))
    plot(aoi_spdf, col = col_tmp, pch=tmp_int, add=T, cex=1)
    plot(aoi.poly.in, add=T)
    plot.new()
    legend("center", title =  paste0("Predicted Top", t_idx, " activity"), legend =  top1_levels, col = pal_top1, pch=1:15, ncol=2)
    
    # par(mfrow=c(1,2))
    # plotRGB(map_rs)
    plot(aoi_spdf, col = col_tmp, pch=tmp_int, add=F, cex=1,  main =  paste0("Predicted Top", t_idx, " activity"))
    plot(aoi.poly.in, add=T)
    plot.new()
    legend("center", title= paste0("Predicted Top", t_idx, " activity"), legend =  top1_levels, col = pal_top1, pch=1:15, ncol=2)
    
    
    # par(mfrow=c(1,2))
    plotRGB(map_rs,   main = paste0("Predicted Top", t_idx, " activity (w/o no activity)"))
    plot(aoi_spdf, col = col_tmp_woNA, pch=tmp_int, add=T, cex=1)
    plot(aoi.poly.in, add=T)
    plot.new()
    legend("center", legend =  top1_levels, col = pal_top1, pch=1:15, title=paste0("Predicted Top", t_idx, " activity (w/o no activity)"), ncol=2)
    
    # par(mfrow=c(1,2))
    # plotRGB(map_rs)
    plot(aoi_spdf, col = col_tmp_woNA, pch=tmp_int, add=F, cex=1, main =paste0("Predicted Top", t_idx, " activity (w/o no activity)"))
    plot(aoi.poly.in, add=T)
    plot.new()
    legend("center", legend =  top1_levels, col = pal_top1, pch=1:15, title=paste0("Predicted Top", t_idx, " activity (w/o no activity)"), ncol=2)
    
    dev.off()
    
}



## Top1 > 80% > 50% plot 

## Top2 plot 



path_trainingphotos = "../LabelledData/Seattle/Photos_iterative_Sep2019/train/"
path_predicted = "../DATA2TB_LINK/Flickr15Oct2019/"

# list.dirs(path_trainingphotos)
# tag_labels


training_filenames_l = sapply(tag_labels, FUN = function(x) list.files(paste0(path_trainingphotos, "/", x)))

predicted_filenames_l = sapply(tag_labels, FUN = function(x) list.files(paste0(path_predicted, "/", x)))

t_idx = 1 

res = foreach (t_idx = 1:length(tag_labels)) %do% { 
    
    
    tag_arrived = sapply(predicted_filenames_l,FUN = function(x) length(which(x %in% (training_filenames_l[[t_idx]]))))
    
    tag_arrived_perc = tag_arrived / sum(tag_arrived)
    list(tag_arrived, tag_arrived_perc)
}

names( res) = tag_labels

pdf("TrainingPhotosInEachClass.pdf", width = 12, height = 10)
par(mfrow=c(2,2))
for (t_idx in 1:length(tag_labels)) {
    
    barplot(res[[t_idx]][[2]], ylim=c(0, 1), ylab= paste0("% predicted photos "), main= paste0(tag_labels[t_idx], "(n=", sum(res[[t_idx]][[1]]), ")"), las=2)
}

dev.off()

stop("ends here")




 



## number of training samples per class 

classes = list.files("../../LabelledData/Seattle/Photos_iterative_Aug2019/train/")

n_of_sampls = sapply(classes, function(x) length(list.files(paste0("../../LabelledData/Seattle/Photos_iterative_Aug2019/train/", x))))

n_of_sampls = n_of_sampls[ names(prob_avg)]
barplot(n_of_sampls, main = "Number of training images", las=2)
barplot(log(n_of_sampls, 10), main = "Log10 Number of training images", las=2)

# prediction accuracy

perf1 = read.csv("../TrainedWeights/InceptionResnetV2_Seattle_retrain_instabram_16classes_finetuning_iterative_final_val_acc0.88_performance_first?.csv")
perf2 = read.csv("../TrainedWeights/InceptionResnetV2_Seattle_retrain_instabram_16classes_finetuning_iterative_final_val_acc0.88_performance.csv")

model_perf = rbind(perf2, perf1)
model_perf$X = NULL

plot(121:144, model_perf$val_acc, type="l", ylim=c(0.5,1), col = "blue", ylab = "Classification Accuracy", xlab = "Training Epoch")
lines(121:144, model_perf$acc, type="l", col = "red")
legend("bottomright", legend = c("Training acc.", "Validation acc. (70/30)"), col = c("red", "blue"), lty=1, bty="n")
legend("bottomleft", legend = c("N of training samples=5365", "N of validation samples=2329"), lty=0, bty="n")




# barplot(prob_avg[-11])

# plot(as.numeric(tag_top1), prob_avg)

barplot(tag_top1, ylab = "N of photos", las=2, main = "N of photos classified for the activity classes")
barplot(log(tag_top1[], base = 10), ylab = "Log n", las=2,  main = "Log10 N of photos classified for the activity classes")

barplot(colMeans(probs_dt), ylim=c(0,1), main = "Dist. of Prediction uncertainty per rank", las=2 )

boxplot(prob_l, main = "Dist. of Prediction uncertainty per class", las=2 )



 



 




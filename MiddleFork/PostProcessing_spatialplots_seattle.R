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
# 
library(latticeExtra)

library(corrplot)
library(reshape2)

library(survey)


setwd("~/Dropbox/KIT/FlickrEU/deepGreen/")




n.thread <- detectCores() # 1 
proj4.LL <- "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs +towgs84=0,0,0"
proj4.DHDN <- "+proj=tmerc +lat_0=0 +lon_0=12 +k=1 +x_0=4500000 +y_0=0 +ellps=bessel +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs" # epsg:31468

# Login credentials
# api.key <- "" # An API key for Flickr API
# api.secret <- "" # Not used
# apikey_con = file("Flickr_API_KEY.txt", open = "w")
# writeLines(api.key, con = apikey_con)
# close(apikey_con)

apikey_con = file("Flickr_API_KEY.txt", open = "r")
readLines(apikey_con)
close(apikey_con)
# 

workdir = "../FlickrCNN/Seattle/"
gisdir = "."
# save.image(paste0(workdir, savedir, "/Flickr_CR_workspace_metadata_download_17Aug2019.RData"))
# load(paste0(workdir, savedir, "/Flickr_CR_workspace_metadata_download_17Aug2019_2.RData"))


# locations 
path_data = "~/Dropbox/KIT/FlickrEU//"


## Read coordinates 

octPhotos = FALSE 
newPhotos = TRUE

if {octPhotos} { 
    
    
    aoi.poly.in = readOGR( dsn = paste0(workdir, gisdir), layer = "middlefork_AOI")  
    n.points <- length(aoi.poly.in)
    
    plot(aoi.poly.in)
    
    
    aoi_coords = read.csv("../FlickrCNN/Seattle/mbsmiddlefork_flickrimages_20191027.csv")
    aoi_sp = SpatialPoints(data.frame(x=aoi_coords$longitude, y = aoi_coords$latitude), proj4string = crs(proj4.LL))
    plot(aoi_sp, add=T)
    names(aoi_coords)[1] = "PhotoID"
    
    ### Read predicted tags 
    path_predictedtags = paste0("../DATA2TB_LINK/Flickr15Oct2019/")
    csv_name = "FlickrSeattle_AllPhotos.csv"
    csv_dt = read.csv(paste0(path_predictedtags, csv_name))
    csv_dt$PhotoID = str_extract(csv_dt$Filename, pattern = "(?<=photoid\\_)[0-9]*")
    
} else if (newPhotos) {
    # The pics are from a second area in the MBS forest, in a corridor around the "Mountain Loop Hwy".
    aoi_coords = read.csv("../FlickrCNN/Seattle/mbsmtloop_flickrimages_20200213.csv") 

    aoi_sp = SpatialPoints(data.frame(x=aoi_coords$longitude, y = aoi_coords$latitude), proj4string = crs(proj4.LL))
    plot(aoi_sp, add=T)
    names(aoi_coords)[1] = "PhotoID"
    
    ### Read predicted tags 
    path_predictedtags = paste0("../DATA2TB_LINK/FlickrSeattle_Tagging_Feb2020/CSV/")
    csv_name = "FlickrSeattle_NewPhotos.csv"
    csv_dt = read.csv(paste0(path_predictedtags, csv_name))
    csv_dt$PhotoID = str_extract(csv_dt$Filename, pattern = "[0-9]*")
    
    aoi.poly.in = rgeos::gConvexHull(aoi_sp)
    plot(aoi.poly.in)
    plot(aoi_sp, add=T)
}


aoi_coords_merged = merge(aoi_coords, csv_dt, by.all=PhotoID, all=T) # sometimes no tags in CSV? 

warning(nrow(aoi_coords_merged) == nrow(aoi_coords))
warning(nrow(aoi_coords) == nrow(csv_dt))

aoi_spdf = SpatialPointsDataFrame(coords = aoi_sp, data = aoi_coords_merged)

# writeOGR(aoi_spdf, dsn = "../DATA2TB_LINK/FlickrSeattle_Tagging_Feb2020/", layer = "FlickrMBSMtLoop_predicted", driver="ESRI Shapefile")

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






## Reorganise 
aoi.poly.in$NAME_1
head(aoi.poly.in)
aoi.poly.in$NAME  # natural park name

costarica_natpark =  readOGR(paste0(path_data, "GIS data/NatParks_CR/"), layer = "NatParks_CR", verbose = T)

costarica_natpark

## AOI csv 
## predicted tags

aois.done.newnames <- list.files(paste0(workdir, "/", savedir, "/Xlsx/"), pattern = "^AOI.*.\\.xlsx$", full.names = T)

aoi_idx = 237 

path_regrouped = "~/Dropbox/KIT/FlickrEU/Costa Rica_Data/Regrouped/"
path_originalphotos= "/DATA2TB/FlickrCR_download/Aug2019_V1_Photo/"

for (aoi_idx in 1:length(aois.done.newnames)){ 
    aoi.dt = read.xlsx(aois.done.newnames[aoi_idx], 1)
    if (nrow(aoi.dt)==0) { 
        next()   
    }
    
    aoi_cellid = str_extract(aois.done.newnames[aoi_idx], pattern = "(?<=CellID\\_)[0-9]*")
    aoi_sp = SpatialPoints(cbind(as.numeric(aoi.dt$Longitude), as.numeric(aoi.dt$Latitude)), proj4string = CRS(proj4.LL))
    
    
    photos_overlap = over(aoi_sp, costarica_natpark[,"NAME"])
    n_overlap = length(photos_overlap[!is.na(photos_overlap)])
    
    if (n_overlap > 0) { 
        print(paste0(aoi_idx, ": ", n_overlap))   
        # print( table(photos_overlap))
        # photos_overlap[!is.na(photos_overlap)]
        
        natparks = unique(as.character(unlist(photos_overlap)))
        natparks = natparks[!is.na(natparks)]
        
        
        for (np_idx in 1:length(natparks)) { 
            natpark = natparks[np_idx] 
            row_ids = which(photos_overlap$NAME == natpark)
            
            np_dt = aoi.dt[row_ids,]
            unique_ids =  unique(np_dt$PhotoID)
            unique_idxs = match(np_dt$PhotoID, unique_ids) # returns only the first matchs 
            np_dt = np_dt[unique_idxs,] 
            
            
            path_np = paste0(path_regrouped, natpark)
            path_np_photos = paste0(path_np, "/Photos/")
            path_np_xlsx = paste0(path_np, "/Xlsx/")
            path_np_shp = paste0(path_np, "/Shp/")
            
            if (!dir.exists(path_np)) { 
                dir.create(path_np_photos, recursive = T)
                dir.create(path_np_xlsx, recursive = T)
                dir.create(path_np_shp, recursive = T)
                
            }
            # copy xlsx 
            write.xlsx(np_dt, file = paste0(path_np_xlsx, "/AOI_NatPark_", natpark, "_CellID_", aoi_cellid, "_n", length(unique_idxs), ".xlsx"))                                   
            # write shp file 
            np_spdf = SpatialPointsDataFrame(coords = aoi_sp[row_ids[unique_idxs]], data =np_dt) 
            writeOGR(np_spdf, dsn = paste0(path_np_shp), layer = paste0("AOI_NatPark_", natpark, "_CellID_", aoi_cellid, "_n", length(unique_idxs), "_points"), driver = "ESRI Shapefile", overwrite_layer = T)                                   
            
            
            
            # copy photos 
            original_path = list.files(path_originalphotos, pattern = as.character(aoi_cellid ))
            stopifnot(length(original_path) == 1)  
            
            for (p_idx in 1:nrow(np_dt)) {
                p_dt = np_dt[p_idx,] 
                # print(p_idx)
                p_filename = list.files(paste0(path_originalphotos, "/", original_path, "/", p_dt$Year), pattern=p_dt$PhotoID, full.names = T)
                p_filename_short = list.files(paste0(path_originalphotos, "/", original_path, "/", p_dt$Year), pattern=p_dt$PhotoID, full.names = F)
                
                path_target = paste0(path_np_photos, "/", p_dt$Year) # "/", p_filename_short)
                if (!dir.exists(path_target)) { 
                    dir.create(path_target, recursive = F)
                }
                
                file.copy(p_filename, to = paste0(path_target, "/", p_filename_short) )
                
            }
            
        }
        
        
        
    }
}



# aoi.idx <- 1
## predicted tags
csv_files = list.files("../Costa Rica_Data/CSV", pattern = "\\.csv$")


aois.done.v <- (sapply(csv_files, FUN = function(x) (str_split(x, pattern = "_")[[1]][2])))
aois.done.v = sapply(str_split(aois.done.v, pattern = "CellID"), FUN = function(x) as.numeric(x[[2]]))

res2 = foreach(csv_idx = 1:length(csv_files)) %do% { 
    
    
    dt = read.csv(paste0("../Costa Rica_Data/CSV/",csv_files[csv_idx]))
    # str(dt)
    
    tags_dt = dt[,2:11]
    probs_dt = dt[,12:21]
    
    # length(table(unlist(tags_dt)))
    
    dt_df = data.frame(unlist(tags_dt), unlist(probs_dt))
    colnames(dt_df) = c("Tag", "Prob")
    
    res = tapply(dt_df$Prob, INDEX = dt_df$Tag, FUN = sum, na.rm=T)
    return(res)
}

names(res2) = aois.done.v

par(mfrow=c(1,1), mar=c(10,4,4,4))
barplot(sort(table(unlist(sapply(res2, FUN = names))), T), las=2)



v1 = (unlist(sapply(res2, names)))
v2 = (unlist(sapply(res2, c)))

v3 = data.frame(v1, v2)

weightedMean = tapply(v2, INDEX = v1, FUN = mean, na.rm=T)
barplot(sort(weightedMean, T), las=2, ylab= "Weighted prob")

predictedTags = sort(unique(v1))

### spdf 
spdf = data.frame(Tag = predictedTags, Prob = NA)
t_idx = 1 


for (t_idx in 21:length(predictedTags)) { 
    tag = predictedTags[t_idx]
    
    prob = sapply(res2, FUN = function(x) x[names(x)==tag])
    
    aoi.poly.out = aoi.poly.in
    
    aoi.poly.out$tmp = NA
    aoi.poly.out$tmp[ match(names(prob), aoi.poly.out$CELL_ID)] = as.numeric(prob)
    
    
    pdf(paste0(tag, "_dist.pdf"), width = 12, height = 15)
    
    print(spplot(aoi.poly.out, "tmp", main = tag))
    dev.off()
}
# dev.off()



tag_top1 = table(tags_dt$Top1)

levelplot(cor(probs_dt))


corrplot(cor(probs_dt), type = "lower", )

tags_v = unlist(tags_dt[,1])
probs_v = unlist(probs_dt[, 1])


probs_all = matrix(NA, nrow =nrow(tags_dt), ncol = 16)

colnames(probs_all) = names(table(unlist(tags_dt)))




cnames = names(table(unlist(tags_dt)))
# for (i in 1:16) { 
#     
#     
#     probs_all[,cnames[i]] = 
# }
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






































# Time span
mindate <- "2005-01-01"
maxdate <- "2019-07-31"
# savedir <- substr(mindate, 6, 10)




# todos 
# 1 read bayern polygon 
# 2 select bayern areas from the climsave polygons 
# 3 export the pixel ids 
# 4 copy those pixels' xls and jpg files 

n.thread <- detectCores() # 1 
proj4.LL <- "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs +towgs84=0,0,0"
proj4.DHDN <- "+proj=tmerc +lat_0=0 +lon_0=12 +k=1 +x_0=4500000 +y_0=0 +ellps=bessel +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs" # epsg:31468

proj4.etrs_laea <- "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs";
# proj4.EUR_ETRS89_LAEA1052 <- "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs" # EPSG:3035
# 
# # corine2012 <- raster("../GIS/Corine/CLC2012/g250_clc12_V18_5a/g250_clc12_V18_5.tif", 1)
# corine2012 <- raster("../GIS/Corine/CLC2012/g100_clc12_V18_5a/g100_clc12_V18_5.tif")
# proj4string(corine2012) <- proj4.etrs_laea
# # corine2012.ll <- projectRaster(corine2012, crs = proj4.LL, method = "ngb")
# 
# 
# corine.tb <- read.csv("../GIS/Corine/CLC2012/g100_clc12_V18_5a/Legend/clc_legend.csv", sep = ";")
# corine.tb$CLC_CODE
# 
# corine.detail.tb <- read.xlsx("../GIS/Corine/clc_legend.xlsx", 1)
# 

# # Login credentials
# apikey_con = file("Flickr_API_KEY.txt", open = "r")
# readLines(apikey_con)
# close(apikey_con)
# 
# climsave <- readOGR("../GIS", layer="CLIMSAVE_EuropeRegions_for_SA")
# climsave.ll <- spTransform(climsave, CRSobj = proj4.LL)
# 

# ctry.ids<-read.csv("~/Dropbox/KIT/CLIMSAVE/IAP/Cell_ID_LatLong.csv")
# nums<-unique(ctry.ids$Nuts0_ID)
# ctrys<-unique(ctry.ids$Dominant_NUTS0)

# ctry.data<-read.csv("M:/CLIMSAVE Uncertainty/Batch files results/WatW/Index sds by country.csv")
# fpcvals<-ctry.data$fpc.sd
# ludvals<-ctry.data$lud.sd
# iivals<-ctry.data$ii.sd
# pfvals<-ctry.data$pf.sd
# wevals<-ctry.data$we.sd
# bvvals<-ctry.data$bv.sd

# countries<-c("AT","BE","BG","HR","CY","CZ","DE","DK","EE","ES","FI","FR","EL",
# "HU","IE","IT","LT","LU","LV","MT","NL","PL","PT","RO","SE","SI","SK","UK")
# table(ctry.ids$Dominant_NUTS2)
# table(ctry.ids$ClimateDescription)


# climsave.pos.idx <- match( climsave.ll$Cell_ID, ctry.ids$Cell_ID)
# 
# # plot(climsave.ll$Cell_ID, ctry.ids$Cell_ID[climsave.pos.idx])
# 
# climsave.ll$Dominant_NUTS2 <- ctry.ids$Dominant_NUTS2[climsave.pos.idx] 
# climsave.ll$ClimateDescription <- ctry.ids$ClimateDescription[climsave.pos.idx] 
# 
# # spplot(climsave.ll, "Dominant_NUTS2")
# # spplot(climsave.ll, "ClimateDescription")
# 
# aoi.countrycode <- climsave.ll$Dominant_NUTS2 
# aoi.climate <- climsave.ll$ClimateDescription 


# aoi.poly.in <- climsave.ll
# rm(climsave, climsave.ll)

# 
# # Time span
# mindate <- "2005-01-01"
# maxdate <- "2017-12-31"
# # savedir <- substr(mindate, 6, 10)
# savedir <- "May2018_V1/"
# workdir <- "FlickrEU_download/"
# newsavedir <- paste0("May2018_V1_LC_reduced")
# shpdir <- paste0("May2018_V1_SHP")

# 
# 
# # Search parameters
# sort <- "interestingness-desc" # Sort by Interestingness (or: relevance)
# max.perpage <- 250 # number per page maximum 250
# n.points <- length(aoi.poly.in)
# 


# Retreiving the data


# target.ids.all <- 1:n.points
target.ids.all <- 1:n.points

aois.done <- list.files(paste0(workdir, "/", savedir, "/Xls"), pattern = "^AOI.*.\\.xlsx$")
aois.done.v <- as.numeric(sapply(aois.done, FUN = function(x)  (str_split(x, pattern = "_")[[1]][3])))


nphotos.done.v <- as.numeric(sapply(aois.done, FUN = function(x)  str_extract(str_split(x, pattern = "_")[[1]][6], "[0-9]+")))
nphotos.done.v2 <- as.numeric(sapply(aois.done, FUN = function(x)  str_extract(str_split(x, pattern = "_")[[1]][8], "[0-9]+")))

nphotos.done.v[is.na(nphotos.done.v)] = nphotos.done.v2[is.na(nphotos.done.v)]
rm(nphotos.done.v2)

hist(nphotos.done.v)


nphotos.done.reduced.v <- numeric(length = n.points) 

# 14297, 14732, 15044

for (aoi.idx in 15045:n.points) { 
    cat(aoi.idx)
    aoi.tmp <- aois.done[aoi.idx]
    # Sys.setlocale("UTF-8")
    tryCatch(
        aoi.dt.raw <- data.frame(read.xlsx(paste0(workdir, savedir, aoi.tmp),sheet = 1)), 
        
    )
    # tryCatch(stop("fred"),  error = function(e) e, finally = print("Hello"))
    # withCallingHandlers({ warning("A"); 1+2 }, warning = function(w) {})
    
    aoi.dt.raw <- data.frame(read_excel(paste0(workdir, savedir, aoi.tmp),sheet = 1))
    
    photo.ids.unique <- as.numeric( unique( aoi.dt.raw$PhotoID)  )
    # aoi.dt <- aoi.dt.raw[match(photo.ids.unique, aoi.dt.raw$PhotoID ),]
    # nr <- max(0, nrow(aoi.dt))
    
    nphotos.done.reduced.v[aoi.idx] <- max(0, length(photo.ids.unique))
}

summary(nphotos.done.reduced.v)


cntrys.done.v <-  str_sub(sapply(aois.done, FUN = function(x)  (str_split(x, pattern = "_")[[1]][5])), 1,2)
climate.done.v <-  str_sub(sapply(aois.done, FUN = function(x)  (str_split(x, pattern = "_")[[1]][6])))


boxplot(nphotos.done.v)
sum(nphotos.done.v)* 1E-6

cntrys.freq <- tapply(X = nphotos.done.v, INDEX = cntrys.done.v, FUN = sum, na.rm=T)
barplot(cntrys.freq, las=2)

cntrys.v <- tapply(X = nphotos.done.v, INDEX = cntrys.done.v, FUN = c, na.rm=T)

boxplot(cntrys.v, las=2)


climate.v <- tapply(X = nphotos.done.v, INDEX = climate.done.v, FUN = c, na.rm=T)

boxplot(climate.v, las=2)# scale=log)





climsave.ll$NPHOTO <-  nphotos.done.v 


# col.nphoto <- log(climsave.ll$NPHOTO + 1 )

# plot(climsave.ll) # , col = col.nphoto)

writeOGR(climsave.ll, dsn = "Data", layer = "FlickrEU_Nphoto", verbose = T, overwrite_layer = T, driver = "ESRI Shapefile")




#### Write SHP files 
aois.done.fullnames <- list.files(paste0(workdir, "/", savedir), pattern = "^AOI.*.\\.xlsx$", full.names = T)
aois.done.newnames <- paste0(workdir, "/", newsavedir, "/", sapply(str_split(aois.done, pattern = "\\.xlsx"), FUN = function(x) x[[1]]), "_LC.xlsx")
aois.done.shpnames.short <- paste0( sapply(str_split(aois.done, pattern = "\\.xlsx"), FUN = function(x) x[[1]]), "_LC.shp")

aoi.idx <- 3242

registerDoMC()

reprocess <- F 
if (reprocess) { 
    
    aoi.meta.res <- foreach (aoi.idx = 1:n.points, .inorder = T, .errorhandling = "stop", .verbose = F, .combine = rbind) %do% { 
        
        print(aoi.idx)
        aoi.tmp <- aois.done.newnames[aoi.idx]
        # Sys.setlocale("UTF-8")
        aoi.dt.raw <- data.frame(read_excel(aoi.tmp,sheet = 1))
        photo.ids.unique <- as.numeric( unique( aoi.dt.raw$PhotoID)  )
        aoi.dt <- aoi.dt.raw[match(photo.ids.unique, aoi.dt.raw$PhotoID ),]
        # table(aoi.dt$PhotoID == 16469798504)
        # nrow(aoi.dt)
        
        
        # numeric.cols <- c( "Year", "Longitude", "Latitude", "Geocontext", "LocationAccuracy", "N_FlickrTag")
        # aoi.dt[, numeric.cols] <- sapply(numeric.cols, FUN = function(x) as.numeric(aoi.dt[,x]))
        # str(aoi.dt)
        
        return(list(Landcover=table(aoi.dt$Landcover), Year = table(aoi.dt$Year)))
    }
    # str(aoi.meta.res)
    
    saveRDS(aoi.meta.res, file = "Data/AOI_META_RES_reduced.Rds")
} else {
    aoi.meta.res <- readRDS(file = "Data/AOI_META_RES_reduced.Rds")
}




# GRID_CODE	CLC_CODE	LABEL1	LABEL2	LABEL3	RGB
# 1	111	Artificial surfaces	Urban fabric	Continuous urban fabric	230-000-077
# 2	112	Artificial surfaces	Urban fabric	Discontinuous urban fabric	255-000-000
# 3	121	Artificial surfaces	Industrial, commercial and transport units	Industrial or commercial units	204-077-242
# 4	122	Artificial surfaces	Industrial, commercial and transport units	Road and rail networks and associated land	204-000-000
# 5	123	Artificial surfaces	Industrial, commercial and transport units	Port areas	230-204-204
# 6	124	Artificial surfaces	Industrial, commercial and transport units	Airports	230-204-230
# 7	131	Artificial surfaces	Mine, dump and construction sites	Mineral extraction sites	166-000-204
# 8	132	Artificial surfaces	Mine, dump and construction sites	Dump sites	166-077-000
# 9	133	Artificial surfaces	Mine, dump and construction sites	Construction sites	255-077-255
# 10	141	Artificial surfaces	Artificial, non-agricultural vegetated areas	Green urban areas	255-166-255
# 11	142	Artificial surfaces	Artificial, non-agricultural vegetated areas	Sport and leisure facilities	255-230-255
# 12	211	Agricultural areas	Arable land	Non-irrigated arable land	255-255-168
# 13	212	Agricultural areas	Arable land	Permanently irrigated land	255-255-000
# 14	213	Agricultural areas	Arable land	Rice fields	230-230-000
# 15	221	Agricultural areas	Permanent crops	Vineyards	230-128-000
# 16	222	Agricultural areas	Permanent crops	Fruit trees and berry plantations	242-166-077
# 17	223	Agricultural areas	Permanent crops	Olive groves	230-166-000
# 18	231	Agricultural areas	Pastures	Pastures	230-230-077
# 19	241	Agricultural areas	Heterogeneous agricultural areas	Annual crops associated with permanent crops	255-230-166
# 20	242	Agricultural areas	Heterogeneous agricultural areas	Complex cultivation patterns	255-230-077
# 21	243	Agricultural areas	Heterogeneous agricultural areas	Land principally occupied by agriculture, with significant areas of natural vegetation	230-204-077
# 22	244	Agricultural areas	Heterogeneous agricultural areas	Agro-forestry areas	242-204-166
# 23	311	Forest and semi natural areas	Forests	Broad-leaved forest	128-255-000
# 24	312	Forest and semi natural areas	Forests	Coniferous forest	000-166-000
# 25	313	Forest and semi natural areas	Forests	Mixed forest	077-255-000
# 26	321	Forest and semi natural areas	Scrub and/or herbaceous vegetation associations	Natural grasslands	204-242-077
# 27	322	Forest and semi natural areas	Scrub and/or herbaceous vegetation associations	Moors and heathland	166-255-128
# 28	323	Forest and semi natural areas	Scrub and/or herbaceous vegetation associations	Sclerophyllous vegetation	166-230-077
# 29	324	Forest and semi natural areas	Scrub and/or herbaceous vegetation associations	Transitional woodland-shrub	166-242-000
# 30	331	Forest and semi natural areas	Open spaces with little or no vegetation	Beaches, dunes, sands	230-230-230
# 31	332	Forest and semi natural areas	Open spaces with little or no vegetation	Bare rocks	204-204-204
# 32	333	Forest and semi natural areas	Open spaces with little or no vegetation	Sparsely vegetated areas	204-255-204
# 33	334	Forest and semi natural areas	Open spaces with little or no vegetation	Burnt areas	000-000-000
# 34	335	Forest and semi natural areas	Open spaces with little or no vegetation	Glaciers and perpetual snow	166-230-204
# 35	411	Wetlands	Inland wetlands	Inland marshes	166-166-255
# 36	412	Wetlands	Inland wetlands	Peat bogs	077-077-255
# 37	421	Wetlands	Maritime wetlands	Salt marshes	204-204-255
# 38	422	Wetlands	Maritime wetlands	Salines	230-230-255
# 39	423	Wetlands	Maritime wetlands	Intertidal flats	166-166-230
# 40	511	Water bodies	Inland waters	Water courses	000-204-242
# 41	512	Water bodies	Inland waters	Water bodies	128-242-230
# 42	521	Water bodies	Marine waters	Coastal lagoons	000-255-166
# 43	522	Water bodies	Marine waters	Estuaries	166-255-230
# 44	523	Water bodies	Marine waters	Sea and ocean	230-242-255
# 48	999	NODATA	NODATA	NODATA	
# 49	990	UNCLASSIFIED	UNCLASSIFIED LAND SURFACE	UNCLASSIFIED LAND SURFACE	
# 50	995	UNCLASSIFIED	UNCLASSIFIED WATER BODIES	UNCLASSIFIED WATER BODIES	230-242-255
# 255	990	UNCLASSIFIED	UNCLASSIFIED	UNCLASSIFIED	



clc.codes <- unique(names(table(unlist(sapply(aoi.meta.res[,1], FUN = function(x) names(x))))))
clc.codes <- as.numeric(clc.codes)
clc.codes <- clc.codes[!is.na(clc.codes)]


dummy.v  <- numeric(length = length(clc.codes))
names(dummy.v) <- clc.codes

for (idx in 1:n.points) { 
    
    v1 <- (aoi.meta.res[[idx, 1]])
    dummy.v[names(v1)] = dummy.v[names(v1)] + v1
}

dummy.v <- dummy.v[!is.na(dummy.v)]

barplot(dummy.v)


dummy.v.clclabel1 <- ((corine.detail.tb$LABEL1[match( names(dummy.v), as.numeric(corine.detail.tb$CLC_CODE) )]))
dummy.v.clclabel2 <- ((corine.detail.tb$LABEL2[match( names(dummy.v), as.numeric(corine.detail.tb$CLC_CODE) )]))
dummy.v.clclabel3 <- ((corine.detail.tb$LABEL3[match( names(dummy.v), as.numeric(corine.detail.tb$CLC_CODE) )]))


clclabel2.col <- factor(dummy.v.clclabel2)

res.lc <- data.frame(dummy.v.clclabel1, dummy.v.clclabel2, dummy.v.clclabel3, Nphoto= dummy.v)

res.lc.label1 <- tapply(res.lc$Nphoto, INDEX = res.lc$dummy.v.clclabel1, FUN = sum, na.rm=T)
res.lc.label2 <- tapply(res.lc$Nphoto, INDEX = res.lc$dummy.v.clclabel2, FUN = sum, na.rm=T)
res.lc.label3 <- tapply(res.lc$Nphoto, INDEX = res.lc$dummy.v.clclabel3, FUN = sum, na.rm=T)



barplot(res.lc.label1, las=2, ylab="Nphoto", main="Corine Land Cover (Label 1)")


pdf("Figures/Flickr_CLC_Label1.pdf", width = 12, height = 9) 
par(oma=c(8,1,1,1), mar=c(4,4,4,4))
barplot(res.lc.label1, las=2, ylab="Nphoto", main="Corine Land Cover (Label 1)")
dev.off()
pdf("Figures/Flickr_CLC_Label2.pdf", width = 12, height = 9) 
par(oma=c(10,5,1,1), mar=c(10,5,4,4))
barplot(res.lc.label2, las=2,  main="Corine Land Cover (Label 2)")
dev.off()
pdf("Figures/Flickr_CLC_Label3.pdf", width = 16, height = 12) 
par(oma=c(1,20,1,1), mar=c(5,12,4,4))
barplot(res.lc.label3, las=1,  main="Number of Flickr photos by land cover (CLC Label 3)", xlab="Nphoto", , horiz=T)
dev.off()


flickr.years <- 2005:2017 

dummy.v2  <- numeric(length = length(flickr.years))
names(dummy.v2) <- flickr.years

for (idx in 1:n.points) { 
    
    v1 <- (aoi.meta.res[[idx, 2]])
    dummy.v2[names(v1)] = dummy.v2[names(v1)] + v1
}

dummy.v2 <- dummy.v2[!is.na(dummy.v2)]

barplot(dummy.v2)
pdf("Figures/Flickr_Year.pdf", width = 12, height = 8) 
par(oma=c(1,1,1,1), mar=c(5,5,4,4))
barplot(dummy.v2, las=1,  main="Number of Flickr photos by year", xlab="Year", horiz=F, ylim=c(0, 2.5E7))
dev.off()


res.years <- sapply(aoi.meta.res[,2], FUN = function(x) {dummy.v[names(x)] <- x; return(dummy.v)})

do.call(rbind, res.years)

table(sapply(res.years, FUN = length))

str(res.years, 1)

names(aoi.meta.res[[1,2]])

























stop("ends here") 
### Write with LC

aois.done <- list.files(paste0(workdir, "/", savedir), pattern = "^AOI.*.\\.xlsx$")
aois.done.v <- as.numeric(sapply(aois.done, FUN = function(x) (str_split(x, pattern = "_")[[1]][3])))

aois.new.done <- list.files(paste0(workdir, "/", shpdir), pattern = "^AOI.*.\\.shp$")
aois.new.done.v <- as.numeric(sapply(aois.new.done, FUN = function(x) (str_split(x, pattern = "_")[[1]][3])))

target.ids <-    (setdiff(aois.done.v, aois.new.done.v))
length(aois.done.v)  - length(target.ids) 
cat(length(target.ids), "to go")

# registerDoMC()

foreach (aoi.idx = target.ids, .inorder = F, .errorhandling = "stop", .verbose = T) %do% { 
    
    print(aoi.idx)
    
    # aois.new.done <- list.files(paste0(workdir, "/", newsavedir), pattern = "^AOI.*.\\.xlsx$")
    # aois.new.done.v <- as.numeric(sapply(aois.new.done, FUN = function(x) (str_split(x, pattern = "_")[[1]][3])))
    # 
    if (aoi.idx %in% aois.new.done.v) {
        print("skip")
        
        return(F)
    }
    
    print("process")
    aoi.tmp <- aois.done.fullnames[aoi.idx]
    Sys.setlocale(, "UTF-8")
    
    # aoi.dt <- read.xlsx(aoi.tmp, 1, detectDates = F)
    aoi.dt.raw <- data.frame(read_excel(aoi.tmp,sheet = 1))
    photo.ids.unique <- as.numeric( unique( aoi.dt.raw$PhotoID)  )
    aoi.dt <- aoi.dt.raw[match(photo.ids.unique, aoi.dt.raw$PhotoID ),]
    # table(aoi.dt$PhotoID == 16469798504)
    # nrow(aoi.dt)
    
    aoi.dt$Title <- iconv(aoi.dt$Title, from="UTF-8", to="ASCII", sub="")
    aoi.dt$Username <- iconv(aoi.dt$Username, from="UTF-8", to="ASCII", sub="") # Special characters.. be careful as we flattend the UTF-8 usernames to ASCII
    
    if (nrow(aoi.dt)==0) {   
        write.xlsx(data.frame(NA), aois.done.newnames[aoi.idx], overwrite=T)
        
        return(F)
    }
    numeric.cols <- c( "Year", "Longitude", "Latitude", "Geocontext", "LocationAccuracy", "N_FlickrTag")
    
    aoi.dt[, numeric.cols] <- sapply(numeric.cols, FUN = function(x) as.numeric(aoi.dt[,x]))
    # str(aoi.dt)
    
    aoi.sp <- SpatialPoints(coords = cbind(x= aoi.dt$Longitude, y = aoi.dt$Latitude), proj4string = crs( proj4.LL))
    aoi.sp.etrs <- spTransform(aoi.sp, CRSobj = proj4.etrs_laea)
    
    aoi.lc <- extract(corine2012, aoi.sp.etrs, fn = c)
    stopifnot(length(aoi.lc)== length(aoi.sp))
    
    aoi.clc_code <- corine.tb$CLC_CODE[ match(aoi.lc,  corine.tb$GRID_CODE)]
    aoi.clc_lebel3 <- corine.tb$LABEL3[ match(aoi.lc,  corine.tb$GRID_CODE)]
    
    aoi.dt$Landcover <- aoi.clc_code
    aoi.dt$LandcoverDesc <- aoi.clc_lebel3
    
    write.xlsx(aoi.dt, file = aois.done.newnames[aoi.idx], overwrite=T)
    
    # read.xlsx(aois.done.newnames[aoi.idx])
    
    aoi.spdf.etrs <- SpatialPointsDataFrame(aoi.sp.etrs, data = aoi.dt)
    writeOGR(aoi.spdf.etrs, dsn = paste0(workdir, "/", shpdir), layer = aois.done.shpnames.short[aoi.idx], driver = "ESRI Shapefile", verbose = T, overwrite_layer = T, encoding = "UTF-8")
    return(T)
}






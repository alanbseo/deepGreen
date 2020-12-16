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
 
workdir = "../FlickrCNN/Seattle/"
gisdir = "."
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

apikey_con = file("Flickr_API_KEY.txt", open = "r")
readLines(apikey_con)
close(apikey_con)
# 




## Read coordinates 

octPhotos = FALSE 
newPhotos = TRUE

if (octPhotos) { 
    
    
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



 



 




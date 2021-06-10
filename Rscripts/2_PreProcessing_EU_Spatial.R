library(doMC)
library(stringr)
library(readxl)
library(openxlsx)



proj4.LL <- "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs +towgs84=0,0,0"



# X570
# path_base = "~/Dropbox/KIT/FlickrEU/deepGreen/"
# path_out_csv = "~/Downloads/FlickrEU_result/MergedCSV"
# path_EU2018_xls = "../FlickrEU_download/May2018_V1/May2018_V1_LC/"
# 
# xls2018_v = list.files(path_EU2018_xls, pattern=".xlsx", full.names = T)
# 
# n_thread = 32


# Keal
path_base = "~/pd/deepGreen/"
path_out_csv = "~/pd/FlickrEU_result/MergedCSV"
path_out_network = "~/pd/FlickrEU_result/Network"

path_EU2018_xls = "~/pd/FlickrEU_DOWNLOAD_14May2018/May2018_V1_LC/"

xls2018_v = list.files(path_EU2018_xls, pattern="\\.xlsx$", full.names = T)


n_thread = detectCores()


places365_tags_raw = read.table("Data/categories_places365.txt")
places365_tags_raw$V2 = NULL
str(places365_tags_raw)

places365_tags_raw = as.character(places365_tags_raw$V1)


places365_tags = str_extract(places365_tags_raw, pattern = "(?<=[a-z]/).*")



imgnet1000_tags_verbose = jsonlite::fromJSON("Data/imagenet_class_index.json", simplifyDataFrame = T)

imgnet1000_tags_df = data.frame(do.call(rbind, imgnet1000_tags_verbose))
imgnet1000_tags_df$X1 = NULL
imgnet1000_tags = as.vector(imgnet1000_tags_df$X2)
imgnet1000_tags = unique(imgnet1000_tags)

setwd(path_base)

if (!dir.exists(path_out_csv)) { 
    dir.create(path_out_csv, recursive = T)
}

if (!dir.exists(path_out_network)) { 
    dir.create(path_out_network, recursive = T)
}

library(abind)


library(doMC)

registerDoMC(n_thread)


n_points = 23871

# idxs = sample(10000:20000)
idxs = 1:n_points


# idxs = 19053


if (FALSE) { 
    
    
    nphotos_places = 0 
    nphotos_imgnet = 0 
    nphotos = 0 
    
    # Number of photos
    
    nphotos_l = foreach (i = idxs, .errorhandling = "stop") %dopar% {
        # for (i in idxs) {
        
        if ((i %% 100 ) ==0) {
            cat("AOI_", i, ">")
        }
        in_filename_tmp = paste0(path_out_csv, "/", basename(xls2018_v[i])) 
        
        
        if (!file.exists(in_filename_tmp)) {
            next
        }
        
        
        aoi_tmp = str_extract(xls2018_v[i], pattern = "Poly\\_[0-9]*")
        
        aoi_tmp = paste0("AOI_", as.numeric(substr(aoi_tmp, start = 6, 11)))
        
        
        dt = read.xlsx(in_filename_tmp)
        
        # places 
        places_df = dt[, paste0("Places365_Top", 1:10)] 
        places_df[places_df==""] = NA
        
        # places_prob_df = dt[, paste0("Places365_Prob", 1:10)] 
        # places_prob_df[places_prob_df==""] = NA
        # 
        # Imagenet 
        imgnet_df = dt[, paste0("IMGNET_Top", 1:10)] 
        imgnet_df[imgnet_df==""] = NA
        
        # imgnet_prob_df = dt[, paste0("IMGNET_Prob", 1:10)] 
        # imgnet_prob_df[imgnet_prob_df==""] = NA
        
        
        places_df = places_df[!is.na(places_df[,1]),] 
        imgnet_df = imgnet_df[!is.na(imgnet_df[,1]),] 
        
        nphotos_places = nphotos_places + nrow(places_df)
        nphotos_imgnet = nphotos_imgnet + nrow(imgnet_df)
        
        nphotos = nphotos + nrow(dt)
        
        return(nphotos, nphotos_places, nphotos_imgnet)
    }
    
    
    
    
    
    places_frac = foreach (i = 1:length(xls2018_v)) %dopar% {
        
        
        
        aoi_tmp = str_extract(xls2018_v[i], pattern = "Poly\\_[0-9]*")
        
        aoi_tmp = paste0("AOI_", as.numeric(substr(aoi_tmp, start = 6, 11)))
        
        pl_name = paste0(path_out_network, "/places_m_", aoi_tmp, ".csv")
        
        print(pl_name)
        
        places_all_m <- data.frame(matrix(data = 0, nrow = length(places365_tags), ncol = length(places365_tags)))
        dimnames(places_all_m) <- list( places365_tags, places365_tags)
        
        if (file.exists(pl_name)) { 
            
            places_1mode = read.csv(file = pl_name)
            places_1mode$X = NULL
            
            colnames(places_1mode) = str_replace(colnames(places_1mode), pattern = "\\.", replacement = "/")
            
            rownames(places_1mode) = colnames(places_1mode)
            places_all_m[rownames(places_1mode ), colnames(places_1mode)] = places_all_m[rownames(places_1mode ), colnames(places_1mode)] + places_1mode
            
        } else {
            # do nothing
        }
        
        return(colSums(places_all_m))
        
    }
    
    
    # places_frac = readRDS("Data/places_frac_l.Rds")
    
    places_frac_df = do.call("rbind", places_frac)
    saveRDS(places_frac_df, file = "Data/places_frac_df.Rds")
    
    
    
    
    imgnet_frac = foreach (i = 1:length(xls2018_v)) %dopar% {
        
        
        aoi_tmp = str_extract(xls2018_v[i], pattern = "Poly\\_[0-9]*")
        
        aoi_tmp = paste0("AOI_", as.numeric(substr(aoi_tmp, start = 6, 11)))
        
        img_name = paste0(path_out_network, "/imgnet_m_", aoi_tmp, ".csv")
        
        print(img_name)
        
        imgnet_all_m <- data.frame(matrix(data = 0, nrow = length(imgnet1000_tags), ncol = length(imgnet1000_tags)))
        dimnames(imgnet_all_m) <- list( imgnet1000_tags, imgnet1000_tags)
        
        
        if (file.exists(img_name)) { 
            imgnet_1mode = read.csv(file = img_name)
            imgnet_1mode$X = NULL
            colnames(imgnet_1mode) = str_replace_all(colnames(imgnet_1mode), pattern = "\\.", replacement = "-")
            colnames(imgnet_1mode) =  str_replace(colnames(imgnet_1mode), pattern = "potter-s_wheel", replacement =    "potter's_wheel")
            colnames(imgnet_1mode) =  str_replace(colnames(imgnet_1mode), pattern = "jack-o--lantern", replacement = "jack-o'-lantern")
            
            colnames(imgnet_1mode) =  str_replace(colnames(imgnet_1mode), pattern = "carpenter-s_kit", replacement = "carpenter's_kit")
            colnames(imgnet_1mode) =  str_replace(colnames(imgnet_1mode), pattern =  "yellow_lady-s_slipper", replacement = "yellow_lady's_slipper")
            
            
            rownames(imgnet_1mode) = colnames(imgnet_1mode)
            
            imgnet_all_m[rownames(imgnet_1mode ), colnames(imgnet_1mode)] = imgnet_all_m[rownames(imgnet_1mode ), colnames(imgnet_1mode)] + imgnet_1mode
            
        }
        
        return(colSums(imgnet_all_m))
        
    }
    
    
    imgnet_frac_df = do.call("rbind", imgnet_frac)
    saveRDS(imgnet_frac_df, file = "Data/imgnet_frac_df.Rds")
    
    
    
    
    
    
    
    
    
    
    joint_find = c("bicycle/built/for/two", "curly/coated_retriever", "flat/coated_retriever", "four/poster", "go/kart","hand/held_computer","hen/of/the/woods","red/backed_sandpiper","red/breasted_merganser","sulphur/crested_cockatoo", "ping/pong_ball", "pay/phone","potter/s_wheel","soft/coated_wheaten_terrier", "potter-s_wheel",  "carpenter/s_kit",  "yellow_lady/s_slipper", "jack/o//lantern", "Shih/Tzu", "German_short/haired_pointer", "three/toed_sloth", "long/horned_beetle", "wire/haired_fox_terrier", "black/footed_ferret", "black/and/tan_coonhound")
    
    
    joint_replaced = c("bicycle-built-for-two", "curly-coated_retriever", "flat-coated_retriever", "four-poster", "go-kart","hand-held_computer","hen-of-the-woods","red-backed_sandpiper","red-breasted_merganser","sulphur-crested_cockatoo", "ping-pong_ball", "pay-phone","potter-s_wheel","soft-coated_wheaten_terrier",  "potter's_wheel",  "carpenter's_kit", "yellow_lady's_slipper", "jack-o'-lantern", "Shih-Tzu", "German_short-haired_pointer", "three-toed_sloth", "long-horned_beetle", "wire-haired_fox_terrier", "black-footed_ferret", "black-and-tan_coonhound")
    
    
    joint_tags =  c(places365_tags, imgnet1000_tags)
    joint_tags = unique(joint_tags)
    
    
    joint_all_m = data.frame(matrix(data = 0,  nrow = length(joint_tags), ncol = length(joint_tags)))
    dimnames(joint_all_m) <- list(joint_tags, joint_tags)
    
    
    
    jt_frac = foreach (i = 1:length(xls2018_v)) %dopar% {
        
        
        aoi_tmp = str_extract(xls2018_v[i], pattern = "Poly\\_[0-9]*")
        aoi_tmp = paste0("AOI_", as.numeric(substr(aoi_tmp, start = 6, 11)))
        
        jt_name = paste0(path_out_network, "/joint_m_", aoi_tmp, ".csv")
        print(jt_name)
        
        if (file.exists(jt_name)) { 
            
            joint_1mode = read.csv(file = jt_name)
            joint_1mode$X = NULL
            
            colnames(joint_1mode) = str_replace_all(colnames(joint_1mode), pattern = "\\.", replacement = "/")
            
            for (jt in 1:length(joint_replaced)) { 
                
                colnames(joint_1mode) =  str_replace(colnames(joint_1mode), pattern = joint_find[jt], replacement =joint_replaced[jt])
            }
            
            print(colnames(joint_1mode)[!colnames(joint_1mode) %in% c(places365_tags, imgnet1000_tags)])
            
            
            
            
            # 
            rownames(joint_1mode) = colnames(joint_1mode)
            
            
            
            joint_all_m[rownames(joint_1mode ), colnames(joint_1mode)] =  joint_all_m[rownames(joint_1mode ), colnames(joint_1mode)]+ joint_1mode
            
        }
        return(colSums(joint_all_m))
        
    }
    
    joint_frac_df = do.call("rbind", jt_frac)
    saveRDS(joint_frac_df, file = "Data/joint_frac_df.Rds")
     
    
    
}






### 
places_frac_df = readRDS("Data/places_frac_df.Rds")
imgnet_frac_df = readRDS("Data/imgnet_frac_df.Rds")
joint_frac_df = readRDS("Data/joint_frac_df.Rds")


# boxplot(places_frac_df[, "park"])



# setwd("~/Dropbox/KIT/FlickrEU/")

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


n.thread <- detectCores() # 1 
proj4.LL <- "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs +towgs84=0,0,0"
# proj4.DHDN <- "+proj=tmerc +lat_0=0 +lon_0=12 +k=1 +x_0=4500000 +y_0=0 +ellps=bessel +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs" # epsg:31468



climsave <- readOGR("../../CLIMSAVE/IAP", layer="CLIMSAVE_EuropeRegions_for_SA")
climsave.ll <- spTransform(climsave, CRSobj = proj4.LL)

# library(maptools)
# library(ggplot2)

ctry.ids<-read.csv("~/Dropbox/KIT/CLIMSAVE/IAP/Cell_ID_LatLong.csv")
nums<-unique(ctry.ids$Nuts0_ID)
ctrys<-unique(ctry.ids$Dominant_NUTS0)


climsave.pos.idx <- match(climsave.ll$Cell_ID, ctry.ids$Cell_ID)

# plot(climsave.ll$Cell_ID, ctry.ids$Cell_ID[climsave.pos.idx])

climsave.ll$Dominant_NUTS2 <- ctry.ids$Dominant_NUTS2[climsave.pos.idx] 
climsave.ll$ClimateDescription <- ctry.ids$ClimateDescription[climsave.pos.idx] 

# spplot(climsave.ll, "Dominant_NUTS2")
# spplot(climsave.ll, "ClimateDescription")

aoi.countrycode <- climsave.ll$Dominant_NUTS2 
aoi.climate <- climsave.ll$ClimateDescription 

# lu.mulde.sachsen.nonurban.LL <- spTransform(lu.mulde.sachsen.nonurban, CRSobj = CRS(proj4.LL))
# plot(lu.mulde.sachsen.nonurban.LL)
aoi.poly.in <- climsave.ll 
# rm(climsave, climsave.ll)
# 



places.ll = climsave.ll 

places_frac_norm_df = t(apply(places_frac_df, MARGIN = 1, FUN = function(x) x / sum(x, na.rm=T)) * 100 )
places.ll@data = data.frame(places_frac_norm_df)


writeOGR(places.ll, dsn = "Data",layer= "Places365_norm", driver="ESRI Shapefile", overwrite_layer = T)




### Cluster info 

dt_name = "Places365"

cl_info = read.xlsx(paste0("Data/ClusterInfo_", dt_name, "_gte5.xlsx"))

k = 12 

cltb= cl_info[, paste0("Cluster_", k)]
places_k12_frac_df = t(apply(places_frac_df, MARGIN = 1, FUN = function(x) tapply(x, INDEX = cltb, FUN = sum)))

places_k12_frac_norm_df = t(apply(places_k12_frac_df, MARGIN = 1, FUN = function(x) x / sum(x, na.rm=T)) * 100 )

places_clusters.ll = climsave.ll 
places_clusters.ll@data = data.frame(places_k12_frac_norm_df)



writeOGR(places_clusters.ll, dsn = "Data",layer= "Places365_k12_norm", driver="ESRI Shapefile", overwrite_layer = T)


































# # str(res_l)
# res_big_df = do.call("rbind", res_l)
# nrow(res_big_df)
# 
# res_big_df = res_big_df[!duplicated(res_big_df$PhotoID),]
# nrow(res_big_df)
# 
# 
# write.xlsx(res_big_df, file = "../ESP2021/FlickrEU_Bayern_tags.xlsx")
# saveRDS(res_big_df, file = "../ESP2021/FlickrEU_Bayern_tags.Rds")
# 
# res_small_df = res_big_df
# 
# res_small_df$Latitude = NULL
# res_small_df$Longitude = NULL
# res_small_df$URL = NULL
# res_small_df$Geocontext = NULL
# 
# library(rgdal)
# 
# br_spdf = SpatialPointsDataFrame(cbind(res_big_df$Longitude, res_big_df$Latitude), data = res_small_df, proj4string = CRS( proj4.LL))
# 
# 
# plot(br_spdf, col = factor(br_spdf$Places365_Top1), pch=15, cex=0.1)
# 
# writeOGR(br_spdf, dsn = "../ESP2021/", layer = "FlickrEU_Bayern_tags", driver = "ESRI Shapefile", overwrite_layer = T)
# 
# 
# 
# table(br_spdf$Places365_Top1)


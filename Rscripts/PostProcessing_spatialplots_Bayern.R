library(doSNOW)
library(stringr)
library(readxl)
library(openxlsx)

proj4.LL <- "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs +towgs84=0,0,0"


# places
path_bayern_places = "../LabelledData/Places_EU/places365_resnet18/"
# list.files(path_bayern_places, pattern = ".csv")

# Imagenet tag
path_imagenet_tag = "../LabelledData/FlickrEU_Bayern/CSV converted/"

path_bayern_xls = "../FlickrEU_download/Bayern/Flickr_Excel_V2_Aug018_Bayern/"

xls_v = list.files(path_bayern_xls, pattern=".xlsx", full.names = T)

i = 1 

cl = makeCluster(8)
registerDoSNOW(cl)

res_l = foreach (i = 1:length(xls_v), .packages = c("stringr", "readxl", "doSNOW")) %dopar% { 
    print(i)
    aoi_tmp = str_extract(xls_v[i], pattern = "AOI\\_[0-9]*")
     
    xl_df = read_xlsx(xls_v[i])
    
    xl_df = xl_df[!duplicated(xl_df$PhotoID),]
    
    
    # unique(xl_df$PhotoID)
    
    places_df_name = paste0(path_bayern_places, "/", aoi_tmp, ".csv")
    
    if (!file.exists(places_df_name)) { 
        return(NULL)
    }
    
    # places365
    places_df = read.csv(places_df_name)
    
    photoid_tmp = str_extract(places_df$Filename, pattern="(?<=_)[0-9]*")
    
    places_df$Filename = NULL                
    
    colnames(places_df) = paste0("Places365_", colnames(places_df))
    places_df$PhotoID = photoid_tmp
    
    
    # Imagenet 
    imgnet_names = list.files(paste0(path_imagenet_tag, "/", aoi_tmp), full.names = T)
    imgnet_l = foreach(imgnet_name = imgnet_names) %do% { 
          read.csv(imgnet_name)
        
    }
    imgnet_df = do.call("rbind", imgnet_l)
    photoid_tmp2 = str_extract(imgnet_df$Filename, pattern="(?<=_)[0-9]*")
    imgnet_df$Filename = NULL
    
    colnames(imgnet_df) = paste0("IMGNET_", colnames(imgnet_df))
     
    imgnet_df$PhotoID = photoid_tmp2
    
    # photoid_tmp[match(xl_df$PhotoID, photoid_tmp)][1]
    merged_df1 = merge(xl_df, places_df, by = "PhotoID", all.x = T)
    merged_df2 = merge(merged_df1, imgnet_df, by = "PhotoID", all.x = T)
    
    
    return(merged_df2)
}

# str(res_l)
res_big_df = do.call("rbind", res_l)
nrow(res_big_df)

res_big_df = res_big_df[!duplicated(res_big_df$PhotoID),]
nrow(res_big_df)


write.xlsx(res_big_df, file = "../ESP2021/FlickrEU_Bayern_tags.xlsx")
saveRDS(res_big_df, file = "../ESP2021/FlickrEU_Bayern_tags.Rds")

res_small_df = res_big_df

res_small_df$Latitude = NULL
res_small_df$Longitude = NULL
res_small_df$URL = NULL
res_small_df$Geocontext = NULL

library(rgdal)

br_spdf = SpatialPointsDataFrame(cbind(res_big_df$Longitude, res_big_df$Latitude), data = res_small_df, proj4string = CRS( proj4.LL))


plot(br_spdf, col = factor(br_spdf$Places365_Top1), pch=15, cex=0.1)

writeOGR(br_spdf, dsn = "../ESP2021/", layer = "FlickrEU_Bayern_tags", driver = "ESRI Shapefile", overwrite_layer = T)



table(br_spdf$Places365_Top1)
 
 
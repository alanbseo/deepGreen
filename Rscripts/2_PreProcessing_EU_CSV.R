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

# i = 1 

# cl = makeCluster(16)
# registerDoSNOW(cl)

reprocessCSV = FALSE

if (reprocessCSV) {
    
    # places
    path_places_2018 = "~/Downloads/FlickrEU_result/Places_EU/places365_resnet18/EU28/Result/CSV/"
    path_places_2019 = "~/Downloads/FlickrEU_result/Places_EU2019/places365_resnet18/EU28/Result/CSV/"
    
    # list.files(path_bayern_places, pattern = ".csv")
    
    # Imagenet tag
    path_imagenet_2018_tag = "~/Downloads/FlickrEU_result/Tagging_EU/InceptionResnetV2/EU/Result/InceptionResnetV2/CSV/"
    path_imagenet_2019_tag = "~/Downloads/FlickrEU_result/Tagging_EU2019/InceptionResnetV2/EU/Result/CSV/"
    
    
    # meta data files
    path_EU2018_xls = "../FlickrEU_download/May2018_V1/May2018_V1_LC/"
    
    xls2018_v = list.files(path_EU2018_xls, pattern=".xlsx", full.names = T)
    
    path_EU2019_xls = "../FlickrEU_download/EU_Jan2019_V1/xls/"
    
    xls2019_v = list.files(path_EU2019_xls, pattern=".xlsx", full.names = T)
    
    
    
    res_l = foreach (i = 1:length(xls2018_v), .packages = c("stringr", "readxl", "doSNOW"), .errorhandling = "stop") %do% { 
        
        aoi_tmp = str_extract(xls2018_v[i], pattern = "Poly\\_[0-9]*")
        
        aoi_tmp = paste0("AOI_", as.numeric(substr(aoi_tmp, start = 6, 11)))
        
        out_filename_tmp = paste0(path_out_csv, "/", basename(xls2018_v[i])) 
        
        if (file.exists(out_filename_tmp)) {
            return()
        }
        print(i)
        
        
        xl2018_df = read_xlsx(xls2018_v[i])
        
        if (file.exists(xls2019_v[i])) { 
            
            xl2019_df = read_xlsx(xls2019_v[i])
            xl2019_df$LandcoverDesc = NA
            
            xl_df = rbind(xl2018_df, xl2019_df)
            rm(xl2019_df)
        } else { 
            xl_df = xl2018_df
        }
        
        if (nrow(xl_df)==0) { 
            print("no data")
            return()
            
        }
        
        colnames(xl_df)
        
        xl_df = xl_df[!duplicated(xl_df$PhotoID),]
        
        
        # unique(xl_df$PhotoID)
        
        places_df2018_name = paste0(path_places_2018, "/", aoi_tmp, ".csv")
        places_df2019_name = paste0(path_places_2019, "/", aoi_tmp, ".csv")
        
        if (!file.exists(places_df2018_name)) { 
            return(NULL)
        }
        
        # places365
        places2018_df = read.csv(places_df2018_name)
        
        
        if (file.exists(places_df2019_name)) { 
            places2019_df = read.csv(places_df2019_name)
            
            places_df = rbind(places2018_df, places2019_df)
            rm(places2019_df)
            rm(places2018_df)
            
        } else { 
            places_df = places2018_df
            rm(places2018_df)
            
        }
        
        places_df = places_df[!duplicated(places_df$Filename),]
        
        
        photoid_tmp = str_extract(places_df$Filename, pattern="(?<=_)[0-9]*")
        
        places_df$Filename = NULL                
        places_df$Year = NULL
        
        colnames(places_df) = paste0("Places365_", colnames(places_df))
        places_df$PhotoID = photoid_tmp
        
        # Imagenet 2018 
        imgnet2018_names = list.files(paste0(path_imagenet_2018_tag, "/", aoi_tmp), full.names = T)
        imgnet2018_l = foreach(imgnet_name = imgnet2018_names) %do% { 
            read.csv(imgnet_name)
            
        }
        imgnet2018_df = do.call("rbind", imgnet2018_l)
        
        if (file.exists(places_df2019_name)) { 
            # Imagenet 2019
            imgnet_df2019_name = paste0(path_imagenet_2019_tag, "/", aoi_tmp, ".csv")
            imgnet2019_df = read.csv(imgnet_df2019_name)
            imgnet2019_df$Year = NULL
            imgnet_df = rbind(imgnet2018_df, imgnet2019_df)
            rm(imgnet2018_df, imgnet2019_df)
            
        } else { 
            imgnet_df = imgnet2018_df
            rm(imgnet2018_df)
        }
        
        photoid_tmp2 = str_extract(imgnet_df$Filename, pattern="(?<=_)[0-9]*")
        imgnet_df$Filename = NULL
        
        colnames(imgnet_df) = paste0("IMGNET_", colnames(imgnet_df))
        
        imgnet_df$PhotoID = photoid_tmp2
        
        
        ### remove empty tags (using places tag)
        
        tag_df = merge(places_df, imgnet_df, by = "PhotoID", all = T)
        rm(places_df, imgnet_df)
        
        tag_df = tag_df[!is.na(tag_df$Places365_Top1),]
        
        
        # photoid_tmp[match(xl_df$PhotoID, photoid_tmp)][1]
        merged_df = merge(xl_df, tag_df, by = "PhotoID", all.x = T)
        rm(xl_df, tag_df)
        
        # return(merged_df2)
        
        write.xlsx(merged_df, file = out_filename_tmp)
        
    }
    
    
    # 
    # aoi_idxs = 1:23871
    # 
    # i = 22109 
    
    
    
    
    ### delete matchstick tags
    cl = makeCluster(16)
    registerDoSNOW(cl)
    
    i = 2 
    
    b =a[1, paste0("IMGNET_Top", 1:10)] 
    white_tags = as.character(b )
    
    res_l = foreach (i = 1:length(xls2018_v), .packages = c("stringr", "openxlsx", "doSNOW"), .errorhandling = "stop") %dopar% { 
        
        
        in_filename_tmp = paste0(path_out_csv, "/", basename(xls2018_v[i])) 
        
        aoi_tmp = str_extract(xls2018_v[i], pattern = "Poly\\_[0-9]*")
        aoi_tmp = paste0("AOI_", as.numeric(substr(aoi_tmp, start = 6, 11)))
        
        # aoi_tmp2 = paste0("AOI_", i)
        # stopifnot(aoi_tmp == aoi_tmp2)    
        
        # aoi_tmp3 = paste0("AOI_",  formatC(i, flag = "0", width = 5, format = "d"))
        
        if (!file.exists(in_filename_tmp)) { 
            return(NA)   
        }
        
        dt = read.xlsx(in_filename_tmp)
        
        
        dt[, paste0("IMGNET_Top", 1:10)][is.na(dt[, paste0("IMGNET_Top", 1:10)])] = ""
        
        
        tags_to_delete = apply(dt[dt$IMGNET_Top1 == "matchstick", paste0("IMGNET_Top", 1:10)], MARGIN = 1, FUN = function(x) all(x==white_tags))
        
        dt[dt$IMGNET_Top1 == "matchstick",][tags_to_delete, paste0("IMGNET_Top", 1:10)] = ""
        
        write.xlsx(dt, file = in_filename_tmp)
        
        
        return (NULL)
    }
    stopCluster(cl)
    
    
} 


createMatrix = FALSE 

if (createMatrix) { 
    ### Create 1-mode matrix
    
    
    
    ### delete matchstick tags
    # cl = makeCluster(16)
    # registerDoSNOW(cl)
    
    i = 11912
    
    
    myFunc = function(x, prob, lev) { 
        # print(x)
        # print(prob)
        # 
        fac = factor(x, levels = lev, ordered = TRUE)
        wt = numeric(length(lev))
        wtsum = tapply(prob, INDEX = as.numeric(fac), FUN = sum, na.rm=T)
        wt[as.numeric(names(wtsum))] = wtsum
        tb = tabulate(fac, nbins = length(lev))
        return( tb * wt )
        
    }
    
    weighted1modematrix = function (df_in, prob_in = NULL) {
        
        lev <- sort(unique(unlist(df_in)))
        
        dat <- do.call(rbind, lapply(1:nrow(df_in), FUN = function(x) myFunc(df_in[x,], as.numeric( prob_in[x,]), lev)))
        
        colnames(dat) <- lev
        t(data.frame(dat, check.names = FALSE))
    }
    # 
    # df_in = imgnet_df # [1:5, 1:5]
    # prob_in = imgnet_prob_df# [1:5, 1:5]
    # 
    # weighted1modematrix(df_in, prob_in)
    
    
    # 
    # make1modematrix <- function(keyword.vec.in, keywords.vec.ref) {
    #     matched <- match(keyword.vec.in, keywords.vec.ref)
    #     # matched.nona <- matched[!is.na(matched )]
    #     
    #     matched.tb <- table(matched)
    #     
    #     res.tmp <- numeric(length(keywords.vec.ref))
    #     res.tmp[as.numeric(names(matched.tb))] <-  (matched.tb)
    #     return(res.tmp)
    # }
    
    
    # print("Making the 1-mode matrix")
    
    # 
    # res.1modematrix <- apply(keyword.in.reduced, MARGIN = 2, function(x) make1modematrix(x, keywords.occurred))
    # apply(res.1modematrix, MARGIN = 1, FUN = table)
    
    
    # matrix_in  = places_mt
    # keywords.vec.ref = sort(unique(unlist(places_mt)))
    
    makefinalmatrix <- function(matrix_in, keywords.vec.ref) {
        
        
        final.m <- matrix(0, nrow = length(keywords.vec.ref), ncol =  length(keywords.vec.ref))
        
        # system.time({ 
        for (i in 1:length(keywords.vec.ref)) { 
            
            # if ((i %% 20) ==0) {
            #     cat(i, ">")
            # }
            
            imat_tmp <- matrix_in[i,]
            imat_idx_tmp = imat_tmp > 0
            
            for (j in 1:length(keywords.vec.ref)) {
                cooccur.idx <- which(imat_idx_tmp &matrix_in[j,] > 0)
                if (length(cooccur.idx)> 0) {
                    final.m[i,j] <- sum(imat_tmp[cooccur.idx] * matrix_in[j,][cooccur.idx], na.rm=T)
                }
            }
            
            diag(final.m) <- rowSums(matrix_in)
            
        } 
        # })
        
        colnames(final.m) <- rownames(final.m) <- keywords.vec.ref
        return(final.m)
    } 
    
    
    # i = 8245
    
    # stopCluster(cl)
    
    registerDoMC(n_thread)
    
    
    n_points = 23871
    
    # idxs = sample(10000:20000)
    idxs = 1:n_points
    
    
    # idxs = 19053
    
    # foreach (i = idxs, .errorhandling = "stop") %dopar% {
    for (i in idxs) {
        
        if ((i %% 100 ) ==0) {
            cat("AOI_", i, ">")
        }
        in_filename_tmp = paste0(path_out_csv, "/", basename(xls2018_v[i])) 
        
        
        if (!file.exists(in_filename_tmp)) {
            next
        }

        
        aoi_tmp = str_extract(xls2018_v[i], pattern = "Poly\\_[0-9]*")
        
        aoi_tmp = paste0("AOI_", as.numeric(substr(aoi_tmp, start = 6, 11)))
        
        
        
        dt = NULL    
        
        pl_name = paste0(path_out_network, "/places_m_", aoi_tmp, ".csv")
        
        if (!file.exists(pl_name)) {
            
            if (is.null(dt)) { 
                dt = read.xlsx(in_filename_tmp)
            }
            
            # places 
            places_df = dt[, paste0("Places365_Top", 1:10)] 
            places_df[places_df==""] = NA
            
            places_prob_df = dt[, paste0("Places365_Prob", 1:10)] 
            places_prob_df[places_prob_df==""] = NA
            
            
             # print("Making the 1-mode matrix")
            # places_mt <- t(mtabulate(places_df)) # non-weighted
            places_mt <- weighted1modematrix(places_df, places_prob_df)
            
            # print("Making the final matrix")
            places_1mode <- makefinalmatrix(places_mt,  sort(unique(unlist(places_df))))
            write.csv(places_1mode, file = pl_name)
            
        }
        
        img_name = paste0(path_out_network, "/imgnet_m_", aoi_tmp, ".csv")
        
        if (!file.exists(img_name)) {
            
            if (is.null(dt)) { 
                dt = read.xlsx(in_filename_tmp)
            }
            
            # Imagenet 
            imgnet_df = dt[, paste0("IMGNET_Top", 1:10)] 
            imgnet_df[imgnet_df==""] = NA
            
            imgnet_prob_df = dt[, paste0("IMGNET_Prob", 1:10)] 
            imgnet_prob_df[imgnet_prob_df==""] = NA
            
            imgnet_mt <- weighted1modematrix(imgnet_df, imgnet_prob_df)
            
            imgnet_1mode <- makefinalmatrix(imgnet_mt,   sort(unique(unlist(imgnet_df))))
            write.csv(imgnet_1mode, file = img_name)
            
        }
        
        jt_name = paste0(path_out_network, "/joint_m_", aoi_tmp, ".csv")
        
        if (!file.exists(jt_name)) {
            
            if (is.null(dt)) { 
                dt = read.xlsx(in_filename_tmp)
            }
            
            # places 
            places_df = dt[, paste0("Places365_Top", 1:10)] 
            places_df[places_df==""] = NA
            
            places_prob_df = dt[, paste0("Places365_Prob", 1:10)] 
            places_prob_df[places_prob_df==""] = NA
            
            
            # Imagenet 
            imgnet_df = dt[, paste0("IMGNET_Top", 1:10)] 
            imgnet_df[imgnet_df==""] = NA
            
            imgnet_prob_df = dt[, paste0("IMGNET_Prob", 1:10)] 
            imgnet_prob_df[imgnet_prob_df==""] = NA
            
            
            
            # joint 
            joint_df = cbind(places_df, imgnet_df)
            joint_prob_df = cbind(places_prob_df, imgnet_prob_df)
            joint_mt <- weighted1modematrix(joint_df, joint_prob_df)
            
            # keywords.vec.ref <- sort(unique(unlist(matrix_in)))
            
            joint_1mode <- makefinalmatrix(joint_mt, sort(unique(unlist(joint_df))))
            
            write.csv(joint_1mode, file = jt_name)
        }
        
    }
    
}


## aggregate csvs



places_all_m <- data.frame(matrix(data = 0, nrow = length(places365_tags), ncol = length(places365_tags)))
dimnames(places_all_m) <- list( places365_tags, places365_tags)


for (i in 1:length(xls2018_v))  {
    
    aoi_tmp = str_extract(xls2018_v[i], pattern = "Poly\\_[0-9]*")
    
    aoi_tmp = paste0("AOI_", as.numeric(substr(aoi_tmp, start = 6, 11)))
    
    pl_name = paste0(path_out_network, "/places_m_", aoi_tmp, ".csv")
    
    print(pl_name)
    
    if (file.exists(pl_name)) { 
        places_1mode = read.csv(file = pl_name)
        places_1mode$X = NULL
        
        colnames(places_1mode) = str_replace(colnames(places_1mode), pattern = "\\.", replacement = "/")
        
        rownames(places_1mode) = colnames(places_1mode)
        places_all_m[rownames(places_1mode ), colnames(places_1mode)] = places_all_m[rownames(places_1mode ), colnames(places_1mode)] +  places_1mode
        
    } 
    
}
write.csv(places_all_m, file = "Data/places_all_m.csv")
saveRDS(places_all_m, file = "Data/places_all_m.Rds")



imgnet_all_m <- data.frame(matrix(data = 0, nrow = length(imgnet1000_tags), ncol = length(imgnet1000_tags)))
dimnames(imgnet_all_m) <- list( imgnet1000_tags, imgnet1000_tags)


# colnames(imgnet_1mode)[!colnames(imgnet_1mode) %in% imgnet1000_tags]

for (i in 1:length(xls2018_v))  {
    
    aoi_tmp = str_extract(xls2018_v[i], pattern = "Poly\\_[0-9]*")
    
    aoi_tmp = paste0("AOI_", as.numeric(substr(aoi_tmp, start = 6, 11)))
    
    img_name = paste0(path_out_network, "/imgnet_m_", aoi_tmp, ".csv")
    print(img_name)
    
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
}

write.csv(imgnet_all_m, file = "Data/imgnet_all_m.csv")
saveRDS(imgnet_all_m, file = "Data/imagenet_all_m.Rds")




joint_tags =  c(places365_tags, imgnet1000_tags)
joint_tags = unique(joint_tags)


joint_all_m = data.frame(matrix(data = 0,  nrow = length(joint_tags), ncol = length(joint_tags)))
dimnames(joint_all_m) <- list(joint_tags, joint_tags)


 


joint_find = c("bicycle/built/for/two", "curly/coated_retriever", "flat/coated_retriever", "four/poster", "go/kart","hand/held_computer","hen/of/the/woods","red/backed_sandpiper","red/breasted_merganser","sulphur/crested_cockatoo", "ping/pong_ball", "pay/phone","potter/s_wheel","soft/coated_wheaten_terrier", "potter-s_wheel",  "carpenter/s_kit",  "yellow_lady/s_slipper", "jack/o//lantern", "Shih/Tzu", "German_short/haired_pointer", "three/toed_sloth", "long/horned_beetle", "wire/haired_fox_terrier", "black/footed_ferret", "black/and/tan_coonhound")


joint_replaced = c("bicycle-built-for-two", "curly-coated_retriever", "flat-coated_retriever", "four-poster", "go-kart","hand-held_computer","hen-of-the-woods","red-backed_sandpiper","red-breasted_merganser","sulphur-crested_cockatoo", "ping-pong_ball", "pay-phone","potter-s_wheel","soft-coated_wheaten_terrier",  "potter's_wheel",  "carpenter's_kit", "yellow_lady's_slipper", "jack-o'-lantern", "Shih-Tzu", "German_short-haired_pointer", "three-toed_sloth", "long-horned_beetle", "wire-haired_fox_terrier", "black-footed_ferret", "black-and-tan_coonhound")

for (i in 1:length(xls2018_v))  {
    
    
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
    
}


write.csv(joint_all_m, file = "Data/joint_all_m.csv")
saveRDS(joint_all_m, file = "Data/joint_all_m.Rds")

# save.image("~/Downloads/FlickrEU_June8_2021.RData")




# res_err_cnt = sapply(res_l, FUN = c)
# table(res_err_cnt > 10 )
# summary(res_err_cnt)
# 
# 
# boxplot(res_l)



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


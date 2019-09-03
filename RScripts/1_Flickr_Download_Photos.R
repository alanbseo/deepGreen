library(imager)
# library(httr)
# library(RCurl)
# library(rjson)
# library(raster)
library(rgdal)
library(rgeos)
library(doMC)
# library(openxlsx)
library(readxl)
library(stringr)


n.thread <- detectCores() * 3 # 1 
proj4.LL <- "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs +towgs84=0,0,0"
proj4.DHDN <- "+proj=tmerc +lat_0=0 +lon_0=12 +k=1 +x_0=4500000 +y_0=0 +ellps=bessel +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs" # epsg:31468

proj4.etrs_laea <- "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs";
# proj4.EUR_ETRS89_LAEA1052 <- "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs" # EPSG:3035

# corine2012 <- raster("../GIS/Corine/CLC2012/g250_clc12_V18_5a/g250_clc12_V18_5.tif", 1)
# corine2012 <- raster("../GIS/Corine/CLC2012/g100_clc12_V18_5a/g100_clc12_V18_5.tif")
# proj4string(corine2012) <- proj4.etrs_laea
# corine2012.ll <- projectRaster(corine2012, crs = proj4.LL, method = "ngb")


# corine.tb <- read.csv("../GIS/Corine/CLC2012/g100_clc12_V18_5a/Legend/clc_legend.csv", sep = ";")
# corine.tb$CLC_CODE

# corine.detail.tb <- read.xlsx("../GIS/Corine/clc_legend.xlsx", 1)


# Login credentials
api.key <- "e8008cb908d630a5f6e9b9d97f351c79" # API key for Flickr API goes here
api.secret <- "f86de9bc07e449fe" # Not used


# climsave <- readOGR("../GIS", layer="CLIMSAVE_EuropeRegions_for_SA")
# climsave.ll <- spTransform(climsave, CRSobj = proj4.LL)

# library(maptools)
# library(ggplot2)

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

# 
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
# 
# 
# aoi.poly.in <- climsave.ll
# # rm(climsave, climsave.ll)


# Time span
mindate <- "2005-01-01"
maxdate <- "2019-07-31"
# savedir <- substr(mindate, 6, 10)
# savedir <- "May2018_V1/"
workdir <-  "~/Dropbox/KIT/FlickrEU/Costa Rica_Data/FlickrCR_download/"

# newsavedir <- paste0("Jan2019_V1_LC") 
newsavedir <- paste0("Aug2019_V1") 
 
# shpdir <- paste0("May2018_V1_SHP")

# photodir <- paste0("/pd/data/crafty/FlickrEU_DOWNLOAD_11Jan2019/Jan2019_V1_Photo")
# photodir <- paste0("~/Dropbox/KIT/FlickrEU/FlickrEU_download/Jan2019_V1_Photo")
# photodir <- paste0("/POCO2TB/DATA2TB/FlickrEU_download/Jan2019_V1_Photo_small/")
# photodir <- paste0("~/Downloads/Jan2019_V1_Photo_Y2018_Part1/")
# photodir <- paste0("/Users/seo-b/nfs_keal_pd/FlickrEU_DOWNLOAD_11Jan2019/Jan2019_V1_Photo_small/")
photodir <- paste0("/DATA2TB/FlickrCR_download/Aug2019_V1_Photo/")


# Search parameters
sort <- "interestingness-desc" # Sort by Interestingness (or: relevance)
max.perpage <- 250 # number per page maximum 250

aoi.poly.in = readOGR( dsn = paste0(workdir, newsavedir), layer = "FlickrCR_AOI")  

n.points <- length(aoi.poly.in)



# Retreiving the data

# 
# # target.ids.all <- 1:n.points
# target.ids.all <- 1:n.points
# 
# aois.done <- list.files(paste0(workdir, "/", newsavedir), pattern = "^AOI.*.\\.xlsx$")
# aois.done.v <- as.numeric(sapply(aois.done, FUN = function(x)  (str_split(x, pattern = "_")[[1]][3])))
# nphotos.done.v <- as.numeric(sapply(aois.done, FUN = function(x)  str_extract(str_split(x, pattern = "_")[[1]][7], "[0-9]+")))
# cntrys.done.v <-  str_sub(sapply(aois.done, FUN = function(x)  (str_split(x, pattern = "_")[[1]][5])), 1,2)
# climate.done.v <-  str_sub(sapply(aois.done, FUN = function(x)  (str_split(x, pattern = "_")[[1]][6])))
# 
# 
# boxplot(nphotos.done.v)
# sum(nphotos.done.v)* 1E-6
# 
# cntrys.freq <- tapply(X = nphotos.done.v, INDEX = cntrys.done.v, FUN = sum, na.rm=T)
# barplot(cntrys.freq, las=2)
# 
# cntrys.v <- tapply(X = nphotos.done.v, INDEX = cntrys.done.v, FUN = c, na.rm=T)
# 
# boxplot(cntrys.v, las=2)
# 
# 
# climate.v <- tapply(X = nphotos.done.v, INDEX = climate.done.v, FUN = c, na.rm=T)
# 
# boxplot(climate.v, las=2)# scale=log)
# 
# 
# 
# 
# 
# climsave.ll$NPHOTO <-  nphotos.done.v 


# col.nphoto <- log(climsave.ll$NPHOTO + 1 )

# plot(climsave.ll) # , col = col.nphoto)

# writeOGR(climsave.ll, dsn = "Data", layer = "FlickrEU_Nphoto", verbose = T, overwrite_layer = T, driver = "ESRI Shapefile")




#### Download photos 
aois.done.newnames <- list.files(paste0(workdir, "/", newsavedir, "/Xlsx/"), pattern = "^AOI.*.\\.xlsx$", full.names = T)

# aoi.idx <- 1


### Write with LC
#
# aois.done <- list.files(paste0(workdir, "/", savedir), pattern = "^AOI.*.\\.xlsx$")
# aois.done.v <- as.numeric(sapply(aois.done, FUN = function(x) (str_split(x, pattern = "_")[[1]][3])))
#
# aois.new.done <- list.files(paste0(workdir, "/", shpdir), pattern = "^AOI.*.\\.shp$")
# aois.new.done.v <- as.numeric(sapply(aois.new.done, FUN = function(x) (str_split(x, pattern = "_")[[1]][3])))
# target.ids <-    (setdiff(aois.done.v, aois.new.done.v))

# aoi.done <- list.files(paste0(photodir), pattern="^AOI_")
# aois.done.v <- as.numeric(sapply(aoi.done, FUN = function(x) (str_split(x, pattern = "_")[[1]][2])))

# target.ids <- setdiff(1:n.points, aois.done.v)

# target.ids <-    (setdiff(aois.done.v, aois.new.done.v))
# length(aois.done.v)  - length(target.ids)
# cat(length(target.ids), "to go")



registerDoMC(n.thread)
Sys.setlocale(category = "LC_ALL", "en_US.UTF-8")


# target.ids <- readRDS("Bayern_aoiid.Rds")

# target.ids <- 1:n.points
# target.ids <- 23490:1
# target.ids <- 1:n.points
# target.ids <- 13225:16370 # Bayern 
# target.ids <- 16371:22990
# target.ids <- 8201:13224
# target.ids <- 13398:16370
# target.ids <- 18744:22683

# target.ids <- 13169:13224
# target.ids <- 18915:22683

# target.ids <- 20001:22683
# target.ids <- 20001:n.points
# target.ids <- 20312:n.points
# target.ids <- 21501:n.points
# target.ids <- c(20501:n.points, 16371:22290, 1:16370

# target.ids <- n.points:1
# target.ids <- 13225:16370 # Bayern

# target.ids <- 16371:n.points
# target.ids <- 10001:13224
# target.ids <- 1:10000
# target.ids <- n.points:20001
# target.ids <- 13504:2000
# target.ids = 1:n.points 
#target.ids = 1:20491
target.ids.all <- aoi.poly.in$CELL_ID

foreach (i = 148:n.points, .inorder = F, .errorhandling = "stop", .verbose = F) %do% { 
    
    
    # aois.new.done <- list.files(paste0(workdir, "/", newsavedir), pattern = "^AOI.*.\\.xlsx$")
    # aois.new.done.v <- as.numeric(sapply(aois.new.done, FUN = function(x) (str_split(x, pattern = "_")[[1]][3])))
    # 
    
    # print("process")
    aoi.tmp <- aois.done.newnames[i]
    aoi_cellid = target.ids.all[i]
    print(paste0("i=", i, " CellID=", aoi_cellid))
    
     
    desc_natpark = aoi.poly.in$NAME[i]
    desc_natpark = ifelse(is.na(desc_natpark), "", paste0( "_NatPark_",desc_natpark))
    aoi.desc <- paste0(aoi.poly.in$NAME_2[i], "", desc_natpark)
    
    
    
    
    # aoi.dt <- read.xlsx(aoi.tmp, 1, detectDates = F)
    aoi.dt.raw <- data.frame(read_excel(aoi.tmp, sheet = 1))
    # aoi.dt$Title <- iconv(aoi.dt$Title, from="UTF-8", to="ASCII", sub="")
    # aoi.dt$Username <- iconv(aoi.dt$Username, from="UTF-8", to="ASCII", sub="") # Special characters.. be careful as we flattend the UTF-8 usernames to ASCII
    
    
    photo.ids.unique <- as.numeric( unique( aoi.dt.raw$PhotoID)  )
    aoi.dt <- aoi.dt.raw[match(photo.ids.unique, aoi.dt.raw$PhotoID ),]
    # table(aoi.dt$PhotoID == 16469798504)
    # print(nrow(aoi.dt))
    
    
    if (is.null(nrow(aoi.dt)) || nrow(aoi.dt)==0) {   
        # write.xlsx(data.frame(NA), aois.done.newnames[aoi.idx], overwrite=T)
        imgdir <- paste0(photodir, "/", "AOI_", i, "_EMPTY")
        if (!dir.exists(imgdir)) { 
            dir.create(imgdir, recursive = T)
            cat("AOI_", aoi_cellid, "_created >")
            
        }
        print(paste(aoi_cellid, " has no photos >")   )
        return(NULL)
    } else {
        print(paste(aoi_cellid, " has ", nrow(aoi.dt), " photos >")) 
        
    }
    
    # numeric.cols <- c( "Year", "Longitude", "Latitude", "Geocontext", "LocationAccuracy", "N_FlickrTag")
    # 
    # aoi.dt[, numeric.cols] <- sapply(numeric.cols, FUN = function(x) as.numeric(aoi.dt[,x]))
    # # str(aoi.dt)
    # 
    # aoi.sp <- SpatialPoints(coords = cbind(x= aoi.dt$Longitude, y = aoi.dt$Latitude), proj4string = crs( proj4.LL))
    # aoi.sp.etrs <- spTransform(aoi.sp, CRSobj = proj4.etrs_laea)
    # 
    # aoi.lc <- extract(corine2012, aoi.sp.etrs, fn = c)
    # stopifnot(length(aoi.lc)== length(aoi.sp))
    # 
    # aoi.clc_code <- corine.tb$CLC_CODE[ match(aoi.lc,  corine.tb$GRID_CODE)]
    # aoi.clc_lebel3 <- corine.tb$LABEL3[ match(aoi.lc,  corine.tb$GRID_CODE)]
    # 
    # aoi.dt$Landcover <- aoi.clc_code
    # aoi.dt$LandcoverDesc <- aoi.clc_lebel3
    
    
    foreach (p.idx = 1:nrow(aoi.dt), .inorder = F, .errorhandling = "stop", .verbose = F) %dopar% { 
        photo.year <- aoi.dt$Year[p.idx]
        photo.landcoverdesc <- aoi.dt$LandcoverDesc[p.idx]
        photo.owner <- aoi.dt$Owner[p.idx]
        photo.id <- aoi.dt$PhotoID[p.idx]
        photo.date <- aoi.dt$Date[p.idx]
        
        imgdir.annual <- paste0(photodir, "/", "AOI_CellID", formatC(aoi_cellid, width = 6, flag = "0"), "_", aoi.desc, "/", photo.year)
        
        if (!dir.exists(imgdir.annual)) { 
            cat("AOI_", aoi_cellid, "_create ", photo.year, "_s>")
            
            dir.create(imgdir.annual, recursive = T)
        }
        
        temp <- paste(imgdir.annual, "/photoid_", photo.id, "_date_", photo.date, "_owner_", photo.owner, ".jpg", sep="")
        
        if (!file.exists(temp)) {
            cat("AOI_", aoi_cellid, "_photoid", photo.id, "_s>")
            tryCatch(download.file(aoi.dt$URL[p.idx], temp, mode="wb", cacheOK = T), error= function(e) print(e))            
        } else { 
            
            checkImage = TRUE
            
            if (checkImage) { 
                img <- imager::load.image(temp)
                
                if (!is.cimg(img)) {
                    print("re-download the file")
                    cat("AOI_", aoi_cellid, "_photoid", photo.id, "_s>")
                    
                    download.file(aoi.dt$URL[p.idx], temp, mode="wb")
                } else {
                    # cat(aoi.idx, "_", photo.id, "_s>")
                    cat(".")
                    
                }
                
            }
        }
        
    }
    
    # write.xlsx(aoi.dt, file = aois.done.newnames[aoi.idx], overwrite=T)
    # read.xlsx(aois.done.newnames[aoi.idx])
    
    return(NULL)
}






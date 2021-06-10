setwd("~/Dropbox/KIT/FlickrEU/")

library(rgdal)
library(httr)
library(RCurl)
library(rjson)
library(raster)
library(rgdal)
library(rgeos)
library(doMC)
library(openxlsx)
library(readxl)
library(stringr)


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

# corine2012 <- raster("../GIS/Corine/CLC2012/g250_clc12_V18_5a/g250_clc12_V18_5.tif", 1)
corine2012 <- raster("../GIS/Corine/CLC2012/g100_clc12_V18_5a/g100_clc12_V18_5.tif")
proj4string(corine2012) <- proj4.etrs_laea
# corine2012.ll <- projectRaster(corine2012, crs = proj4.LL, method = "ngb")


corine.tb <- read.csv("../GIS/Corine/CLC2012/g100_clc12_V18_5a/Legend/clc_legend.csv", sep = ";")
corine.tb$CLC_CODE

corine.detail.tb <- read.xlsx("../GIS/Corine/clc_legend.xlsx", 1)


# Login credentials
api.key <- "e8008cb908d630a5f6e9b9d97f351c79" # API key for Flickr API goes here
api.secret <- "f86de9bc07e449fe" # Not used


climsave <- readOGR("../GIS", layer="CLIMSAVE_EuropeRegions_for_SA")
climsave.ll <- spTransform(climsave, CRSobj = proj4.LL)

# library(maptools)
# library(ggplot2)

ctry.ids<-read.csv("~/Dropbox/KIT/CLIMSAVE/IAP/Cell_ID_LatLong.csv")
nums<-unique(ctry.ids$Nuts0_ID)
ctrys<-unique(ctry.ids$Dominant_NUTS0)

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


climsave.pos.idx <- match( climsave.ll$Cell_ID, ctry.ids$Cell_ID)

# plot(climsave.ll$Cell_ID, ctry.ids$Cell_ID[climsave.pos.idx])

climsave.ll$Dominant_NUTS2 <- ctry.ids$Dominant_NUTS2[climsave.pos.idx] 
climsave.ll$ClimateDescription <- ctry.ids$ClimateDescription[climsave.pos.idx] 

# spplot(climsave.ll, "Dominant_NUTS2")
# spplot(climsave.ll, "ClimateDescription")

aoi.countrycode <- climsave.ll$Dominant_NUTS2 
aoi.climate <- climsave.ll$ClimateDescription 


aoi.poly.in <- climsave.ll
# rm(climsave, climsave.ll)


# Time span
mindate <- "2005-01-01"
maxdate <- "2017-12-31"
# savedir <- substr(mindate, 6, 10)
savedir <- "May2018_V1/"
workdir <- "FlickrEU_download/"
newsavedir <- paste0("May2018_V1_LC_reduced")
shpdir <- paste0("May2018_V1_SHP")



# Search parameters
sort <- "interestingness-desc" # Sort by Interestingness (or: relevance)
max.perpage <- 250 # number per page maximum 250
n.points <- length(aoi.poly.in)



# Retreiving the data


# target.ids.all <- 1:n.points
target.ids.all <- 1:n.points

aois.done <- list.files(paste0(workdir, "/", savedir), pattern = "^AOI.*.\\.xlsx$")
aois.done.v <- as.numeric(sapply(aois.done, FUN = function(x)  (str_split(x, pattern = "_")[[1]][3])))


nphotos.done.v <- as.numeric(sapply(aois.done, FUN = function(x)  str_extract(str_split(x, pattern = "_")[[1]][7], "[0-9]+")))

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






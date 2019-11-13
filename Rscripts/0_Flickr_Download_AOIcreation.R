
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
proj4.DHDN <- "+proj=tmerc +lat_0=0 +lon_0=12 +k=1 +x_0=4500000 +y_0=0 +ellps=bessel +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs" # epsg:31468

# Login credentials
api.key <- "e8008cb908d630a5f6e9b9d97f351c79" # API key for Flickr API goes here
api.secret <- "f86de9bc07e449fe" # Not used


# locations 
path_data = "~/Dropbox/KIT/FlickrEU/Costa Rica_Data/"

# Search data: either hashtag or location (bbox = Bounding Box)
# hashtag <- "landscape" # Set "" for all hashtags
hashtag <- "" # Set "" for all hashtags

 
# library(maptools)
# library(ggplot2)

costarica_adm2_ll = readOGR(paste0(path_data, "GIS data/CRI_adm/"), layer = "CRI_adm2", verbose = T)

# spplot(costarica_adm2_ll, "NAME_2")

costarica_natpark = readOGR(paste0(path_data, "GIS data/NatParks_CR/NatParks_CR_Nov2019.gpkg"), verbose = T)

plot(costarica_adm2_ll, add=F)
plot(costarica_natpark, add=T, col = "red")

### Create 0.25 deg long-lat grid 
costarica_ext1 = extent(costarica_adm2_ll)
costarica_ext2 = extent(costarica_natpark)

# plot(costarica_ext, add=T)
# costarica_ext1
# costarica_ext2

costarica_ext = merge(costarica_ext1, costarica_ext2)


plot(costarica_ext, col="red")
plot(costarica_ext1, add=T)
plot(costarica_ext2, add=T)


costarica_ext










r = raster(resolution=0.1, xmn=floor(costarica_ext@xmin-0.01), xmx=ceiling(costarica_ext@xmax +0.01), ymn=floor(costarica_ext@ymin -0.01), ymx=ceiling(costarica_ext@ymax + 0.01 ))
r[] = 1:ncell(r)
 
plot(r, add=F)
# plot(costarica_adm2_ll, add=T)
# plot(costarica_natpark, add=T, col = "red")

costarica_aoi_all = rasterToPolygons(r)
proj4string(costarica_aoi_all)= proj4string(costarica_adm2_ll)

costarica_over = over(costarica_aoi_all, costarica_adm2_ll, returnList = F)
costarica_over2 = over(costarica_aoi_all, costarica_natpark, returnList = F)

costarica_overlap_idx = !is.na(costarica_over$NAME_2) | !is.na(costarica_over2$NAME) # overlap either with the adm2 or the nat park polygons

costarica_aoi  = costarica_aoi_all[costarica_overlap_idx,]
names(costarica_aoi) <- "CELL_ID"

plot(costarica_ext, xlab= "Lon", ylab="Lat")

plot(costarica_ext, xlab= "Lon", ylab="Lat", border=NA, col=NA)








 
costarica_aoi_adm2_over = over(costarica_aoi, costarica_adm2_ll, fun = mod, returnList = F)

costarica_aoi_adm2_over$NL_NAME_2 = NULL
costarica_aoi_adm2_over$NAME_0 = NULL 
costarica_aoi_adm2_over$ID_0 = NULL
costarica_aoi_adm2_over$TYPE_2 = costarica_aoi_adm2_over$ENGTYPE_2 = NULL 

costarica_aoi_natpark_over = over(costarica_aoi, costarica_natpark,  fun = mod, returnList = F)

costarica_aoi_natpark_over$GIS_AREA = NULL
costarica_aoi_natpark_over$GIS_M_AREA = NULL
costarica_aoi_natpark_over$REP_M_AREA = NULL
costarica_aoi_natpark_over$REP_AREA = NULL
 

costarica_aoi@data = cbind(CELL_ID= costarica_aoi$CELL_ID,costarica_aoi_adm2_over, costarica_aoi_natpark_over)

 
spplot(costarica_aoi, "NAME")


costarica_aoi$CELL_ID
spplot(costarica_aoi, "CELL_ID")

 
writeOGR(costarica_aoi, dsn = paste0(path_data, "GIS data"), layer = "FlickrCR_AOI_Nov2019", driver = "ESRI Shapefile", overwrite_layer = T)
 








# costarica_aoi = readOGR(dsn = "../Costa Rica_Data/FlickrCR_download/Aug2019_V1/", layer = "FlickrC_AOI_Nphotos.shp")
# 
# sum(costarica_aoi$NPHOTOS_)
# 
# sum(costarica_aoi$NPHOTOS)

# 
# #### NEW. TO sort out
# dt = readOGR("../Costa Rica_Data/FlickrCR_download/Aug2019_V1/FlickrCR_AOI.shp")
# dt$NPHOTOS = nphotos.done.v
# dt$NPHOTOS_UNIQUE = nphotos.done.reduced.v
# 
# 
# nphotos_perct = quantile(dt$NPHOTOS_UNIQUE, probs= seq(0, 1, 0.1))
# nphotos_perct = unique(nphotos_perct)
# 
# nphotos_cut = cut(dt$NPHOTOS_UNIQUE, breaks= as.numeric(nphotos_perct))
# 
# 
# # levels(nphotos_cut)
# aoi_col = rev(topo.colors(length(nphotos_perct)))[as.numeric(nphotos_cut)]
# plot(costarica_ext, xlab= "Lon", ylab="Lat", border=NA, col=NA)
# 
# plot(dt, col = aoi_col, add=T, border="grey")
# plot(costarica_adm2_ll, add=T, col=NA, border="grey")
# plot(costarica_natpark, add=T, col=NA, border='red')
# 
# legend("bottomright", title = "# of Flickr photos", legend = levels(nphotos_cut), col = rev(topo.colors(length(nphotos_perct))), pch=15, bty="n")

# plot(costarica_aoi, add=T)

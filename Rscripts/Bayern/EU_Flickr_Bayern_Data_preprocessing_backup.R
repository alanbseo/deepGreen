 library(rgeos)
library(rgdal)
library(doParallel)
n.thread = 1

if (Sys.info()["sysname"][[1]] == "Linux") { 
    base.path <- "~/pd/FlickrEU_DOWNLOAD_14May2018/"
    xls.path <- "~/pd/FlickrEU_DOWNLOAD_14May2018/May2018_V1_LC/"
    photo.path <- paste0(base.path, "/May2018_V1_Photo/")
    target.path <-paste0(base.path,  "/Aug2018_V2_Photo/")
    script.path <- "~/pd/FlickrEU/"
    
} else { 
    base.path <- "~/Dropbox/KIT/FlickrEU/"
    xls.path <- "~/Dropbox/KIT/FlickrEU/FlickrEU_download/May2018_V1_LC/"
    photo.path <- paste0(base.path, "/May2018_V1_Photo/")
    target.path <-paste0(base.path,  "/Aug2018_V2_Photo/")
    script.path <- "~/Dropbox/KIT/FlickrEU/"
}



setwd(script.path)
# 
 bavaria.shp <- readOGR("~/Dropbox/KIT/FlickrEU/Bavaria_Original_PUD/pud_results.shp")
 bavaria_one.shp <- gBuffer(gUnaryUnion(bavaria.shp), 100, byid = F) # three cells missing before (w/o using a buffer)
# 
# 
 plot(bavaria_one.shp)
# 
 proj4string(bavaria_one.shp)
# 
climsave.shp <- readOGR("Data/FlickrEU_Nphoto.shp")
 proj4.ll <- proj4string(climsave.shp)
 proj4.utm32 <- proj4string(bavaria_one.shp)
# # 
 bavaria_one_ll.shp <- spTransform(bavaria_one.shp, CRSobj = proj4.ll)
# # 
 plot(climsave.shp)
 plot(bavaria_one_ll.shp, add=T, col="red")
# # 
# # 
# climsave.bavaria.idx1 <- which(gCovers(bavaria_one_ll.shp, climsave.shp, byid = T))
# climsave.bavaria.idx2 <- which(gOverlaps( climsave.shp, bavaria_one_ll.shp, byid = T))
# # 
# climsave.bavaria.idx <- unique(c(climsave.bavaria.idx1, climsave.bavaria.idx2))
# climsave.bavaria.shp <- climsave.shp[ climsave.bavaria.idx,]
# # 
# plot(bavaria_one_ll.shp, add=F, col="red")
# plot(climsave.bavaria.shp, add=T)
# # 
# 1694 %in%  as.numeric(climsave.shp$Cell_ID)[climsave.bavaria.idx]
# 
# 1694 %in% as.numeric(climsave.bavaria.shp$Cell_ID)
# 
# 1694 %in%  as.numeric(climsave.shp$Cell_ID)
# as.numeric(as.character(climsave.shp$Cell_ID))[climsave.bavaria.idx]


# climsave.bavaria_utm32.shp <- spTransform(climsave.bavaria.shp, CRSobj = proj4.utm32)
# # 
# plot(climsave.bavaria_utm32.shp)
# plot(bavaria_one.shp, add=T)
# # 
# if (FALSE) {
    # writeOGR(climsave.bavaria_utm32.shp, dsn = "../../Hong, Sunhae/Bayern", layer = "Flickrgrid_Bavaria_utm32", driver="ESRI Shapefile", overwrite_layer = T)
# }
# 
# # cell ids 
# climsave.bavaria_utm32.shp$Cell_ID
# climsave.bavaria.idx
# 
# # which(climsave.bavaria.idx==16726)

# length(xls.files) 
# length(photo.folders)
# head(xls.files)


# tmp.dir <- paste0("~/nfs_keal_pd/FlickrEU_DOWNLOAD_14May2018/May2018_V1_Photo", "/", photo.folders[1])
# sum(file.info(list.files(tmp.dir, all.files = TRUE, recursive = TRUE))$size)
# system(paste0("du -sh ", tmp.dir))

# saveRDS(climsave.bavaria.idx, "Data/Bayern_AOI_IDs.Rds")
climsave.bavaria.idx <- readRDS("Data/Bayern_AOI_IDs.Rds")
# photo.folders <- list.dirs(photo.path)
# 
# 
# Poly_ID_char.v <- ((substr(photo.folders, start = 5, 20)))
# Poly_ID.v <- as.numeric(Poly_ID_char.v)
# Poly_ID_pad.v <- formatC(Poly_ID.v, digits=5, width=5, flag="0")
# 
# climsave.bavaria.idx[!(climsave.bavaria.idx %in% Poly_ID.v)]
# 
# 
# target.folders <- photo.folders[ match(climsave.bavaria.idx, Poly_ID.v)]

target.folders <- paste0("AOI_", climsave.bavaria.idx)

# spplot(climsave.bavaria_utm32.shp, "Cell_ID")


Cell_ID.v <- as.numeric(as.character(climsave.shp$Cell_ID))
# 
Cell_ID_pad.v <- formatC(Cell_ID.v, digits = 5, width=5, flag="0")
# 
# plot(Poly_ID.v, Cell_ID.v)

target.cellids <- Cell_ID_pad.v[climsave.bavaria.idx]



if (!dir.exists(target.path)) { 
    dir.create(target.path)
}
target.folders.newnames <- paste0("CellID_", target.cellids, "_", target.folders )
aoi.idx <- 1
# 

 
registerDoParallel(n.thread)

foreach (aoi.idx = 1:length(target.folders)) %do% {
    print(aoi.idx)
    cmd.tmp <- paste0("cp -r ", photo.path, target.folders[aoi.idx], " ", target.path, "/", target.folders.newnames[aoi.idx])
    print (cmd.tmp)
    system(cmd.tmp)
}
# 
xls.files <- list.files(paste0(xls.path), pattern = "^AOI*?.*.xlsx")

# substr(xls.files, start = 1, stop = 15)

xls.new.files.back <- substr(xls.files, start = 16, stop = sapply(xls.files, str_length))

xls.new.files <- paste0("CellID_", formatC(as.numeric(as.character(climsave.shp$Cell_ID)), digits = 5, width=5, flag="0"), "_AOI_", formatC(1:nrow(climsave.shp),digits = 5, width=5, flag="0"), xls.new.files.back)
# 

# 
xls.new.path <- "FlickrEU_download/May2018_V1_LC_renamed/"
dir.create(xls.new.path)
for (aoi.idx in climsave.bavaria.idx) {

# for (aoi.idx in 1:length(xls.files)) {
    print(aoi.idx)

    file.copy(paste0(xls.path, "/", xls.files[aoi.idx]), to = paste0(xls.new.path, xls.new.files[aoi.idx]))
        # print (cmd.tmp)
    # system(cmd.tmp)
}






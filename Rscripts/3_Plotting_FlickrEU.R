# install.packages("devtools")
# library(devtools)
# install_github("DougLuke/UserNetR")
library(UserNetR)

library(rgexf)
library(openxlsx)
library(igraph)
library(rgdal)
library(scales)
library(gplots)
library(reshape2)
library(raster)
library(plyr)

# 1. Load data

# Mac Alan 
path.flickr <- "~/Dropbox/KIT/FlickrEU/deepGreen/"

setwd(path.flickr)

proj4.LL <- "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs +towgs84=0,0,0"

proj4.DHDN <- "+proj=tmerc +lat_0=0 +lon_0=12 +k=1 +x_0=4500000 +y_0=0 +ellps=bessel +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs" # epsg:31468


places_all_m = readRDS( "Data/places_all_m.Rds")
imgnet_all_m = readRDS( "Data/imagenet_all_m.Rds")
joint_all_m = readRDS( "Data/joint_all_m.Rds")

tags_l = list(places_all_m, imgnet_all_m, joint_all_m)
dt_names = c("Places365", "IMAGENET", "Hybrid")




for (idx in 1:3) { 
    tags_dt = tags_l[[idx]]
    dt_name = dt_names[[idx]]
    # dimnames(tags.dt)
    tags_v <- colnames(tags_dt)
    tags_m <- (as.matrix(data.frame(tags_dt)))
    dimnames(tags_m) <- list( tags_v, tags_v)
    # total_occur <- diag(tags_m) # wrong?
    total_occur <- colSums(tags_m)
    
    # tags_m[lower.tri(tags_m, diag=T)] <- 0
    max(tags_m) # previously 2460
    
    head(sort(total_occur/sum(total_occur)*100, T))
    
    pdf(paste0("Data/",dt_name, "_frequency.pdf"), width=12, height = 7)
    par(mar=c(10,4,4,4), mfrow=c(1,1))
    barplot(sort(total_occur, decreasing = T)[1:20], las=2, ylab="Weighted occurrence (=Occurrence * Probability)", main = dt_name)
    # barplot(log(sort(total_occur, decreasing = T)[1:50]), ylim=c(10, 15), las=2, ylab="Log weighted occurrence")
    
    
    
    ### Cluster info 
    
    cl_info = read.xlsx(paste0("Data/ClusterInfo_", dt_name, "_gte5.xlsx"))
    
    k_v = 5:20
    par(mfrow=c(2,2))
    
    for (k in k_v) { 
        cl_occur = tapply(total_occur, cl_info[, paste0("Cluster_", k)], FUN = sum)
        cl_occur = cl_occur / sum(cl_occur)
        cl_occur_sorted = sort(cl_occur, decreasing = T)
        barplot(cl_occur_sorted * 100, main = paste0("Cluster Prportion (",  dt_name, ")"), ylab="%", xlab=paste0("Cluster ID (k=", k, ")" ))
    }
    dev.off()
    
}


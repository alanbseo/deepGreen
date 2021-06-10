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


idx = 1 



for (idx in seq_along(dt_names)) { 
    tags_dt = tags_l[[idx]]
    dt_name = dt_names[[idx]]
    # dimnames(tags.dt)
    tags_v <- colnames(tags_dt)
    tags_m <- (as.matrix(data.frame(tags_dt)))
    dimnames(tags_m) <- list( tags_v, tags_v)
    total_occur <- colSums(tags_m)
    
    tags_m[lower.tri(tags_m, diag=T)] <- 0
    max(tags_m) # previously 2460
    
    
    
    
    # 2. Create weighted graph 
    fl_graph <- graph.adjacency(tags_m,
                                weighted=TRUE,
                                mode="undirected",
                                diag=TRUE)
    # Cluster
    cw <-  cluster_walktrap(fl_graph, steps=10, membership = T, weights = E(fl_graph)$weight)  # step: the length of the random walks to perform. 
    
    # plot(sapply(1:300, FUN = function(x)  modularity(fl_graph,  cut_at(cw, no = x))), type="o", xlab= "K", ylab= "Modularity", main = paste0("Modularity changing with K (Walktrap) for ", dt_name))
    # sapply(1:50, FUN = function(x)  modularity(fl_graph,  cut_at(cw, no = x)))
    
    # plot(fl_graph)
    
    pdf(paste0("Data/",dt_name, "_dendrogram.pdf"), width=30, height = 20)
    igraph::plot_dendrogram(cw, cex=0.2, mode = "phylo")
    igraph::plot_dendrogram(cw, cex=0.2, mode = "hclust")
    # igraph::plot_dendrogram(cw, cex=0.3, mode = "dendrogram")
    
    
    
    
    dev.off()
    
    fl_graph <- set_vertex_attr(fl_graph, "cluster_K5", value = cut_at(cw, no = 5))
    fl_graph <- set_vertex_attr(fl_graph, "cluster_K7", value = cut_at(cw, no = 7))
    fl_graph <- set_vertex_attr(fl_graph, "cluster_K9", value = cut_at(cw, no = 9))
    fl_graph <- set_vertex_attr(fl_graph, "cluster_K11", value = cut_at(cw, no = 11))
    fl_graph <- set_vertex_attr(fl_graph, "cluster_K12", value = cut_at(cw, no = 12))
    fl_graph <- set_vertex_attr(fl_graph, "cluster_K14", value = cut_at(cw, no = 14))
    fl_graph <- set_vertex_attr(fl_graph, "cluster_K50", value = cut_at(cw, no = 50))
    
    # write.graph(fl_graph, file = "cw5.dl", format = "pajek")
    
    # g1.gexf <- igraph.to.gexf(fl_graph)
    write.graph(fl_graph, file = paste0("Data/cw_", dt_name, "_gte5.gml"), format = "gml")
    
    
    k_v = 5:20
    cluster_df = sapply(k_v, FUN = function(x) cut_at(cw, no = x))
    
    rownames(cluster_df) = tags_v
    colnames(cluster_df) = paste0("Cluster_", k_v)
    
    write.xlsx(cluster_df, file = paste0("Data/ClusterInfo_", dt_name, "_gte5.xlsx"), row.names=T)
    
    
    write.xlsx(tags_m, file = paste0("Data/CooccurenceMatrix_", dt_name, ".xlsx"), row.names=T, col.names=T)
    
}


stop("ends here")

cw_5 <- igraph::cut_at(cw, no = 5)
table(cw$membership)
table(cw_5)
# igraph::plot_hierarchy(cw_5, fl_graph)
igraph::plot_dendrogram(cw)

# igraph::plot_hierarchy(cw, fl_graph)



# modularity(fl_graph, membership = cw$membership)
# modularity(fl_graph, membership = cw_5)
#  

# cfg <- cluster_fast_greedy(fl_graph)
# system.time(    clb <- cluster_label_prop(fl_graph, weights = E(fl_graph)$weight))
# system.time(    clv <- cluster_louvain(fl_graph, weights = E(fl_graph)$weight))
# system.time(    ci <- cluster_infomap(fl_graph))


plot(fl_graph, col = cw_5)





# plot(cw_5, fl_graph)


# V(fl_graph)$membership <- tmp$membership
# str(vertex_attr(fl_graph))

k_v <- 3:20
n_k <- length(k_v)



algorithm.names <- c( "Walktrap") # , "FastGreedy")
n.algorithm <- length(algorithm.names)


s.l <- s.major.l <-  in.dt.l <- vector("list", n.algorithm) 



for (idx in 1:3) { 
    
    algorithm.l <- list(cw) # , cfg)  # walktrop, groups: 5, mod: 0.26 <-? different from the plot. 
    
    for (al.idx in 1:n.algorithm) {
        
        cluster.tmp <- algorithm.l[[al.idx]]
        al.name <- algorithm.names[al.idx]
        
        s.l[[al.idx]] <- s.major.l[[al.idx]] <-in.dt.l[[al.idx]] <- vector("list", n.k) 
        
        for (k.idx in 1:n.k) {
            
            k.tmp <- k.v[k.idx]
            
            cluster.tmp$membership <- cut_at(cluster.tmp, no = k.tmp)
            lookup.dt.tmp <- data.frame(tags.v, cluster.tmp$membership )
            
            lookup.dt.tmp.by <- by(lookup.dt.tmp$tags.v, INDICES = lookup.dt.tmp$cluster.tmp.membership, FUN = as.character)
            
            lookup.dt.tmp.by.table <- t(rbind.fill( lapply(lookup.dt.tmp.by, FUN = function(x) data.frame(t(x)))))
            colnames(lookup.dt.tmp.by.table) <- paste0("Cluster_", 1:k.tmp)
            
            #write.xlsx(lookup.dt.tmp.by.table, file = paste0("tmp/", al.name, "_tag_clusters_k", k.tmp, ".xlsx"))
            
            
            clusters.dt.tmp <-  apply(rawdata.dt[, ], MARGIN = 1, FUN = function(x) {
                lookup.dt.tmp[match(x[1:20], lookup.dt.tmp$tags.v), 2]
            })
            
            major.clusters.dt.tmp <-  apply(rawdata.dt[, ], MARGIN = 1, FUN = function(x) {
                x.tb <- lookup.dt.tmp[match(x[1:20], lookup.dt.tmp$tags.v), 2]
                # x.tb[is.na(x.tb)] <- "NoCluster"
                x.tb[is.na(x.tb)] <- k.tmp + 1 
                
                names(x.tb) <- x.tb
                x.wt <- x[21:40]
                x.weighted.cluster <- tapply(as.numeric(x.wt), INDEX = x.tb, FUN = sum)
                return(names(x.weighted.cluster)[which.max(x.weighted.cluster)])
            })
            
            
            s.l[[al.idx]][[k.idx]] <- apply(clusters.dt.tmp, MARGIN = 2, FUN = function(x) {x.f <- factor(x, levels = 1:k.tmp, labels = 1:k.tmp); x.t <- table(x.f, useNA = "always"); x.t / sum(x.t)})
            in.dt.l[[al.idx]][[k.idx]]  <- cbind(names(rawdata.dt), rawdata.dt[, ], t(clusters.dt.tmp))
            
            s.major.l[[al.idx]][[k.idx]] <-  major.clusters.dt.tmp
            
            
        }
        
    }
    
}

### 
cw_clusters_num <- do.call(cbind, lapply(3:20, FUN = function(x) cut_at(cw, no = x)))
cw_clusters <- cbind(cw$names, cw_clusters_num)



# cfg.clusters.num <-  do.call(cbind, lapply(3:20, FUN = function(x) cut_at(cfg, no = x)))
# cfg.clusters <- cbind(cfg$names,cfg.clusters.num)


cw.clusters.prop <-  sapply(1:n_k, FUN = function(x) {
    x2 <- numeric(max(k_v)); 
    x2[1:k_v[x]] <- table(cw_clusters_num[,x])/sum(table(cw_clusters_num[,x])) * 100 ; 
    return(x2)}
)

# 
# cfg.clusters.prop <-  sapply(1:n.k, FUN = function(x) {
#     x2 <- numeric(max(k.v)); 
#     x2[1:k.v[x]] <- table(cfg.clusters.num[,x])/sum(table(cfg.clusters.num[,x])) * 100 ; 
#     return(x2)}
# )


write.xlsx(cw_clusters, file = "Data/cw.clusters_3to20.xlsx")


# dummy.sp <- in.sp 
# 
# in.sp <- SpatialPointsDataFrame(cbind(as.numeric(rawdata.dt$Longitude), as.numeric(rawdata.dt$Latitude)), proj4string = CRS(proj4.LL), data = rawdata.dt)
# 
# dummy.250m.r.ll <- projectRaster(usdyav, crs = proj4.LL)
# dummy.250m.r.ll <- crop(dummy.250m.r.ll, in.sp)
# 
# dummy.250m.r.ll <- setValues(dummy.250m.r.ll, 0)
# dummy.1000m.r.ll <- setValues(aggregate(dummy.250m.r.ll, 4, fun = mean), 0)
# dummy.2000m.r.ll <- setValues(aggregate(dummy.250m.r.ll, 8, fun = mean), 0)
# 
# dummy.1000m.p <- (rasterToPolygons(dummy.1000m.r.ll, dissolve = F))
# dummy.2000m.p <- (rasterToPolygons(dummy.2000m.r.ll, dissolve = F))
# 
# 
# mulde.sachsen.p <- readOGR("Data", layer = "Mulde_boundary_Sachsen")
# mulde.sachsen.p.ll <- spTransform(mulde.sachsen.p, CRSobj = proj4.LL)
# 
# srtm <- raster("Data/SRTM_Mulde_DHDN.tif")
# srtm.ll <- projectRaster(srtm, crs = proj4.LL)
# 
# srtm.ll.crop <- crop(srtm.ll, mulde.sachsen.p.ll)
# 
# 
# 
# ## Functions
# getMoltSP <- function(in.dt, k) {
#     
#     molt.dt <- melt(in.dt, id.vars = c(1:11), measure.vars = c(12:31))
#     molt.sp <- SpatialPointsDataFrame(cbind(as.numeric(molt.dt$Longitude), as.numeric(molt.dt$Latitude)), proj4string = CRS(proj4.LL), data = molt.dt)
#     molt.sp$value <- as.numeric(molt.dt$value)
#     molt.sp$color <- rainbow(k)[ molt.dt$value]
#     return(molt.sp)
# }
# 
# save.image(file = paste0( "tmp/flickr_plotting_workspace", Sys.Date(), ".RData")    )
# 
# 
# stop("ends here")




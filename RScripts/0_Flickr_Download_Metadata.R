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



 
workdir <-  "~/Dropbox/KIT/FlickrEU/Costa Rica_Data/FlickrCR_download/"
newsavedir <- paste0("Aug2019_V1") 
 
aoi.poly.in = readOGR( dsn = paste0(workdir, newsavedir), layer = "FlickrCR_AOI")  


 

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

aoi.poly.in = costarica_aoi


# Time span
mindate <- "2005-01-01"
maxdate <- "2019-07-31"
# savedir <- substr(mindate, 6, 10)
savedir <- "Aug2019_V1/"
workdir <- "~/Dropbox/KIT/FlickrEU/Costa Rica_Data/FlickrCR_download/"



# Search parameters
sort <- "interestingness-desc" # Sort by Interestingness (or: relevance)
max.perpage <- 250 # number per page maximum 250
n.points <- length(aoi.poly.in)
# n.points <- 10


# tok = authenticate(api.key, api.secret)
# flickr.tags.getHotList(api.secret, tok, api.key)    
# 
# s$getHotList(verbose = TRUE, format = 'json')
# s$getHotList(verbose = TRUE, .convert = NA)
# s$getHotList(verbose = TRUE, .convert = xmlRoot)    


# current.dt <- read.xlsx("data/Flickr_ClarifaiTags_16May2016_rawdata.xlsx", 1)

# 
# info <- sapply( current.dt$X1, FUN = function(x) strsplit(x,  split = c("_")))
# 
# photoids <- sapply(info, FUN = function(x) x[[3]])
# 




# http://farm6.staticflickr.com/5479/9538302885_221184bec0_z.jpg



# flickreu.nphotos.v <- numeric(length = n.points)
# i <- 1 

max.try = 5
geturl.opts <- list(timeout = 20, maxredirs = max.try, verbose = F)

# # Retreiving the data
# foreach (i = 1:n.points) %dopar% {
#     
#     
#     
#     aoi <- aoi.poly.in[i, ]
#     aoi.bbox <- bbox (aoi)
#     aoi.bbox.txt <- paste(aoi.bbox[1,1], aoi.bbox[2,1], aoi.bbox[1,2], aoi.bbox[2,2], sep=",")
#     
#      
#     api <- paste("https://api.flickr.com/services/rest/?method=flickr.photos.search&format=json&api_key=", api.key, "&nojsoncallback=1&page=1&per_page=", max.perpage, "&bbox=", aoi.bbox.txt, "&min_taken_date=", mindate, "&max_taken_date=", maxdate, "&sort=", sort, "&privacy_filter=", "1",sep="")
#     
#     
#     raw_data.tmp <- getURL(api, ssl.verifypeer = FALSE, followLocation = TRUE, .opts = list(timeout = 10, maxredirs = max.try , verbose = TRUE))
#      data.1st <- fromJSON(raw_data.tmp, unexpected.escape="skip", method="C")
#     cat("poly_id=", i)
#  
#     
#    
#     # # flickr.photos.suggestions.getList
#     # 
#     # id <- "31837518910"
#     # api <- paste("https://api.flickr.com/services/rest/?method=flickr.tags.getListPhoto&format=json&api_key=", api.key, "&nojsoncallback=1&page=1&per_page=", max.perpage, "&photo_id=", id, "&sort=", sort, sep="")
#     # raw_data.tmp <- getURL(api, ssl.verifypeer = FALSE)
#     # data.1st <- fromJSON(raw_data.tmp, unexpected.escape="skip", method="C")
#     # 
#     # data.1st$photo$tags
#     # 
#     
#     if (data.1st$stat == "ok") { 
#         npages <-  data.1st$photos$pages
#         
#         cat(" npages=", data.1st$photos$pages, "\n") 
#         
#         tmp.photo.n <- length(data.1st$photos$photo)
#         
#         if (npages > 1) { 
#             
#             for (p.idx in 2:npages) {
#                 
#                 print(p.idx)
#                 api.tmp <- paste("https://api.flickr.com/services/rest/?method=flickr.photos.search&format=json&api_key=", api.key, "&nojsoncallback=1&page=", p.idx, "&per_page=", max.perpage, "&bbox=", aoi.bbox.txt, "&min_taken_date=", mindate, "&max_taken_date=", maxdate, "&sort=", sort, "&privacy_filter=", "1",sep="") 
#                 # Unlike standard photo queries, geo (or bounding box) queries will only return 250 results per page. 
#                 # https://www.flickr.com/services/api/flickr.photos.search.html
#                 
#                 raw_data.tmp <- getURL(api.tmp, ssl.verifypeer = FALSE)
#                 data.nxt  <- fromJSON(raw_data.tmp, unexpected.escape="skip", method="C")
#                 tmp.photo.n <- tmp.photo.n + length(data.nxt$photos$photo)
#                 
#                 
#                 
#             }  
#         }
#         climsave.nphotos.v[i] <- tmp.photo.n
#         print(sum(climsave.nphotos.v))
#         # print(tmp.photo.n)
#     } else { 
#         print("stat not ok")   
#         print(data.1st$stat)
#     }
#     
#     if ((i %% 100) == 0 ) { 
#         print("gc")
#         gc()
#         
#     }
#     
# }
# 681 poly 50000 photos

print("metadata download start..")

registerDoMC(n.thread)

# Retreiving the data


# target.ids.all <- 1:n.points
target.ids.all <- aoi.poly.in$CELL_ID

aois.done <- list.files(paste0(workdir, "/", savedir, "/Xlsx"), pattern = "^AOI.*.\\.xlsx$")
aois.done.v <- (as.numeric(sapply(aois.done, FUN = function(x)  (str_split(x, pattern = "_")[[1]][3]))))

wantToDeleteDup <- FALSE

if (wantToDeleteDup) {
    tb1 <- (table(aois.done.v))
    aoiids.dup <- names(tb1[tb1>1])
    for (d.idx in 1:length(aoiids.dup)) {
        aoi.probl <- aoiids.dup[d.idx]
        aoi.dup.tmp.v <- aois.done[aois.done.v %in% aoi.probl]
        nphotos.done.tmp.v <- as.numeric(sapply(aoi.dup.tmp.v, FUN = function(x)  str_extract(str_split(x, pattern = "_")[[1]][7], "[0-9]+")))
        largest <- which.max(nphotos.done.tmp.v)
        
        file.copy(from = paste0(workdir, "/", savedir, "/Xlsx/", aoi.dup.tmp.v[-largest]), to = paste0(workdir, "/", aoi.dup.tmp.v[-largest]))
        file.remove(paste0(workdir, "/", savedir, "/Xlsx/", aoi.dup.tmp.v[-largest]))
    }
}

target.ids <-   (setdiff(target.ids.all, aois.done.v))
length(target.ids.all)  - length(target.ids) 
cat(length(target.ids), "to go")

name.machine <- Sys.info()["nodename"]


sink(paste0(workdir, "/logs/", name.machine, "_", Sys.time(), "_output.txt"))	   # Redirect output to the file



# 

# initiateData <- T
# 
# if (initiateData) {
flickrphotos.metadata.list.l <- flickrphotos.metadata.specific.list.l <- flickrphotos.metadata.specific.df.l <-  vector("list", length = n.points)
flickrphotos.metadata.df <- data.frame(polyid=1:n.points, npages=rep(NA, n.points), nallphotos=rep(NA, n.points), ndownloadedphotos=rep(NA, n.points) )
# }


i <- 5 
# i <- 5428
# i <- 8383 # 10368 # RCurl to httr
# i <- 21994 
# i <- 5304


foreach (i = 1:length(target.ids), .errorhandling = "stop", .inorder = F, .verbose = T) %do% {
    
    
    aoi_cellid = target.ids[i]
    
    desc_natpark = aoi.poly.in$NAME[i]
    desc_natpark = ifelse(is.na(desc_natpark), "", paste0( "_NatPark_",desc_natpark))
    aoi.desc <- paste0(aoi.poly.in$NAME_2[i], "", desc_natpark)
    
    
    initiateData <- T
    
    if (initiateData) {
        flickrphotos.metadata.list.l <- flickrphotos.metadata.specific.list.l <- flickrphotos.metadata.specific.df.l <-  vector("list", length = n.points)
        flickrphotos.metadata.df <- data.frame(polyid=1:n.points, npages=rep(NA, n.points), nallphotos=rep(NA, n.points), ndownloadedphotos=rep(NA, n.points) )
    }
    
    
    aois.done <- list.files(paste0(workdir, "/", savedir, "/xls"), pattern = "^AOI.*.\\.xlsx$")
    aois.done.v <- as.numeric(sapply(aois.done, FUN = function(x) (str_split(x, pattern = "_")[[1]][3])))
    
    if (i %in% aois.done.v) { 
        print("skip")
        return(NA)   
    }
    
    
    print(paste0("poly_id=", target.ids[i]))
    
    aoi <- aoi.poly.in[i, ]
    aoi.bbox <- bbox (aoi)
    aoi.bbox.txt <- paste(aoi.bbox[1,1], aoi.bbox[2,1], aoi.bbox[1,2], aoi.bbox[2,2], sep=",")
    
    
    # api <- paste("https://api.flickr.com/services/rest/?method=flickr.photos.search&format=json&api_key=", api.key, "&nojsoncallback=1&page=", i, "&per_page=", max.perpage, "&bbox=", aoi.bbox.txt, "&min_taken_date=", mindate, "&max_taken_date=", maxdate, "&sort=", sort, sep="")
    
    # extras <- c("description, license, date_upload, date_taken, owner_name, icon_server, original_format, last_update, geo, tags, machine_tags, o_dims, views, media, path_alias, url_sq, url_t, url_s, url_q, url_m, url_n, url_z, url_c, url_l, url_o")
    extras <- c("date_taken,owner_name,geo,tags") #,machine_tags") #,url_z")
    
    # api <-     paste("https://api.flickr.com/services/rest/?method=flickr.photos.search&format=json&api_key=", api.key, "&nojsoncallback=1&page=1&per_page=", max.perpage, "&bbox=", aoi.bbox.txt, "&min_taken_date=", mindate, "&max_taken_date=", maxdate, "&sort=", sort, "&privacy_filter=", "1",sep="")
    api <- paste0("https://api.flickr.com/services/rest/?method=flickr.photos.search&format=json&api_key=", api.key, "&nojsoncallback=1&page=1&per_page=", max.perpage, "&bbox=", aoi.bbox.txt, "&min_taken_date=", mindate, "&max_taken_date=", maxdate, "&sort=", sort, "&privacy_filter=", "1", "&extras=", extras)
    
    
    imgdir <- paste(workdir, savedir, "AOI_CELLID_", aoi_cellid, "/", sep="")
    
    
    # raw_data.tmp <- getURL(api, ssl.verifypeer = FALSE, .opts = geturl.opts)
    
    httr.tmp  <- httr::GET(api)
    raw_data.tmp <- content(httr.tmp, "text", encoding = "UTF-8") 
    
    
    data.1st <- fromJSON(raw_data.tmp, unexpected.escape="keep", method="C")
    
    if (data.1st$stat != "ok") { 
        print(paste0("error..", data.1st$stat)   )
        
        stop(paste0("error..", data.1st$stat)   )
        
        # return(NA)
    }
    
    
    npages <-  data.1st$photos$pages
    cat("i=", i, " poly_id=", aoi_cellid, " npages=", npages, "\n")
    
    ### Download metadata
    if ( data.1st$photos$pages <=0 || length(data.1st$photos$photo) ==0) { # e.g., no photos with one (invalid) page
        if (data.1st$stat == "ok") { 
            print("no photos")
            # write.xlsx(metadata.tmp, file = paste0( workdir, savedir,  "/AOI_Poly_", formatC(i, width = 6, flag = "0"), "_metadata_", aoi.desc, "_n", nrow(metadata.tmp), ".xlsx"), overwrite=T)
            write.xlsx(data.frame(NA), file = paste0( workdir, savedir,  "/Xlsx/AOI_CellID_", formatC(aoi_cellid, width = 6, flag = "0"), "_metadata_",aoi.desc, "_n0.xlsx"), overwrite=T)
            
            return(NA)
        } else {
            print(paste0("error..", data.1st$stat)   )
            stop(paste0("error..", data.1st$stat)   )
            
        }
        
    } else {  
        
        
        data.l.tmp <- vector("list", length = npages)
        data.l.tmp[[1]] <- data.1st
        
        if (npages > 1) {
            
            data.l.tmp[2:length(data.l.tmp)]  <- foreach (p.idx = 2:npages) %do% {
                
                cat(p.idx, ">")
                api.tmp <- paste0("https://api.flickr.com/services/rest/?method=flickr.photos.search&format=json&api_key=", api.key, "&nojsoncallback=1&page=", p.idx, "&per_page=", max.perpage, "&bbox=", aoi.bbox.txt, "&min_taken_date=", mindate, "&max_taken_date=", maxdate, "&sort=", sort, "&privacy_filter=", "1", "&extras=", extras)
                
                # Unlike standard photo queries, geo (or bounding box) queries will only return 250 results per page.
                # https://www.flickr.com/services/api/flickr.photos.search.html
                
                # raw_data.tmp <- getURL(api.tmp, ssl.verifypeer = FALSE, .opts = geturl.opts) # , .encoding = "UTF-8", .mapUnicode = T)
                
                raw_data.tmp <- content(httr::GET(api.tmp), "text", encoding = "UTF-8") 
                #    
                # rjson::fromJSON(raw_data.tmp, unexpected.escape="skip", method="R", encoding="UTF-8")
                #   
                #   library(jsonlite)
                #   str(jsonlite::fromJSON(raw_data.tmp, encoding="UTF-8"))
                #       RJSONIO::fromJSON(raw_data.tmp, encoding = "UTF-8")
                
                
                
                # data.l.tmp[[p.idx]] <- fromJSON(raw_data.tmp, unexpected.escape="skip", method="C")
                res.tmp <- fromJSON(raw_data.tmp, unexpected.escape="skip", method="C")
                
                if (res.tmp$stat != "ok") { 
                    print(paste0("error..", res.tmp$stat))
                    stop("error")
                }
                
                return(res.tmp)
                
            }
            
            
            
        } else {
            # do nothing
        }
        
        
        ## Save meta information
        # flickrphotos.metadata.list.l[[i]] <- data.l.tmp
        flickrphotos.metadata.specific.list.l[[i]] <- vector("list", length = npages)
        flickrphotos.metadata.specific.df.l[[i]] <- vector("list", length = npages)
        
        flickrphotos.metadata.df[i, "npages"] <- npages
        
        
        print("photos exist")
        
        for (p.idx in 1:npages) { 
            
            print(paste("page ", p.idx))
            
            
            data.tmp <- data.l.tmp[[p.idx]] #  flickrphotos.metadata.list.l[[i]][[p.idx]]
            # flickrphotos.metadata.df[i, "nallphotos"] <- sum(flickrphotos.metadata.df[i, "nallphotos"] + length(data.tmp$photos$photo), na.rm = T) # todo fix the code
            # print(" flickrphotos.metadata.df[i, nallphotos")
            # print(flickrphotos.metadata.df[i, "nallphotos"])
            
            
            nphotos.tmp <- length(data.tmp$photos$photo)
            print(paste(nphotos.tmp, "photos exist"))
            
            if (nphotos.tmp < 1) { 
                next() # such conditions existed 
                # stop() # stop does not work!! just pass to the next iter
            }
            
            flickrphotos.metadata.specific.list.l[[i]][[p.idx]] <- vector("list", length = nphotos.tmp)
            
            
            flickrphotos.metadata.specific.df.l[[i]][[p.idx]] <- data.frame(PhotoID=rep(NA, nphotos.tmp), Owner=rep(NA, nphotos.tmp), Date =rep(NA, nphotos.tmp), Year=rep(NA, nphotos.tmp), Landcover=rep(NA, nphotos.tmp), Longitude=rep(NA, nphotos.tmp), Latitude=rep(NA, nphotos.tmp), Place_id = rep(NA, nphotos.tmp), Woeid=  rep(NA, nphotos.tmp), Geocontext= rep(NA, nphotos.tmp), LocationAccuracy =rep(NA, nphotos.tmp), DateTakenGranularity = rep(NA, nphotos.tmp), DateTakenUnkown = rep(NA, nphotos.tmp), Title= rep(NA, nphotos.tmp), N_FlickrTag=rep(NA, nphotos.tmp), FlickrTags =rep(NA, nphotos.tmp), Username = rep(NA, nphotos.tmp), Realname = rep(NA, nphotos.tmp), URL = rep(NA, nphotos.tmp))
            
            # [1] "PhotoID"          "Owner"            "Date"             "Year"             "Landcover"       
            # [6] "Longitude"        "Latitude"         "Place_id"         "Woeid"            "Geocontext"      
            # [11] "LocationAccuracy" "Title"            "N_FlickrTag"      "FlickrTags"       "Username"        
            # [16] "Realname"         "URL"   
            # 
            # for (u in 1:nphotos.tmp) {
            
            res.l1 <- foreach (u = 1:nphotos.tmp, .errorhandling = "stop") %do% {
                
                
                info.l <- data.tmp$photos$photo[[u]]
                
                # if (info.l$stat == "ok") { 
                photo.sp <- SpatialPoints(t(as.matrix(as.numeric(c(info.l$longitude, info.l$latitude)))), proj4string = CRS(proj4string(aoi)))
                
                intersectYN <- gIntersects(photo.sp, aoi)
                cat(paste0(">", u, ifelse(intersectYN, "Y",  "N")))
                # plot(photo.sp, add=T, col=ifelse(intersectYN, "green", "red"))
                # 
                if (intersectYN) {
                    
                    # names(info.l)
                    # [1] "id"                   "owner"                "secret"               "server"              
                    # [5] "farm"                 "title"                "ispublic"             "isfriend"            
                    # [9] "isfamily"             "datetaken"            "datetakengranularity" "datetakenunknown"    
                    # [13] "ownername"            "tags"                 "machine_tags"         "latitude"            
                    # [17] "longitude"            "accuracy"             "context"              "place_id"            
                    # [21] "woeid"                "geo_is_family"        "geo_is_friend"        "geo_is_contact"      
                    # [25] "geo_is_public"        "url_z"                "height_z"             "width_z"      
                    # 
                    # names(info.l)
                    # [1] "id"                   "owner"                "secret"               "server"               "farm"                
                    # [6] "title"                "ispublic"             "isfriend"             "isfamily"             "datetaken"           
                    # [11] "datetakengranularity" "datetakenunknown"     "ownername"            "tags"                 "latitude"            
                    # [16] "longitude"            "accuracy"             "context"              "place_id"             "woeid"               
                    # [21] "geo_is_family"        "geo_is_friend"        "geo_is_contact"       "geo_is_public"       
                    # > 
                    #   
                    photo.id <- info.l$id
                    
                    photo.owner <- info.l$owner #   info.l$photo$owner$nsid
                    photo.date <- as.Date.character(info.l$datetaken, tz = "GMT") # info.l$photo$dates$taken
                    photo.year <- substr(photo.date, start = 1, stop = 4)
                    photo.landcover <- NA #  as.character(aoi$LN)
                    photo.longitude <- info.l$longitude
                    photo.latitude <- info.l$latitude
                    photo.datetakengranularity<- info.l$datetakengranularity
                    photo.datetakenunknown<- info.l$datetakenunknown
                    
                    photo.title <- info.l$title
                    photo.accuracy <- info.l$accuracy
                    # Recorded accuracy level of the location information. Current range is 1-16 :
                    # World level is 1
                    # Country is ~3
                    # Region is ~6
                    # City is ~11
                    # Street is ~16
                    
                    
                    names(info.l)
                    
                    photo.tags <- strsplit(info.l$tags, " ")[[1]]
                    photo.ntag <- length(photo.tags)
                    photo.tags_delimited <- paste0(photo.tags, collapse=", ")
                    
                    photo.username <- info.l$ownername  # $owner$username
                    photo.realname <- NA # (info.l$photo$owner$realname)
                    photo.place_id <- info.l$place_id # (info.l$photo$location$place_id)
                    photo.woeid <- info.l$woeid  # (info.l$photo$location$woeid)
                    photo.geocontext <- info.l$context # 0, not defined. 1, indoors. 2, outdoors.
                    
                    # photo.url <- info.l$url_z
                    # 
                    # if (is.null(photo.url) || is.na(photo.url) || photo.url=="") { 
                    #     
                    farm <- info.l$farm
                    server <- info.l$server
                    secret <-info.l$secret
                    photo.url <-  paste("https://farm", farm, ".staticflickr.com/", server, "/", photo.id, "_", info.l$secret, "_z.jpg", sep="")
                    
                    # }
                    
                    
                    
                    
                    # [1] "PhotoID"          "Owner"            "Date"             "Year"             "Landcover"       
                    # [6] "Longitude"        "Latitude"         "Place_id"         "Woeid"            "Geocontext"      
                    # [11] "LocationAccuracy" "Title"            "N_FlickrTag"      "FlickrTags"       "Username"        
                    # [16] "Realname"         "URL"   
                    
                    res.l.tmp <- list(PhotoID = photo.id, Owner = photo.owner, Date = as.character(photo.date), Year = photo.year, Landcover=photo.landcover,  Longitude =  photo.longitude,Latitude =  photo.latitude, Place_id = photo.place_id, Woeid= photo.woeid, Geocontext=photo.geocontext, LocationAccuracy = photo.accuracy, DateTakenGranularity = photo.datetakengranularity,  Datetakenunknown = photo.datetakenunknown, Title= photo.title, N_FlickrTag= photo.ntag, FlickrTags = photo.tags_delimited, Username = photo.username, Realname=photo.realname, URL = photo.url)
                    
                    res.tmp <- data.frame(t(sapply(res.l.tmp, FUN = function(x) ifelse(is.null(x), yes = NA, no = x))))
                    
                    
                    return(  res.tmp )
                    
                    
                }
                
            }
            
            flickrphotos.metadata.specific.df.l[[i]][[p.idx]] <- do.call(rbind, res.l1)
        }
        
        metadata.tmp <- data.frame(do.call(rbind, flickrphotos.metadata.specific.df.l[[i]]))
        metadata.tmp <- metadata.tmp[!is.na(metadata.tmp$PhotoID),]
        
        # print(nrow(metadata.tmp))
        # print(flickrphotos.metadata.df[i, "nallphotos"] )
        
        # stopifnot(nrow(metadata.tmp)== flickrphotos.metadata.df[i, "nallphotos"] )
        
        saveRDS(metadata.tmp, file = paste0( workdir, savedir,  "Rds/AOI_CellID_", formatC(aoi_cellid, width = 6, flag = "0"), "_metadata_", aoi.desc, "_n", nrow(metadata.tmp), ".Rds"))
        
        
        # metadata.tmp$FlickrTags <- iconv(metadata.tmp$FlickrTags, from="UTF-8", to="ASCII", sub="")
        metadata.tmp$Title <- iconv(metadata.tmp$Title, from="UTF-8", to="ASCII", sub="")
        
        
        write.xlsx(metadata.tmp, file = paste0( workdir, savedir,  "Xlsx/AOI_CellID_", formatC(aoi_cellid, width = 6, flag = "0"), "_metadata_", aoi.desc, "_n", nrow(metadata.tmp), ".xlsx"), overwrite=T, )
        
        
        
        # a <- read.xlsx("FlickrEU_download/May2018_V2/AOI_Poly_006410_metadata_FI18_Boreal_n7250.xlsx", 1)
        
    }
    
    
    if ((i %%300) ==0) {
        # print (paste0 ("Saving temporary results by ", i, ">"))
        # save.image( file =paste0( "tmp/FlickrEU_temp_workspace_by_", i, "_",Sys.time(), ".RData"))
        
        print("gc")
        gc()
    }
}

sink()				   # close the file output.txt


# save.image(paste0(workdir, savedir, "/Flickr_CR_workspace_metadata_download_17Aug2019.RData"))
load(paste0(workdir, savedir, "/Flickr_CR_workspace_metadata_download_17Aug2019_2.RData"))
stop("ends here")

 






# Retreiving the data
 

aois.done <- list.files(paste0(workdir, "/", savedir, "/Xlsx"), pattern = "^AOI.*.\\.xlsx$")
aois.done.v <- as.numeric(sapply(aois.done, FUN = function(x)  (str_split(x, pattern = "_")[[1]][3])))


nphotos.done.v <- as.numeric(sapply(aois.done, FUN = function(x)  str_extract(str_split(x, pattern = "_")[[1]][6], "[0-9]+")))
nphotos.done.v2 <- as.numeric(sapply(aois.done, FUN = function(x)  str_extract(str_split(x, pattern = "_")[[1]][8], "[0-9]+")))

nphotos.done.v[is.na(nphotos.done.v)] = nphotos.done.v2[is.na(nphotos.done.v)]


nphotos.done.reduced.v <- numeric(length = n.points) 
 

# 14297, 14732, 15044
library(readxl)

for (i in 1:n.points) { 
    cat(i)
    aoi.tmp <- aois.done[i]
    # Sys.setlocale("UTF-8")
    tryCatch(
        aoi.dt.raw <- data.frame(read.xlsx(paste0(workdir, savedir, "/Xlsx/", aoi.tmp), sheet = 1)) 
    )      
    
    # tryCatch(stop("fred"),  error = function(e) e, finally = print("Hello"))
    # withCallingHandlers({ warning("A"); 1+2 }, warning = function(w) {})
    
    # aoi.dt.raw <- data.frame(read_excel(paste0(workdir, savedir, aoi.tmp),sheet = 1))
    
    photo.ids.unique <- as.numeric( unique( aoi.dt.raw$PhotoID)  )
    # # aoi.dt <- aoi.dt.raw[match(photo.ids.unique, aoi.dt.raw$PhotoID ),]
    # # nr <- max(0, nrow(aoi.dt))
    # if (length(unique(aoi.dt.raw$PhotoID))< nrow(aoi.dt.raw)) { 
    #     print(length(unique(aoi.dt.raw$PhotoID)))
    #     print(nrow(aoi.dt.raw))
    #     
    #     table(sapply(aoi.dt.raw$PhotoID, FUN = function(x) table(match(x, unique(aoi.dt.raw$PhotoID)))>1))
    #     print(head(aoi.dt.raw[!(aoi.dt.raw$PhotoID %in%  ),]))
    #     stop()
    # }
    nphotos.done.reduced.v[i] <- max(0, length(photo.ids.unique))
}


summary(nphotos.done.reduced.v)







dt = readOGR("../Costa Rica_Data/FlickrCR_download/Aug2019_V1/FlickrCR_AOI.shp")
dt$NPHOTOS = nphotos.done.v
dt$NPHOTOS_UNIQUE = nphotos.done.reduced.v

writeOGR(dt, dsn = "../Costa Rica_Data/FlickrCR_download/Aug2019_V1", layer = "FlickrC_AOI_Nphotos.shp", driver = "ESRI Shapefile")

spplot(dt, "NPHOTOS_UNIQUE")






plot(costarica_ext, xlab= "Lon", ylab="Lat")

plot(costarica_ext, xlab= "Lon", ylab="Lat", border=NA, col=NA)


nphotos_perct = quantile(dt$NPHOTOS_UNIQUE, probs= seq(0, 1, 0.1))
nphotos_perct = unique(nphotos_perct)

nphotos_cut = cut(dt$NPHOTOS_UNIQUE, breaks= as.numeric(nphotos_perct))


# levels(nphotos_cut)
aoi_col = rev(topo.colors(length(nphotos_perct)))[as.numeric(nphotos_cut)]
plot(costarica_ext, xlab= "Lon", ylab="Lat", border=NA, col=NA)

plot(dt, col = aoi_col, add=T, border="grey")
plot(costarica_adm2_ll, add=T, col=NA, border="grey")
plot(costarica_natpark, add=T, col=NA, border='red')

legend("bottomright", title = "# of Flickr photos", legend = levels(nphotos_cut), col = rev(topo.colors(length(nphotos_perct))), pch=15, bty="n")

# plot(costarica_aoi, add=T)








options(warn=2)
# options(warn=1)


processFlickrList <- function(x) {
    print(x)
    
    
    xin <- flickrphotos.metadata.specific.df.l[[x]]
    plen <- length(xin) 
    
    if (plen > 1 ) {
        # x3.l <- lapply(xin, FUN = function(x2) (
        #     if (is.null(x2) || nrow(x2)==1 ) {
        #         return(x2)
        #     } else {
        #         return(x2)
        #         
        #         # return(do.call(cbind, x2))
        #     }
        # ))
        
        x3.l <- xin[!sapply(xin, FUN = is.null)]
        x3 <- do.call(rbind, x3.l)
        
        
    } else if (plen == 1) {
        x3 <- do.call(cbind, xin)
    } else {
        return(NULL)   
    }
    
    
    if (is.null(x3) || (nrow(x3) == 0 )) { 
        return (NULL) 
    } else {
        x4 <- x3[!is.na(x3[,1]),]
        
        return (x4)
    }
    
    
}

flickrphotos.metadata.specific.df.l.df <-lapply(1:n.points, FUN = processFlickrList)
# processFlickrList(5812)
# processFlickrList(995)
# processFlickrList(490)
# processFlickrList(3278)
# 
# processFlickrList(1046)

poly.pages.idx <- which(sapply(flickrphotos.metadata.specific.df.l.df, FUN = function(x) !is.null(x)))

str(flickrphotos.metadata.specific.final.df <- do.call(rbind, flickrphotos.metadata.specific.df.l.df[poly.pages.idx]))
length(table(unique(flickrphotos.metadata.specific.final.df$PhotoID)))
length(( (flickrphotos.metadata.specific.final.df$PhotoID)))

which.max(sort(table( (flickrphotos.metadata.specific.final.df$PhotoID)), T))
sort(table((flickrphotos.metadata.specific.final.df$PhotoID)), T)

# flickrphotos.metadata.specific.final.df[(flickrphotos.metadata.specific.final.df$PhotoID == 31573925952), ]

unique.photoids <- (unique(flickrphotos.metadata.specific.final.df$PhotoID))

flickrphotos.metadata.specific.final.df.unique <- flickrphotos.metadata.specific.final.df[!(duplicated(flickrphotos.metadata.specific.final.df$PhotoID)),]
summary(table(unique(flickrphotos.metadata.specific.final.df.unique$PhotoID)))

unique.photoids

# 
load(file = "Rdata/clarifai_final_workspace_by_12635_2017-01-02.RData")

flickr.final.df <- read.xlsx(xlsxFile  = "Data/Flickr_ClarifaiTags_2017-01-09.xlsx", sheet =  1)



table(flickr.final.df[,2] %in% unique.photoids)
# photoids.overlap.idx <- match(flickr.final.df[,2], unique.photoids)
# photoids.overlap.rev.idx <- match( unique.photoids, flickr.final.df[,2])

# yesinfo.idx <- which(!is.na(photoids.overlap.idx ) )
# noinfo.idx <- which(is.na(photoids.overlap.idx ) )

# photoids.overlap.idx[!is.na(photoids.overlap.idx)]


flickr.final.df.merged <- merge(x = flickr.final.df, y = flickrphotos.metadata.specific.final.df.unique[, c("PhotoID", "Latitude", "Longitude", "Ntag")], all.x = TRUE, by.x = "Photo_ID", by.y = "PhotoID")

str(flickr.final.df.merged)

flickr.final.df.merged$Poly_ID <- as.numeric(as.character(flickr.final.df.merged$Poly_ID))

flickr.final.df.merged$Photo_ID <- as.numeric(as.character(flickr.final.df.merged$Photo_ID))
flickr.final.df.merged$Poly_ID <- as.numeric(as.character(flickr.final.df.merged$Poly_ID))

flickr.final.df.merged$Year <- as.numeric(as.character(flickr.final.df.merged$Year))
flickr.final.df.merged$Date <-  (as.character(flickr.final.df.merged$Date))

flickr.final.df.merged$Latitude <- as.numeric(as.character(flickr.final.df.merged$Latitude))
flickr.final.df.merged$Longitude <- as.numeric(as.character(flickr.final.df.merged$Longitude))
flickr.final.df.merged$Ntag <- as.numeric(as.character(flickr.final.df.merged$Ntag))



noinfo.idx <- which(is.na(flickr.final.df.merged$Latitude))
flickr.final.df.merged[noinfo.idx,]




noinfo.photoids <- flickr.final.df.merged[noinfo.idx, "Photo_ID"] 


library(rjson)
library(RCurl)

n.noinfo.photos <- length(noinfo.photoids)

ntag <- longitude <- latitude <- numeric(n.noinfo.photos)

flickr.photo.noinfo.l <- vector("list", n.noinfo.photos)

for (i in 1:n.noinfo.photos) {
    cat(i,",")
    id <- noinfo.photoids[i]
    query.info <- paste("https://api.flickr.com/services/rest/?method=flickr.photos.getInfo&format=json&api_key=", api.key, "&nojsoncallback=1&photo_id=", id, sep="")
    
    info.json <- getURL(query.info, ssl.verifypeer = FALSE, .opts = geturl.opts)
    info.l <- fromJSON(info.json, unexpected.escape="skip", method="C")
    
    
    if (info.l$stat == "ok") {
        longitude[i]  <-  info.l$photo$location$longitude
        latitude[i]  <-  info.l$photo$location$latitude
        ntag[i] <- length(info.l$photo$tags$tag)
    }
    
    flickr.photo.noinfo.l[[i]] <- info.l
    
}


flickr.final.df.merged[noinfo.idx, c("Latitude", "Longitude", "Ntag")] <- cbind(latitude, longitude, ntag)




flickr.final.df.merged$Landcover <- aoi.poly.in$LN[flickr.final.df.merged$Poly_ID]

table(flickr.final.df.merged$Landcover)


library(rgeos)
aoi.poly.in$AreaKM2 <- gArea(lu.mulde.sachsen.nonurban, byid = T) / 1E6

flickr.final.df.merged$AreaKM2 <- aoi.poly.in$AreaKM2[flickr.final.df.merged$Poly_ID]


#  
# 
write.xlsx(flickr.final.df.merged, file = paste0("Data/Flickr_Clarifai_Final_Results_n12635_", Sys.Date(), ".xlsx"))
# 



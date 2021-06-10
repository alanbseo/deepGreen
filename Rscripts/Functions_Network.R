
## Functions
getMoltSP <- function(in.dt, k) {
    
    molt.dt <- melt(in.dt, id.vars = c(1:11), measure.vars = c(12:31))
    molt.sp <- SpatialPointsDataFrame(cbind(as.numeric(molt.dt$Longitude), as.numeric(molt.dt$Latitude)), proj4string = CRS(proj4.LL), data = molt.dt)
    molt.sp$value <- as.numeric(molt.dt$value)
    molt.sp$color <- rainbow(k)[ molt.dt$value]
    return(molt.sp)
}


prepareUCINETinput <- function(keyword.reclass, data.in, n.threshold, name.project) {
    
    colnames(keyword.reclass) <- c("Org", "Grouped")
    
    
    # n.keywords.cases <- apply(data.in, MARGIN = 1, FUN = function(x) length(which(!is.na(x))))
    # n.keywordmax <- max(n.keywords.cases)
    n.keywordmax = ncol(data.in)
    keyword.m <- data.in[, 1:n.keywordmax]
    
    print("Keywords before cleaning:")
    
    print(sort(table(unlist(keyword.m)), decreasing = T))
    cat("Total", length(table(unlist(keyword.m))), "keywords")
    
    keyword.in.reduced <- (apply(keyword.m, 1, FUN = function(x) keyword.reclass[match(x, keyword.reclass[,"Org"]), "Grouped"]))
    
    
    # Delete keywords based on the frequency of the keywords
    keywords.toDelete <- names(table(keyword.in.reduced))[table(keyword.in.reduced) <= n.threshold]
    keyword.in.reduced[keyword.in.reduced %in% keywords.toDelete ] <- NA
    
    
    # Create columns
    keywords.occurred <- sort(unique(as.vector(keyword.in.reduced)))
    n.keyword.occurred <- length(keywords.occurred)
    
    
    
    print("Keywords after cleaning:")
    # Frequency table of the reduced keyword groups
    print(sort(table(keyword.in.reduced), decreasing = T))
    cat("Total", length(table(keyword.in.reduced)), "keywords")
    
    # ifelse(!is.na(match(g2.keywords.kor, keyword.new[3,])), yes = "Y", no = "N")
    # res.g2 <- apply(keyword.in.reduced, MARGIN = 1, function(x) t(ifelse(!is.na(match(g2.keywords.kor, x)), yes = "Y", no = "N")))
    # apply(res.g2, MARGIN = 1, FUN = table)
    
    make1modematrix <- function(keyword.vec.in, keywords.vec.ref) {
        matched <- match(keyword.vec.in, keywords.vec.ref)
        # matched.nona <- matched[!is.na(matched )]
        
        matched.tb <- table(matched)
        
        res.tmp <- numeric(length(keywords.vec.ref))
        res.tmp[as.numeric(names(matched.tb))] <-  (matched.tb)
        return(res.tmp)
    }
    
    print("Making the 1-mode matrix")
    
    # 
    res.1modematrix <- apply(keyword.in.reduced, MARGIN = 2, function(x) make1modematrix(x, keywords.occurred))
    # apply(res.1modematrix, MARGIN = 1, FUN = table)
    
    
    makefinalmatrix <- function(matrix.res, keywords.vec.ref) {
        
        final.m <- matrix(NA, nrow = length(keywords.vec.ref), ncol =  length(keywords.vec.ref))
        
        for (i in 1:length(keywords.vec.ref)) { 
            
            for (j in 1:length(keywords.vec.ref)) { 
                in.tmp <- matrix.res[i,] 
                in.tmp2 <- matrix.res[j,] 
                in.comb <- cbind(in.tmp, in.tmp2)
                cooccur.idx <- which(in.tmp > 0 & in.tmp2 > 0)
                final.m[i,j] <- sum(in.comb[cooccur.idx, 1] * in.comb[cooccur.idx, 2])
            }
            
            diag(final.m) <- rowSums(matrix.res)
            
        }
        colnames(final.m) <- rownames(final.m) <- paste0("K", 1:length(keywords.vec.ref))
        return(final.m)
    }
    
    
    print("Making the final matrix")
    
    final.df <- final.df.char <-  makefinalmatrix(res.1modematrix, keywords.occurred)
    
    # table(keyword.in.reduced)
    
    colnames(final.df.char) <- rownames(final.df.char) <- keywords.occurred
    
    print("Write xls files")
    
    write.xlsx(final.df, file = paste0("PeatlandData/Peatland_ID_sym_K", n.keyword.occurred, "_", Sys.Date(), ".xlsx"), row.names=T )
    write.xlsx(final.df.char, file = paste0("PeatlandData/Peatland_Char_sym_K", n.keyword.occurred, "_", Sys.Date(), ".xlsx"), row.names=T)
    
    look.tb <- cbind( paste0("K", 1:n.keyword.occurred), keywords.occurred)
    colnames(look.tb) <- c("Keyword_ID", "Keyword_Char")
    
    write.xlsx(look.tb, file = paste0("PeatlandData/Peatland_Table_K", n.keyword.occurred, "_", Sys.Date(), ".xlsx"),row.names = F)
    
}
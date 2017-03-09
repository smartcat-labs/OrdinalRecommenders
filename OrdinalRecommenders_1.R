
### --------------------------------------------------
### --- Recommender Systems: Ordinal Logistic Regression
### --- Goran S. Milovanović, PhD
### --- Data Kolektiv, Belgrade, Serbia
### --- Developed for: SmartCat, Novi Sad, Serbia
### --- 25 February 2017.
### --- MovieLens 100K Data Set
### --------------------------------------------------

### --------------------------------------------------
### --- The MovieLens 100K Dataset:
### --- F. Maxwell Harper and Joseph A. Konstan. 2015. 
### --- The MovieLens Datasets: History and Context. 
### --- ACM Transactions on Interactive Intelligent Systems
### --- (TiiS) 5, 4, Article 19 (December 2015), 19 pages. 
### --- DOI=http://dx.doi.org/10.1145/2827872
### --------------------------------------------------

### --------------------------------------------------
### --- Part 1A: Import Data + export CSV
### --------------------------------------------------

rm(list = ls())
library(readr)
library(dplyr)
library(tidyr)
library(Matrix)
library(text2vec)

### --- ratings data
setwd('./data100K')
ratingsData <- read_delim('u.data',
                          col_names = F,
                          delim = '\t')
colnames(ratingsData) <- c('UserID', 'MovieID','Rating', 'Timestamp')
setwd('../outputs100K')
write_csv(ratingsData, 
          path = paste0(getwd(),'/ratings.csv'), 
          append = F, col_names = T)
### --- user data
setwd('../data100K')
usersData <- read_delim('u.user',
                        col_names = F,
                        delim = '|',
                        col_types = list(col_integer(), col_integer(), 
                                         col_character(), col_character(),
                                         col_character()))
colnames(usersData) <- c('UserID', 'Age', 'Gender', 'Occupation', 'Zip-code')
setwd('../outputs100K')
write_csv(usersData, 
          path = paste0(getwd(),'/users.csv'), 
          append = F, col_names = T)
### --- movies data
setwd('../data100K')
moviesData <- read_delim('u.item',
                        col_names = F,
                        delim = '|')
moviesData$X4 <- NULL
moviesData$X5 <- NULL
colnames(moviesData) <- c('MovieID', 'Title', 'Date',  
                          'unknown', 'Action', 'Adventure' , 'Animation', 
                          'Children\'s', 'Comedy', 'Crime', 'Documentary', 
                          'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                          'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western')
setwd('../outputs100K')
write_csv(moviesData, 
          path = paste0(getwd(),'/movies.csv'), 
          append = F, col_names = T)

### --------------------------------------------------
### --- Part 1B: Feature Engineering
### --------------------------------------------------

rm(list = ls())
### --- load data
ratingsData <- read_csv("ratings.csv",
                        col_names = T)
usersData <- read_csv("users.csv",
                      col_names = T)
moviesData <- read_csv("movies.csv",
                       col_names = T)

### -- clean-up a bit (one movie is 'unknown'):
w <- which(moviesData$Title == 'unknown')
unknownID <- moviesData$MovieID[w]
# - fix for the 'unknown' movie in moviesData
moviesData <- moviesData[-unknownID, ]
# - fix ID numbers after removing the 'unknown' movie
moviesData$MovieID[moviesData$MovieID > unknownID] <-
  moviesData$MovieID[moviesData$MovieID > unknownID] - 1
# - fix for the 'unknown' movie in ratingsData
w <- which(ratingsData$MovieID == unknownID)
ratingsData <- ratingsData[-w, ]
# - fix ID numbers after removing the 'unknown' movie in ratingsData
ratingsData$MovieID[ratingsData$MovieID > unknownID] <- 
  ratingsData$MovieID[ratingsData$MovieID > unknownID] - 1
# - save ratingsData without the 'unknow' movie: model version
write.csv(ratingsData, "ratingsData_Model.csv")

### --- Compute moviesDistance w. Jaccard {text2vec} from movie genres
moviesData <- moviesData %>% separate(col = Date, 
                                      into = c('Day', 'Month','Year'), 
                                      sep = "-")
moviesData$Day <- NULL
moviesData$Month <- NULL
moviesData$Year <- as.numeric(moviesData$Year)
range(moviesData$Year)
# - that would be: [1] 1922 1998
# - Introduce Movie Decade in place of Year:
decadeBoundary <- seq(1920, 2000, by = 10)
moviesData$Year <- sapply(moviesData$Year, function(x) {
  wL <- x < decadeBoundary
  wU <- x >= decadeBoundary
  if (sum(wL) == length(decadeBoundary))  {
    return(1)
  } else if (sum(wU) == length(decadeBoundary)) {
    decadeBoundary[length(decadeBoundary)] 
  } else {
    decadeBoundary[max(which(wL-wU == -1))]
  }
}) 
# - Match moviesData$Year with ratingsData:
mD <- moviesData %>% 
  select(MovieID, Year)
ratingsData <- merge(ratingsData, mD,
                     by = 'MovieID')
# - Movie Year (now Decade) as binary: 
moviesData <- moviesData %>%
  spread(key = Year,
         value = Year,
         fill = 0,
         sep = "_")
# - compute moviesDistance:
moviesDistance <- moviesData[, 3:ncol(moviesData)]
w <- which(moviesDistance > 0, arr.ind = T)
moviesDistance[w] <- 1
moviesDistance <- dist2(Matrix(as.matrix(moviesData[, 4:ncol(moviesData)])), 
                        method = "jaccard")
moviesDistance <- as.matrix(moviesDistance)
rm(moviesData); gc()
# - save objects and clear:
numMovies <- length(unique(ratingsData$MovieID))
write_csv(as.data.frame(moviesDistance), 
          path = paste0(getwd(),'/moviesDistance.csv'), 
          append = F, col_names = T)
rm(moviesDistance); gc()
### --- produce binary User-Item Matrix (who rated what only):
userItemMat <- matrix(rep(0, dim(usersData)[1]*numMovies),
                      nrow = dim(usersData)[1],
                      ncol = numMovies)
userItemMat[as.matrix(ratingsData[c('UserID', 'MovieID')])] <- 1
rm('w', 'ratingsData', 'usersData'); gc()
### --- Compute userDistance w. Jaccard {text2vec}
userItemMat <- Matrix(userItemMat)
usersDistance <- dist2(userItemMat, 
                       method = "jaccard")
rm(userItemMat); gc()
usersDistance <- as.matrix(usersDistance)
write_csv(as.data.frame(usersDistance), 
          path = paste0(getwd(),'/usersDistance.csv'), 
          append = F, col_names = T)
rm(usersDistance); gc()

### --- Compute User-User and Item-Item Ratings Similarity Matrices
ratingsData <- read_csv("ratingsData_Model.csv",
                        col_names = T)
ratingsData$X1 <- NULL
# - User-Item Ratings Matrix
ratingsData$Timestamp <- NULL
ratingsData <- ratingsData %>% 
  spread(key = MovieID,
         value = Rating, 
         sep = "_") %>% 
  arrange(UserID)
# - Pearson Correlations: User-User Sim Matrix
UserUserSim <- ratingsData %>% 
  select(starts_with("Movie"))
UserUserSim <- t(UserUserSim)
UserUserSim <- cor(UserUserSim, 
                   use = 'pairwise.complete.obs')
UserUserSim <- as.data.frame(UserUserSim)
write_csv(UserUserSim, 
          path = paste0(getwd(),'/UserUserSim.csv'), 
          append = F, col_names = T)
rm(UserUserSim); gc()
# - Pearson Correlations: Item-Item Sim Matrix
ItemItemSim <- ratingsData %>% 
  select(starts_with("Movie"))
rm(ratingsData); gc()
ItemItemSim <- cor(ItemItemSim,
                   use = 'pairwise.complete.obs')
ItemItemSim <- as.data.frame(as.matrix(ItemItemSim))
write_csv(ItemItemSim, 
          path = paste0(getwd(),'/ItemItemSim.csv'), 
          append = F, col_names = T)
rm(ItemItemSim); gc()

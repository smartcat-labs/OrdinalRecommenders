
### --------------------------------------------------
### --- Recommender Systems: Feature Engineering Exps
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
### --- Part 2: Prepare Data for Cumulative Logit Models
### --- No imputation
### --- Store 100 nearest neighbours from
### --- distance and similarity matrices
### --------------------------------------------------

rm(list = ls())
library(dplyr)
setwd('./outputs100K')

### --- load data
usersDistance <- as.matrix(read.csv('usersDistance.csv',
                                    header = T))
diag(usersDistance) <- 1 
itemsDistance <- as.matrix(read.csv('moviesDistance.csv',
                                    header = T))
diag(itemsDistance) <- 1 
UserUserSim <- as.matrix(read.csv('UserUserSim.csv',
                                  header = T))
diag(UserUserSim) <- 0
ItemItemSim <- as.matrix(read.csv('ItemItemSim.csv',
                                  header = T))
diag(ItemItemSim) <- 0
ratingsData <- read_csv("ratingsData_Model.csv",
                        col_names = T)
ratingsData$X1 <- NULL
ratingsData$Timestamp <- NULL

### --- prepare modelFrame data set:
# - numSims constant: number of nearest neighbours
# - from similarity/distance matrices over users and items used:
numSims <- 100
proxUsersRatingsFrame <- matrix(rep(0, dim(ratingsData)[1]*numSims*2),
                                nrow = dim(ratingsData)[1],
                                ncol = numSims*2)
simUsersRatingsFrame <- matrix(rep(0, dim(ratingsData)[1]*numSims*2),
                               nrow = dim(ratingsData)[1],
                               ncol = numSims*2)
proxItemsRatingsFrame <- matrix(rep(0, dim(ratingsData)[1]*numSims*2),
                                nrow = dim(ratingsData)[1],
                                ncol = numSims*2)
simItemsRatingsFrame <- matrix(rep(0, dim(ratingsData)[1]*numSims*2),
                               nrow = dim(ratingsData)[1],
                               ncol = numSims*2)

for (i in 1:length(ratingsData$Rating)) {
  # - current user ID:
  uID <- ratingsData$UserID[i]
  # - current movie ID:
  mID <- ratingsData$MovieID[i]
  # - select only mID data from ratingsData for Parts 1 and 2:
  rData <- ratingsData %>% 
    filter(MovieID == mID)
  ### --- Part 1: most proximal users + their ratings of the same movie
  ### --- Part 1: use usersDistance matrix
  # - most proximal users from usersDistance:
  proxUsers <- order(usersDistance[uID, ], decreasing = F)
  # - keep only those proxUsers who have rated the movie mID:
  wRated <- which(proxUsers %in% rData$UserID)
  proxUsers <- proxUsers[wRated]
  # - select only a numSims number of proxUsers
  proxUsers <- proxUsers[1:numSims] 
  # - get their ratings of the movie miD -- OUTPUT 1A
  proxUsersRatings <- rData$Rating[match(proxUsers, rData$UserID)][1:numSims]
  # - get their distances from the user uID: -- OUTPUT 1B
  proxUserDistances <- usersDistance[uID, ][proxUsers]
  # - store outputs from Part 1: proxUsersRatings, proxUserDistances
  proxUsersRatingsFrame[i, ] <- c(proxUsersRatings, proxUserDistances)
  ### --- Part 2: most similar users + their ratings of the same movie
  ### --- Part 2: use UserUserSim matrix
  # - most similar users from UserUserSim
  simUsers <- order(UserUserSim[uID, ], decreasing = T)
  # - keep only those simUsers who have rated movie mID:
  wRated <- which(simUsers %in% rData$UserID)
  simUsers <- simUsers[wRated]
  # - select only numSims number of simUsers
  simUsers <- simUsers[1:numSims] 
  # - get their ratings of the movie miD -- OUTPUT 2A
  simUsersRatings <- rData$Rating[match(simUsers, rData$UserID)][1:numSims]
  # - get their similarities from the user uID: -- OUTPUT 2B
  simUserSimilarities <- UserUserSim[uID, ][simUsers]
  # - store outputs 2A, 2B
  simUsersRatingsFrame[i, ] <- c(simUsersRatings, simUserSimilarities)
  # - select only uID data from ratingsData for Parts 1/2
  rData <- ratingsData %>%
    filter(UserID == uID)
  ### --- Part 3: most proximal items + their ratings from the user uID
  ### --- Part 3: use itemsDistance matrix
  # - most proximal items from itemsDistance
  proxItems <- order(itemsDistance[mID, ], decreasing = F)
  # - which of proxItems was rated by user uID:
  wRated <- which(proxItems %in% rData$MovieID)
  proxItems <- proxItems[wRated]
  # - select only numSims number of proxItems
  proxItems <- proxItems[1:numSims] 
  # - get their ratings from the user uiD -- OUTPUT 3A
  proxItemsRatings <- rData$Rating[match(proxItems, rData$MovieID)][1:numSims]
  # - get their distances from the movie uID: -- OUTPUT 3B
  proxItemsDistances <- itemsDistance[mID, ][proxItems]
  # - store outputs 3A, 3B
  proxItemsRatingsFrame[i, ] <- c(proxItemsRatings, proxItemsDistances)
  ### --- Part 4: most similar items + their ratings from the user uID
  ### --- Part 4: use ItemItemSim matrix
  # - most similar items from ItemItemSim
  simItems <- order(ItemItemSim[mID, ], decreasing = T)
  # - which of simItems was rated by user uID:
  wRated <- which(simItems %in% rData$MovieID)
  simItems <- simItems[wRated]
  # - select only numSims number of proxItems
  simItems <- simItems[1:numSims] 
  # - get their ratings from the user uiD -- OUTPUT 4A
  simItemsRatings <- rData$Rating[match(simItems, rData$MovieID)][1:numSims]
  # - get their similarities from the movie uID: -- OUTPUT 4B
  simItemsSimilarities <- ItemItemSim[mID, ][simItems]
  # - store outputs 4A, 4B
  simItemsRatingsFrame[i, ] <- c(simItemsRatings, simItemsSimilarities)
  # - report:
  print(paste0("Completed: ", i, "-th rating out of: ", length(ratingsData$Rating)))
}
# - outputs as.data.frames:
proxUsersRatingsFrame <- as.data.frame(proxUsersRatingsFrame)
colnames(proxUsersRatingsFrame) <- c(paste0("proxUsersRatings_",1:numSims),
                                     paste0("proxUserDistances_",1:numSims))
simUsersRatingsFrame <- as.data.frame(simUsersRatingsFrame)
colnames(simUsersRatingsFrame) <- c(paste0("simUsersRatings_",1:numSims),
                                    paste0("simUserSimilarities_",1:numSims))
proxItemsRatingsFrame <- as.data.frame(proxItemsRatingsFrame)
colnames(proxItemsRatingsFrame) <- c(paste0("proxItemsRatings_",1:numSims),
                                     paste0("proxItemsDistances_",1:numSims))
simItemsRatingsFrame <- as.data.frame(simItemsRatingsFrame)
colnames(simItemsRatingsFrame) <- c(paste0("simItemsRatings_",1:numSims),
                                    paste0("simItemsSimilarities_",1:numSims)
)
# - save feature outputs
write.csv(proxUsersRatingsFrame, 
          paste0('proxUsersRatingsFrame', numSims, '_Feat.csv'))
write.csv(simUsersRatingsFrame, 
          paste0('simUsersRatingsFrame', numSims, '_Feat.csv'))
write.csv(proxItemsRatingsFrame, 
          paste0('proxItemsRatingsFrame', numSims, '_Feat.csv'))
write.csv(simItemsRatingsFrame, 
          paste0('simItemsRatingsFrame', numSims, '_Feat.csv'))


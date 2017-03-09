
### --- Illustration
user1 <- c(5, NA, NA, 4, 3, NA, 1, NA, NA, NA)
user2 <- c(4, NA, 3, 3, 2, NA, NA, 1, 2, 4)
# -- User1 and User2 have a high Pearson Correlation Coefficient:
cor(user1, user2, use = "pairwise.complete.obs")
user3 <- c(3, NA, 1, 5, 4, NA, NA, 2, 5, 4)
# -- User2 and User3 have a low Pearson Correlation Coefficient:
cor(user2, user3, use = "pairwise.complete.obs")
# -- User2 and User3 have a high Jaccard similarity
# -- i.e. 1 minus the Jaccard distance between them:
library(proxy)
as.numeric(1 - dist(t(user2), t(user3), method = "jaccard"))
as.numeric(1 - dist(t(user1), t(user2), method = "jaccard"))
as.numeric(1 - dist(t(user1), t(user3), method = "jaccard"))

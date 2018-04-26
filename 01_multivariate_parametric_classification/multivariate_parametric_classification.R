library(MASS)

set.seed(420)
# mean parameters
class_means <- matrix(c(+0.0, +1.5,
                        -2.5, -3.0,
                        +2.5, -3.0), 2, 3)
# covariance parameters
class_covariances <- array(c(+1.0, +0.2, +0.2, +3.2,
                             +1.6, -0.8, -0.8, +1.0,
                             +1.6, +0.8, +0.8, +1.0), c(2, 2, 3))
# sample sizes
class_sizes <- c(100, 100, 100)

# generate random samples
points1 <- mvrnorm(n = class_sizes[1], mu = class_means[,1], Sigma = class_covariances[,,1])
points2 <- mvrnorm(n = class_sizes[2], mu = class_means[,2], Sigma = class_covariances[,,2])
points3 <- mvrnorm(n = class_sizes[3], mu = class_means[,3], Sigma = class_covariances[,,3])
X <- rbind(points1, points2, points3)
colnames(X) <- c("x1", "x2")

# generate corresponding labels
y <- c(rep(1, class_sizes[1]), rep(2, class_sizes[2]), rep(3, class_sizes[3]))

# write data to a file
write.csv(x = cbind(X, y), file = "hw01_data_set.csv", row.names = FALSE)

# plot data points generated
plot(points1[,1], points1[,2], type = "p", pch = 19, col = "red", las = 1,
     xlim = c(-6, 6), ylim = c(-6, 6),
     xlab = "x1", ylab = "x2")
points(points2[,1], points2[,2], type = "p", pch = 19, col = "green")
points(points3[,1], points3[,2], type = "p", pch = 19, col = "blue")

# read data into memory
data_set <- read.csv("hw01_data_set.csv")

# get x and y values
X <- cbind(data_set$x1, data_set$x2)
y_truth <- data_set$y

# get number of classes and number of samples
K <- max(y_truth)
N <- length(y_truth)

# calculate sample means
sample_means <- sapply(1:K, FUN = function(c) sapply(1:ncol(X), FUN=function(a) mean(X[,a][y_truth == c])))

# calculate sample deviations
sample_covariances <- array(sapply(X = 1:K, FUN = function(c) sapply(1:ncol(X), FUN=function(a) sapply(1:ncol(X), FUN=function(b) mean((X[,a][y_truth == c] - sample_means[a,c])*(X[,b][y_truth == c] - sample_means[b,c]))))), c(2,2,3))

# calculate prior probabilities
class_priors <- sapply(X = 1:K, FUN = function(c) {mean(y_truth == c)})

# calculate determinants for covariances
det_covariances <- c(det(sample_covariances[,,1]), det(sample_covariances[,,2]), det(sample_covariances[,,3]))

# define score functions
scoring_function <- function(x1,x2) sapply(X = 1:K, FUN = function(c) {(-ncol(X)/2 * log(2 * pi)) - (0.5 * log(det_covariances[c])) - (0.5*t(c(x1,x2)-sample_means[,c])%*%chol2inv(chol(sample_covariances[,,c]))%*%(c(x1,x2)-sample_means[,c]))+log(class_priors[c])})

# evaluate score values for X
score_values <- mapply(scoring_function,X[,1],X[,2])

#predict y
y_predicted <- apply(X = score_values, MARGIN = 2, FUN = which.max)

#get confusion matrix
confusion_matrix <- table(y_predicted, y_truth)
print(confusion_matrix)

# evaluate on a grid
x1_interval <- seq(from = -6, to = +6, by = 0.06)
x2_interval <- seq(from = -6, to = +6, by = 0.06)
x1_grid <- matrix(x1_interval, nrow = length(x1_interval), ncol = length(x1_interval), byrow = FALSE)
x2_grid <- matrix(x2_interval, nrow = length(x2_interval), ncol = length(x2_interval), byrow = TRUE)

grid_scores <- mapply(scoring_function,x1_grid,x2_grid)
grid_classes <- matrix(apply(X = grid_scores, MARGIN = 2, FUN = which.max), nrow(x2_grid), ncol(x2_grid))

plot(X[y_truth == 1, 1], X[y_truth == 1, 2], type = "p", pch = 19, col = "red",
     xlim = c(-6, +6),
     ylim = c(-6, +6),
     xlab = "x1", ylab = "x2", las = 1)
points(X[y_truth == 2, 1], X[y_truth == 2, 2], type = "p", pch = 19, col = "green")
points(X[y_truth == 3, 1], X[y_truth == 3, 2], type = "p", pch = 19, col = "blue")
points(X[y_predicted != y_truth, 1], X[y_predicted != y_truth, 2], cex = 1.5, lwd = 2)
points(x1_grid[grid_classes == 1], x2_grid[grid_classes == 1], col = rgb(red = 1, green = 0, blue = 0, alpha = 0.03), pch = 16)
points(x1_grid[grid_classes == 2], x2_grid[grid_classes == 2], col = rgb(red = 0, green = 1, blue = 0, alpha = 0.03), pch = 16)
points(x1_grid[grid_classes == 3], x2_grid[grid_classes == 3], col = rgb(red = 0, green = 0, blue = 1, alpha = 0.03), pch = 16)
contour(x1_interval, x2_interval, grid_classes, levels = c(1,2,3), add = TRUE, lwd = 2, drawlabels = FALSE)

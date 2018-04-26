library(MASS)

set.seed(421)
# mean parameters
class_means <- matrix(c(+2.0, +2.0,
                        -4.0, -4.0,
                        -2.0, +2.0,
                        +4.0, -4.0,
                        -2.0, -2.0,
                        +4.0, +4.0,
                        +2.0, -2.0,
                        -4.0, +4.0), 2, 8)
# covariance parameters
class_covariances <- array(c(+0.8, -0.6, -0.6, +0.8,
                             +0.4, +0.0, +0.0, +0.4,
                             +0.8, +0.6, +0.6, +0.8,
                             +0.4, +0.0, +0.0, +0.4,
                             +0.8, -0.6, -0.6, +0.8,
                             +0.4, +0.0, +0.0, +0.4,
                             +0.8, +0.6, +0.6, +0.8,
                             +0.4, +0.0, +0.0, +0.4), c(2, 2, 8))
# sample sizes
class_sizes <- c(100, 100, 100, 100)

# generate random samples
points1 <- mvrnorm(n = class_sizes[1] / 2, mu = class_means[,1], Sigma = class_covariances[,,1])
points2 <- mvrnorm(n = class_sizes[1] / 2, mu = class_means[,2], Sigma = class_covariances[,,2])
points3 <- mvrnorm(n = class_sizes[2] / 2, mu = class_means[,3], Sigma = class_covariances[,,3])
points4 <- mvrnorm(n = class_sizes[2] / 2, mu = class_means[,4], Sigma = class_covariances[,,4])
points5 <- mvrnorm(n = class_sizes[3] / 2, mu = class_means[,5], Sigma = class_covariances[,,5])
points6 <- mvrnorm(n = class_sizes[3] / 2, mu = class_means[,6], Sigma = class_covariances[,,6])
points7 <- mvrnorm(n = class_sizes[4] / 2, mu = class_means[,7], Sigma = class_covariances[,,7])
points8 <- mvrnorm(n = class_sizes[4] / 2, mu = class_means[,8], Sigma = class_covariances[,,8])
X <- rbind(points1, points2, points3, points4, points5, points6, points7, points8)
colnames(X) <- c("x1", "x2")

# generate corresponding labels
y_truth <- cbind(rep(c(1,0,0,0), each=class_sizes[1]), rep(c(0,1,0,0), each=class_sizes[2]), rep(c(0,0,1,0), each=class_sizes[3]), rep(c(0,0,0,1), each=class_sizes[4]))

# plot data points generated
plot(points1[,1], points1[,2], type = "p", pch = 19, col = "red", las = 1,
     xlim = c(-6, 6), ylim = c(-6, 6),
     xlab = "x1", ylab = "x2")
points(points2[,1], points2[,2], type = "p", pch = 19, col = "red")
points(points3[,1], points3[,2], type = "p", pch = 19, col = "green")
points(points4[,1], points4[,2], type = "p", pch = 19, col = "green")
points(points5[,1], points5[,2], type = "p", pch = 19, col = "blue")
points(points6[,1], points6[,2], type = "p", pch = 19, col = "blue")
points(points7[,1], points7[,2], type = "p", pch = 19, col = "magenta")
points(points8[,1], points8[,2], type = "p", pch = 19, col = "magenta")

# get number of samples and number of features
N <- nrow(y_truth)
D <- ncol(X)
K <- ncol(y_truth)

# define the sigmoid function
sigmoid <- function(x) {
  return (1 / (1 + exp(-x)))
}

# define the softmax function
softmax <- function(x) {
  max_x = apply(x, MARGIN = 1, FUN = max) # for numerical stability
  e_x = exp(x-max_x)
  return (e_x/rowSums(e_x))
}

# set learning parameters
eta <- 0.1
epsilon <- 1e-3
H <- 20
max_iteration <- 200

# randomly initalize W and v
set.seed(421)
W <- matrix(runif((D + 1)*H, min = -0.01, max = 0.01), D + 1, H)
V <- matrix(runif((H + 1)*K, min = -0.01, max = 0.01), H + 1, K)

Z <- sigmoid(cbind(1, X) %*% W)
y_predicted <- softmax(cbind(1, Z) %*% V)
objective_values <- c(-sum(y_truth * log(y_predicted+1e-100)))

# learn W and v using gradient descent and online learning
iteration <- 1
while (1) {
  for(t in sample(N)) {
    Z[t,] <- sigmoid(c(1, X[t,]) %*% W)
    y_predicted[t,] <- softmax(c(1, Z[t,]) %*% V)
    
    V <- V + eta * (c(1,Z[t,])%*%t(y_truth[t,] - y_predicted[t,]))
    W <- W + eta * (c(1,X[t,])%*%t(c((y_truth[t,] - y_predicted[t,])%*%t(V[-1,])) * Z[t,] * (1 - Z[t,])))
  }

  Z <- sigmoid(cbind(1, X) %*% W)
  y_predicted <- softmax(cbind(1, Z) %*% V)
  objective_values <- c(objective_values, -sum(y_truth * log(y_predicted + 1e-100)))
  
  if (abs(objective_values[iteration + 1] - objective_values[iteration]) < epsilon | iteration >= max_iteration) {
    break
  }
  
  iteration <- iteration + 1
}

# plot objective function during iterations
plot(1:(iteration + 1), objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

# calculate confusion matrix
y_predicted <- apply(y_predicted, MARGIN=1, FUN=which.max)
y_truth <- apply(y_truth, MARGIN=1, FUN=which.max)
confusion_matrix <- table(y_predicted, y_truth)
print(confusion_matrix)

# evaluate discriminant function on a grid
x1_interval <- seq(from = -6, to = +6, by = 0.06)
x2_interval <- seq(from = -6, to = +6, by = 0.06)
x1_grid <- matrix(x1_interval, nrow = length(x1_interval), ncol = length(x1_interval), byrow = FALSE)
x2_grid <- matrix(x2_interval, nrow = length(x2_interval), ncol = length(x2_interval), byrow = TRUE)

f <- function(x1, x2) { c(1, sigmoid(c(1, x1, x2) %*% W)) %*% V }

discriminant_scores <- mapply(f,x1_grid,x2_grid)
discriminant_values <- matrix(apply(discriminant_scores, MARGIN = 2, which.max), nrow(x2_grid), ncol(x2_grid))

plot(X[y_truth == 1, 1], X[y_truth == 1, 2], type = "p", pch = 19, col = "red",
     xlim = c(-6, +6),
     ylim = c(-6, +6),
     xlab = "x1", ylab = "x2", las = 1)
points(X[y_truth == 2, 1], X[y_truth == 2, 2], type = "p", pch = 19, col = "green")
points(X[y_truth == 3, 1], X[y_truth == 3, 2], type = "p", pch = 19, col = "blue")
points(X[y_truth == 4, 1], X[y_truth == 4, 2], type = "p", pch = 19, col = "magenta")
points(X[y_predicted != y_truth, 1], X[y_predicted != y_truth, 2], cex = 1.5, lwd = 2)
points(x1_grid[discriminant_values == 1], x2_grid[discriminant_values == 1], col = rgb(red = 1, green = 0, blue = 0, alpha = 0.03), pch = 16)
points(x1_grid[discriminant_values == 2], x2_grid[discriminant_values == 2], col = rgb(red = 0, green = 1, blue = 0, alpha = 0.03), pch = 16)
points(x1_grid[discriminant_values == 3], x2_grid[discriminant_values == 3], col = rgb(red = 0, green = 0, blue = 1, alpha = 0.03), pch = 16)
points(x1_grid[discriminant_values == 4], x2_grid[discriminant_values == 4], col = rgb(red = 1, green = 0, blue = 1, alpha = 0.03), pch = 16)
contour(x1_interval, x2_interval, discriminant_values, levels = c(1,2,3,4), add = TRUE, lwd = 2, drawlabels = FALSE)
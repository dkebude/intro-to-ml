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

# generate corresponding labels
y_truth <- cbind(rep(c(1,0,0), each=class_sizes[1]), rep(c(0,1,0), each=class_sizes[2]), rep(c(0,0,1), each=class_sizes[3]))

# plot data points generated
plot(points1[,1], points1[,2], type = "p", pch = 19, col = "red", las = 1,
     xlim = c(-6, 6), ylim = c(-6, 6),
     xlab = "x1", ylab = "x2")
points(points2[,1], points2[,2], type = "p", pch = 19, col = "green")
points(points3[,1], points3[,2], type = "p", pch = 19, col = "blue")

# get number of classes and number of samples
K <- ncol(y_truth)
N <- nrow(y_truth)

# set learning parameters
eta <- 0.01
epsilon <- 1e-3

# randomly initalize w and w0
set.seed(420)
W <- matrix(runif(6, min = -0.01, max = 0.01), 2, 3)
w0 <- runif(3, min = -0.01, max = 0.01)

# define the softmax function
softmax <- function(X, W, w0) {
  x = X %*% W
  for(k in 1:K) {
   x[,k] = x[,k] + w0[k]
  }
  max_x = apply(x, MARGIN = 1, FUN = max) # for numerical stability
  e_x = exp(x-max_x)
  return (e_x/rowSums(e_x))
}

# define error function
objective <- function(y_truth, y_predicted) {
  return (-sum(y_truth*log(y_predicted)))
}

# define the gradient functions
grad_W <- function(X, y_truth, y_predicted) {
  gJgy = (y_truth-y_predicted)
  gJgW = (t(X)%*%gJgy)
  return (gJgW)
}

grad_w0 <- function(y_truth, y_predicted ) {
  gJgy = (y_truth-y_predicted)
  gJgw0 = colSums(gJgy)
  return (gJgw0)
}

# learn W and w0 using gradient descent
iteration <- 1
objective_values <- c()
while (1) {
  y_predicted <- softmax(X, W, w0)

  objective_values <- c(objective_values, objective(y_truth, y_predicted))

  W_old <- W
  w0_old <- w0

  W <- W + eta*grad_W(X, y_truth, y_predicted)
  w0 <- w0 + eta*grad_w0(y_truth, y_predicted)

  if (sqrt(sum((w0 - w0_old)^2) + sum((W - W_old)^2)) < epsilon) {
    break
  }

  iteration <- iteration + 1
}
print(W)
print(w0)

# plot objective function during iterations
plot(1:iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

# get classes for y_predicted and y_truth
y_predicted <- apply(softmax(X,W,w0), MARGIN=1, FUN=which.max)
y_truth <- apply(y_truth, MARGIN=1, FUN=which.max)

# get confusion matrix
confusion_matrix <- table(y_predicted, y_truth)
print(confusion_matrix)

# evaluate on a grid
x1_interval <- seq(from = -6, to = +6, by = 0.06)
x2_interval <- seq(from = -6, to = +6, by = 0.06)
x1_grid <- matrix(x1_interval, nrow = length(x1_interval), ncol = length(x1_interval), byrow = FALSE)
x2_grid <- matrix(x2_interval, nrow = length(x2_interval), ncol = length(x2_interval), byrow = TRUE)

discriminant <- function(x1, x2){
  x1*W[1,]+x2*W[2,]+w0
}

discriminant_scores <- mapply(discriminant,x1_grid,x2_grid)
discriminant_values <- matrix(apply(discriminant_scores, MARGIN = 2, which.max), nrow(x2_grid), ncol(x2_grid))

plot(X[y_truth == 1, 1], X[y_truth == 1, 2], type = "p", pch = 19, col = "red",
     xlim = c(-6, +6),
     ylim = c(-6, +6),
     xlab = "x1", ylab = "x2", las = 1)
points(X[y_truth == 2, 1], X[y_truth == 2, 2], type = "p", pch = 19, col = "green")
points(X[y_truth == 3, 1], X[y_truth == 3, 2], type = "p", pch = 19, col = "blue")
points(X[y_predicted != y_truth, 1], X[y_predicted != y_truth, 2], cex = 1.5, lwd = 2)
points(x1_grid[discriminant_values == 1], x2_grid[discriminant_values == 1], col = rgb(red = 1, green = 0, blue = 0, alpha = 0.03), pch = 16)
points(x1_grid[discriminant_values == 2], x2_grid[discriminant_values == 2], col = rgb(red = 0, green = 1, blue = 0, alpha = 0.03), pch = 16)
points(x1_grid[discriminant_values == 3], x2_grid[discriminant_values == 3], col = rgb(red = 0, green = 0, blue = 1, alpha = 0.03), pch = 16)
contour(x1_interval, x2_interval, discriminant_values, levels = c(1,2,3), add = TRUE, lwd = 2, drawlabels = FALSE)

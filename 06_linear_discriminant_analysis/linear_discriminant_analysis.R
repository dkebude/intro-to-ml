# read data into memory
training_digits <- read.csv("mnist_training_digits.csv", header = FALSE)
training_labels <- read.csv("mnist_training_labels.csv", header = FALSE)
test_digits <- read.csv("mnist_test_digits.csv", header = FALSE)
test_labels <- read.csv("mnist_test_labels.csv", header = FALSE)

# get X and y values
X_train <- as.matrix(training_digits) / 255
y_train <- training_labels[,1]
X_test <- as.matrix(test_digits) / 255
y_test <- test_labels[,1]

# get number of samples and number of features
N_train <- length(y_train)
D_train <- ncol(X_train)
N_test <- length(y_test)
D_test <- ncol(X_test)
K <- max(y_train)

# calculate the sample means
sample_means <- sapply(1:K, FUN = function(c) colMeans(X_train[y_train == c,]))

within_class <- sapply(1:K, FUN=function(c) t(X_train[y_train==c,]-matrix(sample_means[,c], sum(y_train==c), D_train, byrow = TRUE))%*%(X_train[y_train==c,]-matrix(sample_means[,c], sum(y_train==c), D_train, byrow = TRUE)))
within_class <- array(within_class, c(D_train, D_train, K))

# Calculate the total within-class scatter matrix
S_W <- rowSums(within_class, dims=2) + 1e-10*diag(D_train)
invS_W <- chol2inv(chol(S_W))

# Calculate overall mean
overall_mean <- rowSums(sample_means)/K

# Calculate between-class scatter matrix
between_class <- sapply(1:K, FUN=function(c) sum(y_train==c)*((sample_means[,c]-overall_mean)%*%t(sample_means[,c]-overall_mean)))
between_class <- array(between_class, c(D_train, D_train, K))
S_B <- rowSums(between_class, dims=2)

# calculate the eigenvalues and eigenvectors
decomposition <- eigen(invS_W%*%S_B, symmetric = TRUE)
W <- decomposition$vectors

R <- 2
# calculate two-dimensional projections
Z_train <- (X_train - matrix(colMeans(X_train), N_train, D_train, byrow = TRUE)) %*% decomposition$vectors[,1:R]
Z_test <- (X_test - matrix(colMeans(X_test), N_test, D_test, byrow = TRUE)) %*% decomposition$vectors[,1:R]

# plot two-dimensional projections
point_colors <- c("#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6")
plot(Z_train[,1], Z_train[,2], type = "p", pch = 19, col = point_colors[y_train], cex = 0,
     xlab = "Dimension 1", ylab = "Dimension 2", las = 1)
text(Z_train[,1], Z_train[,2], labels = y_train %% 10, col = point_colors[y_train])

plot(Z_test[,1], Z_test[,2], type = "p", pch = 19, col = point_colors[y_test], cex = 0,
     xlab = "Dimension 1", ylab = "Dimension 2", las = 1)
text(Z_test[,1], Z_test[,2], labels = y_test %% 10, col = point_colors[y_test])

# knn classification
k <- 5
accuracies <- rep(0,9)
R <- c(1:9)
for(r in R) {
  y_hat <- matrix(0, N_test, K)
  Z_train <- (X_train - matrix(colMeans(X_train), N_train, D_train, byrow = TRUE)) %*% decomposition$vectors[,1:r]
  Z_test <- (X_test - matrix(colMeans(X_train), N_test, D_test, byrow = TRUE)) %*% decomposition$vectors[,1:r]
  for(c in 1:K) {
    y_hat[,c] <- sapply(1:N_test, function(i) {sum(y_train[order(sapply(1:N_train, function(j) {sqrt(sum((Z_test[i,] - Z_train[j,])^2))}), decreasing = FALSE)[1:k]] == c) / k})
  }
  y_hat <- sapply(1:N_test, function(i) which.max(y_hat[i,]))
  accuracies[r] <- (sum(y_hat == y_test)/N_test)*100
  print(accuracies[r])
}

plot(R, accuracies, lwd = 2)
lines(R, accuracies, type = "c", lwd = 2)

library(MASS)
if(!"mixtools" %in% installed.packages()[,"Package"]) install.packages("mixtools")
library(mixtools)

set.seed(521)
# mean parameters
class_means <- matrix(c(+2.5, +2.5,
                        -2.5, +2.5,
                        -2.5, -2.5,
                        +2.5, -2.5,
                        +0.0, +0.0), 2, 5)
# covariance parameters
class_covariances <- array(c(+0.8, -0.6, -0.6, +0.8,
                             +0.8, +0.6, +0.6, +0.8,
                             +0.8, -0.6, -0.6, +0.8,
                             +0.8, +0.6, +0.6, +0.8,
                             +1.6, +0.0, +0.0, +1.6), c(2, 2, 5))
# sample sizes
class_sizes <- c(50, 50, 50, 50, 100)

# generate random samples
points1 <- mvrnorm(n = class_sizes[1], mu = class_means[,1], Sigma = class_covariances[,,1])
points2 <- mvrnorm(n = class_sizes[2], mu = class_means[,2], Sigma = class_covariances[,,2])
points3 <- mvrnorm(n = class_sizes[3], mu = class_means[,3], Sigma = class_covariances[,,3])
points4 <- mvrnorm(n = class_sizes[4], mu = class_means[,4], Sigma = class_covariances[,,4])
points5 <- mvrnorm(n = class_sizes[5], mu = class_means[,5], Sigma = class_covariances[,,5])
X <- rbind(points1, points2, points3, points4, points5)
colnames(X) <- c("x1", "x2")

# plot data points generated
plot(points1[,1], points1[,2], type = "p", pch = 19, col = "black", las = 1,
     xlim = c(-6, 6), ylim = c(-6, 6),
     xlab = "x1", ylab = "x2")
points(points2[,1], points2[,2], type = "p", pch = 19, col = "black")
points(points3[,1], points3[,2], type = "p", pch = 19, col = "black")
points(points4[,1], points4[,2], type = "p", pch = 19, col = "black")
points(points5[,1], points5[,2], type = "p", pch = 19, col = "black")

# get number of samples and number of features
N <- nrow(X)
D <- ncol(X)
K <- 5

# k-means clustering (two iterations)
centroids <- X[sample(1:N, K),]
for(i in 1:2) {
  distances <- as.matrix(dist(rbind(centroids, X), method = "euclidean"))
  distances <- distances[1:nrow(centroids), (nrow(centroids) + 1):(nrow(centroids) + nrow(X))]
  assignments <- sapply(1:ncol(distances), function(i) {which.min(distances[,i])})
  for (k in 1:K) {
    centroids[k,] <- colMeans(X[assignments == k,])
  }
}

# Initializations for expectation-maximization clustering
means <- centroids
covariances <- array(0, c(2, 2, 5))
priors <- rep(0, 5)
for(k in 1:K) {
  covariances[,,k] = (t(X[assignments == k,] - means[k,])%*%(X[assignments == k,] - means[k,]))/sum(assignments==k)
  priors[k] = sum(assignments==k)/length(assignments)
}

## Expectation-Maximization Algorithm
iterations <- 100
posteriors <- matrix(0, N, K)
h <- matrix(0, N, K)
for(iter in 1:iterations) {
  # Expectation Step
  for(c in 1:K) {
      covinv <- chol2inv(chol(covariances[,,c]));
      posteriors[,c] <- sapply(1:N, function(t) priors[c]*det(covariances[,,c])^(-1/2)*exp(-(1/2)*t(X[t,] - means[c,])%*%covinv%*%(X[t,] - means[c,])))
  }
  h = posteriors/rowSums(posteriors)
  z = sapply(1:N, function(t) which.max(h[t,]))
  # Maximization Step
  for(c in 1:K) {
    priors[c] <- sum(h[,c])/N
    means[c,] <- h[,c]%*%X/sum(h[,c])
    numerator <- sapply(1:N, function(t) h[t,c]*(X[t,]-means[c,])%*%t(X[t,]-means[c,]))
    covariances[,,c] <- rowSums(array(numerator, c(D,D,N)),dims=2)/sum(h[,c])
  }
}
print(means)

colors <- c("#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a")
plot(X[,1], X[,2], type = "p", pch = 19, col = colors[z], las = 1,
     xlim = c(-6, 6), ylim = c(-6, 6), 
     xlab = "x1", ylab = "x2")
ellipse(mu=class_means[,1],sigma=class_covariances[,,1], npoints=class_sizes[1], col="black", alpha=.15, lwd = 2, lty="dashed")
ellipse(mu=class_means[,2],sigma=class_covariances[,,2], npoints=class_sizes[2], col="black", alpha=.15, lwd = 2, lty="dashed")
ellipse(mu=class_means[,3],sigma=class_covariances[,,3], npoints=class_sizes[3], col="black", alpha=.15, lwd = 2, lty="dashed")
ellipse(mu=class_means[,4],sigma=class_covariances[,,4], npoints=class_sizes[4], col="black", alpha=.15, lwd = 2, lty="dashed")
ellipse(mu=class_means[,5],sigma=class_covariances[,,5], npoints=class_sizes[5], col="black", alpha=.15, lwd = 2, lty="dashed")
ellipse(mu=means[1,],sigma=covariances[,,1], npoints=class_sizes[1], col="black", alpha=.15, lwd = 2, lty="solid")
ellipse(mu=means[2,],sigma=covariances[,,2], npoints=class_sizes[2], col="black", alpha=.15, lwd = 2, lty="solid")
ellipse(mu=means[3,],sigma=covariances[,,3], npoints=class_sizes[3], col="black", alpha=.15, lwd = 2, lty="solid")
ellipse(mu=means[4,],sigma=covariances[,,4], npoints=class_sizes[4], col="black", alpha=.15, lwd = 2, lty="solid")
ellipse(mu=means[5,],sigma=covariances[,,5], npoints=class_sizes[5], col="black", alpha=.15, lwd = 2, lty="solid")
set.seed(521)
# read data into memory
data_set <- read.csv("npr_data_set.csv")

# get x and y values
x <- data_set$x
y <- data_set$y
indices <- c(1:133)
indices <- sample(indices)

x_train <- x[indices[1:100]]
y_train <- y[indices[1:100]]
x_test <- x[indices[101:133]]
y_test <- y[indices[101:133]]

# get number of samples in test data
N_test <- length(y_test)

point_colors <- c("blue", "red")
minimum_value <- min(x)-2.4
maximum_value <- max(x)+2.4
data_interval <- seq(from = minimum_value, to = maximum_value, by = 0.1)

# rmse
rmse <- function(y_truth, y_hat, N) {
  return (sqrt(sum((y_truth-y_hat)^2)/N))
}
  
# regressogram
p_regressogram <- function(left_borders, right_borders, x, y) {
  numerator <- rep(c(0), length(left_borders))
  denominator <- rep(c(0), length(left_borders))
  for(b in 1:length(left_borders)) {
    numerator[b] = sum((left_borders[b] < x & x <= right_borders[b])*y)
    denominator[b] = sum(left_borders[b] < x & x <= right_borders[b])+1e-100
  }
  return (numerator/denominator)
}

# predict through fitted regressogram
regressogram_predict <- function(x, p_hat, left_borders, right_borders) {
  y_hat <- rep(c(0), length(x))
  for(i in 1:length(x)) {
    for(b in 1:length(left_borders)) {
      if(left_borders[b] < x[i] & x[i] <= right_borders[b]) {
        y_hat[i] = p_hat[b]
      }
    }
  }
  return (y_hat)
}

bin_width <- 3
left_borders <- seq(from = minimum_value, to = maximum_value - bin_width, by = bin_width)
right_borders <- seq(from = minimum_value + bin_width, to = maximum_value, by = bin_width)
p_hat <- p_regressogram(left_borders, right_borders, x_train, y_train)#sapply(1:length(left_borders), function(b) p_regressogram(b, left_borders, right_borders, x_train, y_train))

plot(x_train, y_train, type = "p", pch = 19, col = point_colors[1],
     ylim = c(min(y_train)-20, max(y_train)+20), xlim = c(minimum_value, maximum_value),
     ylab = "y", xlab = "x", las = 1, main = sprintf("h = %g", bin_width))
points(x_test, y_test, type = "p", pch = 19, col = point_colors[2])
for (b in 1:length(left_borders)) {
  lines(c(left_borders[b], right_borders[b]), c(p_hat[b], p_hat[b]), lwd = 2, col = "black")
  if (b < length(left_borders)) {
    lines(c(right_borders[b], right_borders[b]), c(p_hat[b], p_hat[b + 1]), lwd = 2, col = "black") 
  }
}
y_hat <- regressogram_predict(x_test, p_hat, left_borders, right_borders)
rmse_regressogram <- rmse(y_test, y_hat, N_test)
regressogram_str = sprintf("Regressogram => RMSE is %.4f when h is %d", rmse_regressogram, bin_width)
print(regressogram_str)

# running mean smoother
p_rms <- function(data_interval, bin_width, x, y) {
  numerator <- rep(c(0), length(data_interval))
  denominator <- rep(c(0), length(data_interval))
  for(i in 1:length(data_interval)) {
    numerator[i] = sum(((data_interval[i] - 0.5 * bin_width) < x & x <= (data_interval[i] + 0.5 * bin_width))*y)
    denominator[i] = sum((data_interval[i] - 0.5 * bin_width) < x & x <= (data_interval[i] + 0.5 * bin_width))+1e-100
  }
  return (numerator/denominator)
}

rms_predict <- function(x, p_hat, data_interval) {
  y_hat <- rep(c(0), length(x))
  for(i in 1:length(x)) {
    for(d in 1:length(data_interval))
    {
      if(all.equal(data_interval[d], x[i]) == TRUE) {
        y_hat[i] = p_hat[d]
      }
    }
  }
  return (y_hat)
}

bin_width <- 3
p_hat <- p_rms(data_interval, bin_width, x_train, y_train)

plot(x_train, y_train, type = "p", pch = 19, col = point_colors[1],
     ylim = c(min(y_train)-20, max(y_train)+20), xlim = c(minimum_value, maximum_value),
     ylab = "y", xlab = "x", las = 1, main = sprintf("h = %g", bin_width))
points(x_test, y_test, type = "p", pch = 19, col = point_colors[2])
lines(data_interval, p_hat, type = "l", lwd = 2, col = "black")

y_hat <- rms_predict(x_test, p_hat, data_interval)
rmse_regressogram <- rmse(y_test, y_hat, N_test)
rms_str = sprintf("Running Mean Smoother => RMSE is %.4f when h is %d", rmse_regressogram, bin_width)
print(rms_str)

# kernel smoother
p_kernel <- function(data_interval, bin_width, x, y) {
  numerator <- rep(c(0), length(data_interval))
  denominator <- rep(c(0), length(data_interval))
  for(i in 1:length(data_interval)) {
    numerator[i] = sum((1 / sqrt(2 * pi) * exp(-0.5 * (data_interval[i] - x)^2 / bin_width^2))*y)
    denominator[i] = sum((1 / sqrt(2 * pi) * exp(-0.5 * (data_interval[i] - x)^2 / bin_width^2)))+1e-100
  }
  return (numerator/denominator)
}

kernel_predict <- function(x, p_hat, data_interval) {
  y_hat <- rep(c(0), length(x))
  for(i in 1:length(x)) {
    for(d in 1:length(data_interval))
    {
      if(all.equal(data_interval[d], x[i]) == TRUE) {
        y_hat[i] = p_hat[d]
      }
    }
  }
  return (y_hat)
}

bin_width <- 1
p_hat <- p_kernel(data_interval, bin_width, x_train, y_train)

plot(x_train, y_train, type = "p", pch = 19, col = point_colors[1],
     ylim = c(min(y_train)-20, max(y_train)+20), xlim = c(minimum_value, maximum_value),
     ylab = "y", xlab = "x", las = 1, main = sprintf("h = %g", bin_width))
points(x_test, y_test, type = "p", pch = 19, col = point_colors[2])
lines(data_interval, p_hat, type = "l", lwd = 2, col = "black")

y_hat <- kernel_predict(x_test, p_hat, data_interval)
rmse_regressogram <- rmse(y_test, y_hat, N_test)
kernel_str = sprintf("Kernel Smoother => RMSE is %.4f when h is %d", rmse_regressogram, bin_width)
print(kernel_str)
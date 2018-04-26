set.seed(521)
# read data into memory
data_set <- read.csv("dtr_data_set.csv")

# get x and y values
x <- data_set$x
y <- data_set$y
indices <- c(1:133)
indices <- sample(indices)

x_train <- x[indices[1:100]]
y_train <- y[indices[1:100]]
x_test <- x[indices[101:133]]
y_test <- y[indices[101:133]]

# get numbers of train and test samples
N_train <- length(y_train)
N_test <- length(y_test)

# create necessary data structures
node_splits <- c()
node_values <- list()

# put all training instances into the root node
node_indices <- list(1:N_train)
is_terminal <- c(FALSE)
need_split <- c(TRUE)

# pre-pruning threshold
P <- 10

# learning algorithm
while (1) {
  # find nodes that need splitting
  split_nodes <- which(need_split)
  # check whether we reach all terminal nodes
  if (length(split_nodes) == 0) {
    break
  }
  # find best split positions for all nodes
  for (split_node in split_nodes) {
    data_indices <- node_indices[[split_node]]
    need_split[split_node] <- FALSE
    node_values[[split_node]] <- sum(y_train[data_indices], na.rm=TRUE)/length(data_indices)
    # check whether node is pure
    if (length(y_train[data_indices]) <= P || ((P == 1 || P == 2) && length(unique(x_train[data_indices])) <= P)) {
      is_terminal[split_node] <- TRUE
    } else {
      is_terminal[split_node] <- FALSE
      
      unique_values <- sort(unique(x_train[data_indices]))
      split_positions <- (unique_values[-1] + unique_values[-length(unique_values)]) / 2
      split_scores <- rep(0, length(split_positions))
      for (s in 1:length(split_positions)) {
        left_indices <- data_indices[which(x_train[data_indices] <= split_positions[s])]
        g_left <- sum(y_train[left_indices])/length(left_indices)
        left_sqr_err <- (y_train-g_left)^2
        left_sum <- sum(left_sqr_err[left_indices])
        right_indices <- data_indices[which(x_train[data_indices] > split_positions[s])]
        g_right <- sum(y_train[right_indices])/length(right_indices)
        right_sqr_err <- (y_train-g_right)^2
        right_sum <- sum(right_sqr_err[right_indices], na.rm=TRUE)
        split_scores[s] <- (1 / length(data_indices)) * (left_sum + right_sum)
      }
      
      # decide where to split on which feature
      split <- which.min(split_scores)
      node_splits[split_node] <- split_positions[split]
      
      # create left node using the selected split
      left_indices <- data_indices[which(x_train[data_indices] <= split_positions[split])]
      node_indices[[2 * split_node]] <- left_indices
      is_terminal[2 * split_node] <- FALSE
      need_split[2 * split_node] <- TRUE
      
      # create left node using the selected split
      right_indices <- data_indices[which(x_train[data_indices] > split_positions[split])]
      node_indices[[2 * split_node + 1]] <- right_indices
      is_terminal[2 * split_node + 1] <- FALSE
      need_split[2 * split_node + 1] <- TRUE
    }
  }
}

terminal_nodes <- which(is_terminal)
x_min <- rep(0, length(terminal_nodes))
x_max <- rep(0, length(terminal_nodes))
terminal_values <- rep(0, length(terminal_nodes))
for (i in 1:length(terminal_nodes)) {
  index <- terminal_nodes[i]
  x_min[i] <- 0.0
  x_max[i] <- 60.0
  while (index > 1) {
    parent <- floor(index / 2)
    if (index %% 2 == 0) {
      # if node is left child of its parent
      x_max[i] <- min(x_max[i], node_splits[parent])
    } else {
      # if node is right child of its parent
      x_min[i] <- max(x_min[i], node_splits[parent])
    }
    index <- parent
  }
  terminal_values[i] <- node_values[[terminal_nodes[i]]]
}
indices <- order(x_min)
x_min <- x_min[indices]
x_max <- x_max[indices]
terminal_values <- terminal_values[indices]

point_colors <- c("blue", "red")
minimum_value <- min(x)-2.4
maximum_value <- max(x)+2.4
plot(x_train, y_train, type = "p", pch = 19, col = point_colors[1],
     ylim = c(min(y_train)-20, max(y_train)+20), xlim = c(minimum_value, maximum_value),
     ylab = "y", xlab = "x", las = 1, main = sprintf("P = %g", P))
points(x_test, y_test, type = "p", pch = 19, col = point_colors[2])
legend("topright", legend = c("training","test"), pch = 19, col = point_colors)
for(i in 1:length(x_min)) {
  lines(c(x_min[i], x_max[i]), c(terminal_values[i], terminal_values[i]), lwd = 2, col = "black")
  lines(c(x_max[i], x_min[i+1]), c(terminal_values[i], terminal_values[i+1]), lwd = 2, col = "black")
}

# traverse tree for test data points
y_predicted <- rep(0, N_test)
for (i in 1:N_test) {
  index <- 1
  while (1) {
    if (is_terminal[index] == TRUE) {
      y_predicted[i] <- node_values[[index]]
      break
    } else {
      if (x_test[i] <= node_splits[index]) {
        index <- index * 2
      } else {
        index <- index * 2 + 1
      }
    }
  }
}

rmse <- sqrt(sum((y_test-y_predicted)^2)/N_test)
rmse_str = sprintf("RMSE is %.4f when P is %d", rmse, P)
print(rmse_str)

RMSE <- rep(0, 20)
P <- c(1:20)
for(p in P) {
  # create necessary data structures
  node_splits <- c()
  node_values <- list()
  
  # put all training instances into the root node
  node_indices <- list(1:N_train)
  is_terminal <- c(FALSE)
  need_split <- c(TRUE)
  
  while (1) {
    # find nodes that need splitting
    split_nodes <- which(need_split)
    # check whether we reach all terminal nodes
    if (length(split_nodes) == 0) {
      break
    }
    # find best split positions for all nodes
    for (split_node in split_nodes) {
      data_indices <- node_indices[[split_node]]
      need_split[split_node] <- FALSE
      node_values[[split_node]] <- sum(y_train[data_indices], na.rm=TRUE)/length(data_indices)
      # check whether node is pure
      if (length(y_train[data_indices]) <= p || ((p == 1 || p ==2) && length(unique(x_train[data_indices])) <= p)) {
        is_terminal[split_node] <- TRUE
      } else {
        is_terminal[split_node] <- FALSE
        
        unique_values <- sort(unique(x_train[data_indices]))
        split_positions <- (unique_values[-1] + unique_values[-length(unique_values)]) / 2
        split_scores <- rep(0, length(split_positions))
        for (s in 1:length(split_positions)) {
          left_indices <- data_indices[which(x_train[data_indices] <= split_positions[s])]
          g_left <- sum(y_train[left_indices])/length(left_indices)
          left_sqr_err <- (y_train-g_left)^2
          left_sum <- sum(left_sqr_err[left_indices])
          right_indices <- data_indices[which(x_train[data_indices] > split_positions[s])]
          g_right <- sum(y_train[right_indices])/length(right_indices)
          right_sqr_err <- (y_train-g_right)^2
          right_sum <- sum(right_sqr_err[right_indices], na.rm=TRUE)
          split_scores[s] <- (1 / length(data_indices)) * (left_sum + right_sum)
        }
        
        # decide where to split on which feature
        split <- which.min(split_scores)
        node_splits[split_node] <- split_positions[split]
        
        # create left node using the selected split
        left_indices <- data_indices[which(x_train[data_indices] <= split_positions[split])]
        node_indices[[2 * split_node]] <- left_indices
        is_terminal[2 * split_node] <- FALSE
        need_split[2 * split_node] <- TRUE
        
        # create left node using the selected split
        right_indices <- data_indices[which(x_train[data_indices] > split_positions[split])]
        node_indices[[2 * split_node + 1]] <- right_indices
        is_terminal[2 * split_node + 1] <- FALSE
        need_split[2 * split_node + 1] <- TRUE
      }
    }
  }
  
  # traverse tree for test data points
  y_predicted <- rep(0, N_test)
  for (i in 1:N_test) {
    index <- 1
    while (1) {
      if (is_terminal[index] == TRUE) {
        y_predicted[i] <- node_values[[index]]
        break
      } else {
        if (x_test[i] <= node_splits[index]) {
          index <- index * 2
        } else {
          index <- index * 2 + 1
        }
      }
    }
  }
  
  RMSE[p] <- sqrt(sum((y_test-y_predicted)^2)/N_test)
}

plot(P, RMSE, lwd = 2)
lines(P, RMSE, type = "c", lwd = 2)
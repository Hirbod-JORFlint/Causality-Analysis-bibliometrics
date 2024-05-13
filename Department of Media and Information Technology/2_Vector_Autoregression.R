library(vars)
library(car)

# Load the dataset
df <- read.csv("time_series.csv")

# Drop the 'Year' column
df$Year <- NULL

# Drop constant columns
df <- df[, sapply(df, function(x) length(unique(x)) > 1)]

# Perform stationary testing and store the stationary variables
for (column in colnames(df)) {
  result <- ur.df(df[[column]], type = "drift")
  test_statistic <- summary(result)@teststat[1, "tau2"]
  critical_value_5pct <- summary(result)@cval[1, "5pct"]
  if (test_statistic < critical_value_5pct) {  # if test statistic is less than the 5% critical value, the series is stationary
    stationary_columns <- c(stationary_columns, column)
  }
}

# Perform Vector autoregression on the stationary columns two by two
results <- list()
for (i in seq(1, length(stationary_columns), by = 2)) {
  tryCatch({
    model <- VAR(df[, stationary_columns[i:(i+1)]], lag.max = 4, type = "const")
    results[[length(results) + 1]] <- summary(model)
  }, error = function(e) {
    print(paste("Skipping VAR for columns", stationary_columns[i:(i+1)], "due to LinAlgError"))
  })
}

# Save the results in a new csv file
write.table(do.call(rbind, results), file = "var_results.csv", sep = ";", row.names = FALSE)

# Remove rows with missing values
df_complete <- df[complete.cases(df), ]

# Check if the number of observations is greater than the number of variables
if (nrow(df_complete) > ncol(df_complete)) {
  vif_data <- data.frame(feature = colnames(df_complete))
  vif_data$VIF <- vif(lm(paste0("~ ."), data = df_complete))
  print(vif_data)
} else {
  print("Cannot calculate VIF: Too many variables compared to the number of observations.")
}

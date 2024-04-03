# Split the keywords and convert them to lowercase
df$Keywords <- strsplit(tolower(df$Keywords), split = separator)
df$Keywords <- lapply(df$Keywords, function(x) trimws(x))

# Compute the co-occurrences of keywords
co_occurrences <- df %>%
  unnest(Keywords) %>%
  group_by(Keywords) %>%
  summarise(co_occurrence = n()) %>%
  arrange(desc(co_occurrence))

# Create a Bayesian network
bn <- empty.graph(nodes = unique(unlist(df$Keywords)))

# Add edges between keywords that co-occur
for (i in 1:(nrow(co_occurrences) - 1)) {
  for (j in (i + 1):nrow(co_occurrences)) {
    if (co_occurrences$co_occurrence[i] == co_occurrences$co_occurrence[j]) {
      arc.set <- c(arc.set, paste(co_occurrences$Keywords[i], co_occurrences$Keywords[j]))
      bn <- set.arc(bn, arc.set)
    }
  }
}

# Prepare the data for the model
unique_keywords <- unique(unlist(df$Keywords))
data <- data.frame(matrix(ncol = length(unique_keywords), nrow = nrow(df)))
colnames(data) <- unique_keywords
for (i in 1:nrow(df)) {
  for (keyword in df$Keywords[[i]]) {
    data[i, keyword] <- 1
  }
}

# Fit the data to the model using Maximum Likelihood Estimator
bn.fitted <- bn.fit(bn, data = data)

# Print the model and co-occurrences
print(bn.fitted)
print(co_occurrences)

# Plot the learned Bayesian network
graph <- as.bn.fit(bn.fitted)
graphviz.plot(graph)

# Load variables from the RData file
load("all_variables.RData")

# Install and load the necessary packages
# List of required packages
required_packages <- c("bnlearn", "tidyverse", "BiocManager", "Rgraphviz", "igraph", "plotly", "magrittr")

# Check if each package is installed, and install if not
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}


if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("Rgraphviz")


library(bnlearn)
library(dplyr)
library(stringr)
#library(tidyr)
#library(Rgraphviz)
#library(igraph)
#library(plotly)
#library(magrittr)


separator <- ';'
df <- read.csv("E:/Dars/Liu Thesis/Reports/Causality Analysis/Department of mathematics/math.csv", encoding="UTF-8")
keyword_column <- 'Keywords'
A_num <- 2


# Compute co-occurrences
df <- df %>% 
  select(PID, Year, Keywords) %>%
  filter(!is.na(.data[[keyword_column]]), .data[[keyword_column]] != "")

# Convert lists of keywords to tuples
df[[keyword_column]] <- lapply(df[[keyword_column]], function(x) {
  keywords <- unlist(strsplit(tolower(x), separator))
  trimws(keywords)  # Remove leading and trailing whitespaces
})
# Compute co-occurrences
co_occurrences <- table(unlist(lapply(df[[keyword_column]], function(x) {
  if(length(x) >= 2) {
    combn(sort(x), 2, paste, collapse = "::")
  } else {
    character(0)  # Return empty character vector if less than 2 keywords
  }
})))
co_occurrences <- sort(co_occurrences, decreasing = TRUE)

# Extract edges from co-occurrences
edges <- as.data.frame(matrix(names(co_occurrences), ncol = 1, byrow = TRUE))
# Create a Bayesian Network from the co-occurrences
bn <- empty.graph(nodes = unique(unlist(df[[keyword_column]])))

# Add arcs to the network based on co-occurrences
# for (i in 1:nrow(edges)) {
#   edge <- unlist(strsplit(as.character(edges[i, ]), "::"))  # Corrected parsing of edge
#   if (length(edge) == 2 && edge[1] != edge[2]) {  # Ensure from and to are different
#     bn <- set.arc(bn, from = edge[1], to = edge[2])
#   }
# }

# Use lapply to iterate over edges
lapply(1:nrow(edges), function(i) {
  edge <- unlist(strsplit(as.character(edges[i, ]), "::"))  # Corrected parsing of edge
  if (length(edge) == 2 && edge[1] != edge[2]) {  # Ensure from and to are different
    bn <<- set.arc(bn, from = edge[1], to = edge[2])
  }
})




# Prepare the data for the model
data <- data.frame(matrix(0, nrow = nrow(df), ncol = length(unique(unlist(df[[keyword_column]])))))
colnames(data) <- unique(unlist(df[[keyword_column]]))
for (i in 1:nrow(df)) {
  keywords <- unique(unlist(df[[keyword_column]][i]))  # Unlist and remove duplicates
  data[i, keywords] <- 1
}


write.csv(data, file = "gg.csv")

# Perform parameter learning
fitted_bn <- bn.fit(bn, data = data)


save(list = ls(all.names = TRUE), file = "all_variables.RData")

# Convert the bn object to an igraph object
g <- as.igraph(fitted_bn)

# Create a plotly object
p <- plot_ly() %>%
  add_trace(
    x = V(g)$layout[,1], y = V(g)$layout[,2], text = V(g)$name,
    mode = "markers+text", type = "scatter", textposition = "bottom",
    hoverinfo = "text", marker = list(size = 10)
  ) %>%
  add_trace(
    x = c(t(sapply(E(g), function(e) V(g)$layout[e,][1,]))),
    y = c(t(sapply(E(g), function(e) V(g)$layout[e,][2,]))),
    mode = "lines", type = "scatter", line = list(color = "#CCCCCC")
  )

# Update the layout
p <- layout(p, showlegend = FALSE, 
            xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
            yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))

# Print the plot
print(p)

# Load necessary libraries
library(dplyr)
library(ggplot2)
library(caret)
library(e1071)
library(corrplot)
library(jsonlite)
library(tm)   
library(SnowballC)
library(wordcloud)
library(tidyverse)
library(tidytext)
#Loading Data
data1 <- read.csv("Sarcasm_Headlines_Dataset11.csv", stringsAsFactors = FALSE)
# Select the first 500 rows
data1 <- data1[1:500, ]
# Preprocess text data (headlines and text)
clean_text <- function(text) {
  text <- tolower(text)                              # Convert to lowercase
  text <- removePunctuation(text)                    # Remove punctuation
  text <- removeNumbers(text)                        # Remove numbers
  text <- removeWords(text, stopwords("en"))         # Remove stop words
  text <- stripWhitespace(text)                      # Remove extra whitespace
  text <- wordStem(text)                             # Applying stemming
  return(text)
}

# preprocessing to headlines and text columns
data1$cleaned_headline <- sapply(data1$headline, clean_text)

# Combine cleaned headlines and text into a single column for analysis
data1$combined_text <- paste(data1$cleaned_headline, data1$cleaned_text)

# Text vectorization (Convert text to a bag-of-words model)
# Using TF-IDF (Term Frequency-Inverse Document Frequency)

data1_tidy <- data1 %>%
  unnest_tokens(word, combined_text) %>%
  count(word, sort = TRUE) %>%
  anti_join(stop_words)  # Remove common stopwords

# You can visualize a word cloud of the most frequent terms
wordcloud(words = data1_tidy$word, freq = data1_tidy$n, min.freq = 2, scale=c(3,0.5))

#sarcasm is in a column 'is_sarcastic' with values 0 or 1
data1$is_sarcastic <- as.factor(data1$is_sarcastic)

# Feature Selection (Use text features as predictors for sarcasm)
# use the cleaned text as a  feature
data1$combined_text <- as.factor(data1$combined_text)

# Split Data
set.seed(200)
train_index1 <- createDataPartition(data1$is_sarcastic, p = 0.8, list = FALSE)
train_data1 <- data1[train_index1, ]
test_data1 <- data1[-train_index1, ]

# Logistic Regression Hyperparameter Tuning
lr_tune_model <- train(
  is_sarcastic ~ combined_text, 
  data = train_data1, 
  method = "glm",   # Generalized Linear Model (Logistic Regression)
  family = "binomial", 
  trControl = trainControl(method = "cv", number = 5)  # Cross-validation with 5 folds
)

# Get the predictions from the Logistic Regression model
lr_pred1_tuned <- predict(lr_tune_model, newdata = test_data1)
lr_cm_tuned <- confusionMatrix(as.factor(lr_pred1_tuned), test_data1$is_sarcastic)

# k-Nearest Neighbors (k-NN) Hyperparameter Tuning
knn_grid <- expand.grid(k = c(3, 5, 7))  # Number of neighbors to consider
knn_tune_model <- train(
  is_sarcastic ~ combined_text, 
  data = train_data1, 
  method = "knn",   # k-Nearest Neighbors
  tuneGrid = knn_grid, 
  trControl = trainControl(method = "cv", number = 5)  # Cross-validation with 5 folds
)

# Get the predictions from the k-NN model
knn_pred1_tuned <- predict(knn_tune_model, newdata = test_data1)
knn_cm_tuned <- confusionMatrix(as.factor(knn_pred1_tuned), test_data1$is_sarcastic)

# Results Summary for both models
sarcastic_count <- sum(test_data1$is_sarcastic == 1)
non_sarcastic_count <- sum(test_data1$is_sarcastic == 0)

cat(sprintf("Number of Sarcastic News: %d\n", sarcastic_count))
cat(sprintf("Number of Non-Sarcastic News: %d\n", non_sarcastic_count))

# Extract accuracy for both models
lr_accuracy <- lr_cm_tuned$overall["Accuracy"]
knn_accuracy <- knn_cm_tuned$overall["Accuracy"]

# Display accuracies in the console
cat(sprintf("Logistic Regression Accuracy: %.2f%%\n", lr_accuracy * 100))
cat(sprintf("k-NN Accuracy: %.2f%%\n", knn_accuracy * 100))

# Create a data frame for graph
accuracy_data <- data.frame(
  Model = c("Logistic Regression", "k-NN"),
  Accuracy = c(lr_accuracy, knn_accuracy)
)

# Enhanced line chart with accuracy values displayed on points
ggplot(accuracy_data, aes(x = Model, y = Accuracy, group = 1)) +
  geom_line(aes(color = "Accuracy"), size = 1) +
  geom_point(aes(color = "Accuracy"), size = 3) +
  geom_text(aes(label = sprintf("%.2f%%", Accuracy * 100)), 
            vjust = -0.5, size = 5, color = "black") +  # Display accuracy values
  labs(title = "Model Comparison: Logistic Regression vs k-NN", 
       y = "Accuracy", x = "Model") +
  theme_minimal() +
  scale_y_continuous(labels = scales::percent_format())

# Histogram for data analysis (Distribution of sarcasm labels in the dataset)
ggplot(data1, aes(x = is_sarcastic)) +
  geom_bar(aes(y = ..count..), fill = "skyblue", color = "black", width = 0.7) +
  labs(title = "Distribution of Sarcastic vs Non-Sarcastic News", 
       x = "Sarcastic (1) / Non-Sarcastic (0)", 
       y = "Count") +
  theme_minimal()


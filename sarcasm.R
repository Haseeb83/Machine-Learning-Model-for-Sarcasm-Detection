# Load necessary libraries
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(e1071)
library(corrplot)
library(tm)
library(SnowballC)
library(wordcloud)
library(tidyverse)
library(tidytext)

# Load CSV file
data1 <- read.csv("Sarcasm_Headlines_Dataset_v.csv", stringsAsFactors = FALSE)
# Select the first 600 rows
data1 <- data1[1:600,]
# Inspect structure of the data
str(data1)

# Preprocess text data (headlines and text)
clean_text <- function(text) {
  text <- tolower(text)                              # Convert to lowercase
  text <- removePunctuation(text)                   # Remove punctuation
  text <- removeNumbers(text)                       # Remove numbers
  text <- removeWords(text, stopwords("en"))        # Remove stop words
  text <- stripWhitespace(text)                     # Remove extra whitespace
  text <- wordStem(text)                            # Apply stemming
  return(text)
}

# Ensure necessary columns exist
if (!all(c("headline", "article_link", "is_sarcastic") %in% colnames(data1))) {
  stop("The CSV file must contain 'headline', 'article_link', and 'is_sarcastic' columns.")
}

# Preprocess data
data1$cleaned_headline <- sapply(data1$headline, clean_text)

# Combine data
data1$combined_text <- paste(data1$cleaned_headline, data1$cleaned_text)

# Text vectorization using TF-IDF
data1_tidy <- data1 %>%
  unnest_tokens(word, combined_text) %>%
  count(word, sort = TRUE) %>%
  anti_join(stop_words)  # Remove common stopwords

# Visualizing a word cloud of the most frequent terms
wordcloud(words = data1_tidy$word, freq = data1_tidy$n, min.freq = 2, scale = c(3, 0.5))

# Convert is_sarcastic to factor
data1$is_sarcastic <- as.factor(data1$is_sarcastic)

# Feature selection
data1$combined_text <- as.factor(data1$combined_text)

# Split data
set.seed(200)
train_index1 <- createDataPartition(data1$is_sarcastic, p = 0.8, list = FALSE)
train_data1 <- data1[train_index1, ]
test_data1 <- data1[-train_index1, ]

# Random Forest Hyperparameter Tuning
rf_grid <- expand.grid(mtry = c(1:5))  # Number of variables to try for splitting a node
rf_tune_model <- train(
  is_sarcastic ~ combined_text, 
  data = train_data1, 
  method = "rf", 
  tuneGrid = rf_grid, 
  trControl = trainControl(method = "cv", number = 5)  # Cross-validation with 5 folds
)

# Print the best tuning parameters
print(rf_tune_model$bestTune)

# Get the predictions from the best model
rf_pred1_tuned <- predict(rf_tune_model, newdata = test_data1)

# Evaluate the model
rf_cm_tuned <- confusionMatrix(as.factor(rf_pred1_tuned), test_data1$is_sarcastic)

cat("Random Forest Confusion Matrix (Tuned):\n")
print(rf_cm_tuned)

# Support Vector Machine (SVM) Hyperparameter Tuning
svm_grid <- expand.grid(C = c(0.1, 1, 10), sigma = c(0.01, 0.1, 1))  # Regularization and kernel parameters
svm_tune_model <- train(
  is_sarcastic ~ combined_text, 
  data = train_data1, 
  method = "svmRadial",  # Radial kernel for SVM
  tuneGrid = svm_grid, 
  trControl = trainControl(method = "cv", number = 5)  # Cross-validation with 5 folds
)

# Print the best tuning parameters
print(svm_tune_model$bestTune)

# Get the predictions from the best model
svm_pred1_tuned <- predict(svm_tune_model, newdata = test_data1)

# Evaluate the model
svm_cm_tuned <- confusionMatrix(as.factor(svm_pred1_tuned), test_data1$is_sarcastic)

cat("SVM Confusion Matrix (Tuned):\n")
print(svm_cm_tuned)

# Results Summary: Sarcasm Prediction
sarcastic_count <- sum(test_data1$is_sarcastic == 1)
non_sarcastic_count <- sum(test_data1$is_sarcastic == 0)

cat(sprintf("Number of Sarcastic News: %d\n", sarcastic_count))
cat(sprintf("Number of Non-Sarcastic News: %d\n", non_sarcastic_count))

# Add graphs
# Histogram of sarcastic and non-sarcastic news
ggplot(data1, aes(x = is_sarcastic)) +
  geom_bar(fill = "skyblue") +
  labs(title = "Histogram of Sarcastic and Non-Sarcastic News", x = "Is Sarcastic", y = "Count")

# Scatter plot example
ggplot(data1, aes(x = nchar(headline), y = is_sarcastic)) +
  geom_point(alpha = 0.5) +
  labs(title = "Scatter Plot of Headline Length vs Sarcasm", x = "Headline Length", y = "Is Sarcastic")

# Line graph example
ggplot(data1, aes(x = 1:nrow(data1), y = nchar(headline))) +
  geom_line(color = "blue") +
  labs(title = "Line Graph of Headline Length Over Observations", x = "Observation Index", y = "Headline Length")

# Line chart for end results
end_results <- data.frame(
  Category = c("Sarcastic", "Non-Sarcastic"),
  Count = c(sarcastic_count, non_sarcastic_count)
)

ggplot(end_results, aes(x = Category, y = Count, group = 1)) +
  geom_line(color = "red", size = 1) +
  geom_point(size = 3) +
  labs(title = "Number of Sarcastic vs Non-Sarcastic News", x = "Category", y = "Count")



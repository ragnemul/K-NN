# Load the data
# https://cran.r-project.org/web/packages/MASS/MASS.pdf
data("Boston", package = "MASS")

#install.packages("caret")
library(caret)

# Inspect the data
sample(Boston, 3)

# Split the data into training and test set
set.seed(123)
training.samples <- caret::createDataPartition(Boston$medv, p = 0.8, list = FALSE)
train.data  <- Boston[training.samples, ]
test.data <- Boston[-training.samples, ]

# Fit the model on the training set
model <- caret::train(
  medv~., data = train.data, method = "knn",
  trControl = caret::trainControl("cv", number = 10),
  preProcess = c("range"),
  tuneLength = 10
)

# Plot model error RMSE vs different values of k
plot(model)

# Best tuning parameter k that minimize the RMSE
model$bestTune

# Make predictions on the test data
predictions <- predict(model,test.data)
head(predictions)

# Compute the prediction error RMSE
RMSE(predictions, test.data$medv)

# plots real vs predicted values
x = 1:dim(test.data)[1]
plot(x, test.data$medv, col = "red", type = "l", lwd=2,
     main = "Boston housing test data prediction")
lines(x, predictions, col = "blue", lwd=2)
legend("topright",  legend = c("original-medv", "predicted-medv"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.6))
grid() 

library(caret)

#install.packages('brio')
library(devtools)
library(roxygen2)
source_url("https://raw.githubusercontent.com/ragnemul/K-NN/main/Ejercicio_4.2/DrawConfusionMatrix.R")


data.df = read.csv("https://raw.githubusercontent.com/ragnemul/K-NN/main/Ejercicio_4.2/IBM-HR-Employee-Attrition.csv")
# Eliminamos datos no necesarios
# Eliminamos datos invariantes, variables independientes (no afecan al target), colineales múltiples, 

data <- subset(data.df, select = -c(EmployeeCount,StandardHours,Over18,EmployeeNumber) )

sum(is.na(data))

summary(data.df)

library(ggplot2)
library(grid)
library(gridExtra)
agePlot <- ggplot(data.df,aes(Age,fill=Attrition))+geom_density()+facet_grid(~Attrition)
travelPlot <- ggplot(data.df,aes(BusinessTravel,fill=Attrition))+geom_bar()
genderPlot <- ggplot(data.df,aes(Gender,fill=Attrition))+geom_bar()
jobLevelPlot <- ggplot(data.df,aes(JobLevel,fill=Attrition))+geom_bar()
jobInvPlot <- ggplot(data.df,aes(JobInvolvement,fill=Attrition))+geom_bar()
marPlot <- ggplot(data.df,aes(MaritalStatus,fill=Attrition))+geom_bar()
numCompPlot <- ggplot(data.df,aes(NumCompaniesWorked,fill=Attrition))+geom_bar()
overTimePlot <- ggplot(data.df,aes(OverTime,fill=Attrition))+geom_bar()
perfPlot <- ggplot(data.df,aes(PerformanceRating,fill = Attrition))+geom_bar()
StockPlot <- ggplot(data.df,aes(StockOptionLevel,fill = Attrition))+geom_bar()


grid.arrange(agePlot,travelPlot,jobLevelPlot,genderPlot,jobInvPlot,marPlot, ncol=2,numCompPlot, overTimePlot, perfPlot, StockPlot,  top = "Fig 1")

# Factorización de valores categóricos
data$MaritalStatus <- as.factor(data$MaritalStatus)
data$EducationField <- as.factor(data$EducationField)
data$Department <- as.factor(data$Department)
data$BusinessTravel <- as.factor(data$BusinessTravel)
data$Gender <- as.factor(data$Gender)
data$JobRole <- as.factor(data$JobRole)
data$OverTime <- as.factor(data$OverTime)

set.seed(123)

# Particionamiento de los datos en conjuntos de entrenamiento y test
train_split_idx <- caret::createDataPartition(data$Attrition, p = 0.8, list = FALSE)
train <- data[train_split_idx, ]
test <- data[-train_split_idx, ]


fitControl1 <- caret::trainControl(method = "cv", 
                           number = 10, 
                           classProbs = TRUE, 
                           sampling = "smote",
                           summaryFunction = caret::twoClassSummary,
                           savePredictions = TRUE)

fit_knn1 <- caret::train(Attrition ~ ., 
                         data=train, 
                         method="knn",
                         trControl = fitControl1,	
                         preProcess = c("range"),			
                         metric = "Sens",    
                         tuneGrid = expand.grid(k = 1:50))

plot(fit_knn1)

#Make predictions to expose class labels
preds_1 <- predict(fit_knn1, newdata=test, type="raw")
confussionMatrix_1 <- caret::confusionMatrix(as.factor(preds_1), as.factor(test$Attrition),positive="Yes")

# Mostramos la matriz de confusión
draw_2D_confusion_matrix(cm = confussionMatrix_1, caption = "Matriz de confusión test 2")



fitControl2 <- trainControl(method = "repeatedcv", 
                           number = 5, 
                           repeats = 10,
                           classProbs = TRUE, 
                           sampling = "down",
                           summaryFunction = twoClassSummary,
                           savePredictions = TRUE)


fit_knn2 <- caret::train(Attrition ~ ., 
                        data=train, 
                        method="knn",
                        trControl = fitControl2,	
                        preProcess = c("range"),			
                        metric = "Sens",    
                        tuneGrid = expand.grid(k = 1:50))


plot(fit_knn2)

#Make predictions to expose class labels
preds_2 <- predict(fit_knn2, newdata=test, type="raw")
confussionMatrix_2 <- caret::confusionMatrix(as.factor(preds_2), as.factor(test$Attrition),positive="Yes")


# Mostramos la matriz de confusión
draw_2D_confusion_matrix(cm = confussionMatrix_2, caption = "Matriz de confusión test 2")


library(plyr)
library(pROC)

rocs_fit1 <- llply(unique(fit_knn1$pred$obs), function(cls) {
  roc(response = fit_knn1$pred$obs==cls, predictor = fit_knn1$pred[,as.character(cls)])
  })


rocs_fit2 <- llply(unique(fit_knn2$pred$obs), function(cls) {
  roc(response = fit_knn2$pred$obs==cls, predictor = fit_knn2$pred[,as.character(cls)])
})

plot(rocs_fit1[[1]],print.auc = TRUE, print.auc.y = 0.6, col = "red") 
plot(rocs_fit2[[2]],print.auc = TRUE, print.auc.y = 0.55, col = "blue", add=T, )



############################################
# En caso de que necesitemos poner umbrales para asociar a las clases a predecir

pred_prob_fit1 <- predict(fit_knn1, newdata=test, type="prob")
calc_fit1 <- measureit(score = as.numeric(unlist(pred_prob_fit1["Yes"])), 
                            class = as.factor(test$Attrition), 
                            measure = c("TPR", "FPR", "SENS", "SPEC", "PREC", "NPV", "ACC", "FSCR"),
                            negref = "No")

pred_prob_fit2 <- predict(fit_knn2, newdata=test, type="prob")
calc_fit2 <- measureit(score = as.numeric(unlist(pred_prob_fit2["Yes"])), 
                       class = as.factor(test$Attrition), 
                       measure = c("TPR", "FPR", "SENS", "SPEC", "PREC", "NPV", "ACC", "FSCR"),
                       negref = "No")


calc_fit <- calc_fit2
calc_fit <- rapply( calc_fit2, f=function(x) ifelse(is.nan(x),0,x), how="replace" )
calc_fit <- rapply( calc_fit2, f=function(x) ifelse(is.infinite(x),1,x), how="replace" )

calc <- as.data.frame(do.call(cbind, calc_fit))

FN_cost <- 10
FP_cost <- 1

calc$cost <- calc$FN * FN_cost + calc$FP * FP_cost
min.cost.value <- min(calc$cost)
cost.threshold <- calc$Cutoff[which((calc$cost == min.cost.value))]

threshold.cost.plot <- ggplot(calc, aes(x=Cutoff)) +
  geom_line(aes(y = cost),   size = 1.5, alpha = 0.5) +
  labs(x="threshold", y="custom func.", title = "threshold effect in custom func.") +
  theme(plot.title = element_text(hjust = 0.5))

threshold.cost.plot

threshold.cost.plot + geom_vline(xintercept=cost.threshold, linetype="dashed", color="red")

cat ("Recommended threhold:", round(cost.threshold,2))

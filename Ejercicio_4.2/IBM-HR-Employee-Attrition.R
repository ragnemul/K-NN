library(caret)

#install.packages('brio')
library(devtools)
library(roxygen2)
source_url("https://raw.githubusercontent.com/ragnemul/K-NN/main/Ejercicio_4.2/DrawConfusionMatrix.R")


data.df = read.csv("https://raw.githubusercontent.com/ragnemul/K-NN/main/Ejercicio_4.2/IBM-HR-Employee-Attrition.csv")
# Eliminamos datos no necesarios
# Eliminamos datos invariantes, variables independientes (no afecan al target), colineales múltiples, 

data <- subset(data.df, select = -c(EmployeeCount,StandardHours,Over18,EmployeeNumber) )

# Eliminamos los datos nulos
sum(is.na(data))

# visualizamos el aspecto de los datos
summary(data.df)

# Mostramos algunas gráficas sobre el contenido de los datos
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

# Fijamos semilla para inicializar datos aleatorios, así podremos obtener 
#repetitividad en los experimentos
set.seed(123)

# Particionamiento de los datos en conjuntos de entrenamiento y test
train_split_idx <- caret::createDataPartition(data$Attrition, p = 0.8, list = FALSE)
train <- data[train_split_idx, ]
test <- data[-train_split_idx, ]

#############################
# primer modelo

# Estrategia de control
fitControl1 <- caret::trainControl(method = "cv", # validación cruzada
                           number = 10, # número de paquetes de muestra
                           classProbs = TRUE, # clasificación
                           # reduce la muestra de la clase mayoritaria y sintetiza nuevos puntos de datos en la clase minoritaria
                           sampling = "smote", 
                           summaryFunction = caret::twoClassSummary, # para optimizar métricas 
                           savePredictions = TRUE) # guardamos las predicciones

# proceso de entrenamiento
fit_knn1 <- caret::train(Attrition ~ ., # Clases 
                         data=train, # datos de entrenamiento
                         method="knn", # método de estimación
                         trControl = fitControl1,	# estructura de contorl
                         preProcess = c("range"),	# normalización de los datos
                         metric = "Sens",    # métrica que se utilizará para seleccionar el modelo óptimo
                         tuneGrid = expand.grid(k = 1:50)) # valores a probar de k

# mostramos la gráfica K vs métrica (validación cruzada)
plot(fit_knn1)

# Predicciones para los datos de test usando el modelo
preds_1 <- predict(fit_knn1, newdata=test, type="raw")

# Obtenemos la matriz de confusión
confussionMatrix_1 <- caret::confusionMatrix(as.factor(preds_1), as.factor(test$Attrition),positive="Yes")

# Mostramos la matriz de confusión
draw_2D_confusion_matrix(cm = confussionMatrix_1, caption = "Matriz de confusión test 2")

# primer model
###########################



#############################
# segundo modelo

# Estrategia de control
fitControl2 <- trainControl(method = "repeatedcv", # validaciones cruzadas con repetición
                           number = 5, 
                           repeats = 10,
                           classProbs = TRUE, 
                           # muestreo adicional que se realiza después del remuestreo 
                           # (normalmente para resolver los desequilibrios de clase).
                           # sub-conjunto aleatorio de todas las clases en el conjunto de entrenamiento 
                           # para que sus frecuencias de clase coincidan con la clase menos prevalente
                           sampling = "down", 
                           summaryFunction = twoClassSummary,
                           savePredictions = TRUE)

# proceso de entrenamiento
fit_knn2 <- caret::train(Attrition ~ ., 
                        data=train,  
                        method="knn",
                        trControl = fitControl2,	
                        preProcess = c("range"),			
                        metric = "Sens",    
                        tuneGrid = expand.grid(k = 1:50))


plot(fit_knn2)

# predicción de los datos de test usando el modelo fit_knn2
preds_2 <- predict(fit_knn2, newdata=test, type="raw")
confussionMatrix_2 <- caret::confusionMatrix(as.factor(preds_2), as.factor(test$Attrition),positive="Yes")

# Mostramos la matriz de confusión
draw_2D_confusion_matrix(cm = confussionMatrix_2, caption = "Matriz de confusión test 2")

# segundo modelo
#############################


######################################
# curvas ROC PARA COMPARAR LOS MODELOS

library(plyr)
library(pROC)

# ROC del primero model
rocs_fit1 <- llply(unique(fit_knn1$pred$obs), function(cls) {
  roc(response = fit_knn1$pred$obs==cls, predictor = fit_knn1$pred[,as.character(cls)])
  })

# ROC del segundo model
rocs_fit2 <- llply(unique(fit_knn2$pred$obs), function(cls) {
  roc(response = fit_knn2$pred$obs==cls, predictor = fit_knn2$pred[,as.character(cls)])
})

# Mostramos las curvas ROC
plot(rocs_fit1[[1]],print.auc = TRUE, print.auc.y = 0.6, col = "red") 
plot(rocs_fit2[[2]],print.auc = TRUE, print.auc.y = 0.55, col = "blue", add=T, )


# curvas ROC PARA COMPARAR LOS MODELOS
######################################



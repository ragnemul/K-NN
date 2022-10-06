library(class)
library(caret)

# Establecemos el directorio desde donde trabajaremos para cargar los ficheros de datos
setwd("/Users/luismg/Library/CloudStorage/GoogleDrive-ragnemul@gmail.com/My Drive/Comillas/MABA/KNN/Ejercicios/Ejercicio 4.1")

# Cargamos el fichero de datos
data = read.csv("RidingMowers.csv")

# Especificando la misma semilla para la inicialización de los números aleatorios obtendremos los mismos resultados
set.seed(123)

# --------------
# NORNMALIZACION

# Ahora normalizamos los datos. Esto se puede hacer de diferentes maneras

# Primeramente definimos la normalización a las columnas 1 y 2 del conjunto de datos 
norm.values <- preProcess(data[, 1:2], method=c("range"))
# y ahora aplicamos la normalización mediante la llamda a predict 
data.norm <- predict(norm.values, data[, 1:2])

# NORNMALIZACION
# --------------


# ---------------------------------
# MOSTRAMOS LA GRÁFICA DE LOS DATOS

plot(Lot_Size ~ Income, # campos a mostrar
     data=data.norm,    # fuente de los datos
     pch=ifelse(data$Ownership =="Owner", 1, 3),    # caracter a pintar en función de si es propietario
     col=ifelse(data$Ownership=="Owner","dark green","dark red"))    # color del caracter a pintar en función de si es propietario

# Podemos mostrar el número de registro dentro del conjunto de datos
text(data.norm$Income, data.norm$Lot_Size, rownames(data.norm), pos=4)
# mostramos la leyenda de la gráfica
legend("topright", # ubicación
       c("owner", "non-owner", "new"), # campos a mostrar
       pch = c(1, 3, 4),   # caracter a mostrar en función de owner, non-owner o new
       col=c("dark green","dark red","dark blue")) # color a mostrar en función de owner, non-owner o new

# mostramos el dato que queremos predecir
X <- data.frame("Income"=60,"Lot_Size"=20)
# tenemos que normalizar el dato
X.norm <- predict(norm.values, X)
# mostramos la posición del nuevo dato
text(X.norm, "X", col="dark blue")

# MOSTRAMOS LA GRÁFICA DE LOS DATOS
# --------------------------------- 


# Ahora hacermos la predicción del nuevo dato
class::knn (train=data.norm, # datos a usar para el entrenamiento
            test=X.norm, # Nuevo dato a predecir
            cl=data$Ownership, # factor de la clasificación
            k = 1,  # número de vecinos que usaremos para el cálculo de la predicción
            prob = TRUE) # Si queremos que nos muestre la proporción de los votos
     
# ¿A qué clase pertenecerá el nuevo dato X?

# Veamos qué ocurre si seleccionamos otro valor para el número de vecinos
class::knn (train=data.norm, test=X.norm, cl=data$Ownership, 
            k = 8,  # Ahora usaremos 8 vecinos para la predicción del nuevo dato
            prob = TRUE) 

# ¿A qué clase pertenece ahora el nuevo dato a predecir?



# ---------------------
# OBTENCIÓN DEL VALOR K 

# ---------------------------------------
# CONJUNTOS DE ENTRENAMIENTO Y VALIDACIÓN

# Obtenemos un vector con el 80% de los registros (aleatorizado)
train.idx <- sample(1:nrow(data),size=nrow(data)*0.8,replace = FALSE) 

# en train metemos los valores de data seleccionados por el indice
train <- data[train.idx,] # 70% training data

# creamos un conjunto de validación con el resto de los registros
valid <- data[-train.idx,] # remaining 30% test data

# normalizamos los valores del conjunto de entrenamiento
norm.values <- preProcess(train[, 1:2], method=c("range"))
# los datos de entrenamiento normalizados ahora están en train.norm
train.norm <- predict(norm.values, train[, 1:2])
# guardamos la catalogación original de los valores del conjunto de entrenamiento
train.labels <- data[train.idx,3]

# nornalizamos los valores usados para la validación
valid.norm <- predict(norm.values, valid[, 1:2])
# guardamos la clasificación del conjunto de validación
valid.labels <- data[-train.idx,3]

# CONJUNTOS DE ENTRENAMIENTO Y VALIDACIÓN
# ---------------------------------------


# vector para almacenar la precisión en la predicción para cada valor de k
acc <- vector("numeric",NROW(train.norm))

# hacemos un bucle para i desde 1 hasta el número de registros
for (i in 1:NROW(train.norm)) {
  # se obtiene el modelo KNN para valor de k=i
  mod <- knn (train=train.norm, test=valid.norm, cl=train.labels, k=i)
  # se almacena en el vector de precisión, contando los aciertos entre los casos
  acc[i] <- 100*sum(valid.labels == mod) / NROW(valid.norm)
  # también se podría hacer accediendo directamente a la matriz de confusión
  #acc[i] <- confusionMatrix(mod, as.factor(valid.labels))$overall[1]
}

# mostramos la gráfica con los valores de K vs % de aciertos
plot(acc, type="b", xlab="K- Value",ylab="Accuracy level %")

# Obtenemos el índice (el valor de K) cuyo valor de precisión es el más alto
which.max(acc)
# ---------------------
# OBTENCIÓN DEL VALOR K 
# ---------------------


# Función creada para MABA ML por Rodrigo de Marcos

extract_confusion_matrix <- function(cm){
  
  listOut = list(Sensibilidad  = round(as.numeric(cm$byClass[1]), 3),
                 Especificidad = round(as.numeric(cm$byClass[2]), 3),
                 Precision     = round(as.numeric(cm$byClass[5]), 3),
                 ValPredNeg    = round(as.numeric(cm$byClass[4]), 3),
                 Exactitud     = round(as.numeric(cm$overall[1]), 3),
                 ValorF1       = round(as.numeric(cm$byClass[7]), 3))
  
  return(listOut)
  
}
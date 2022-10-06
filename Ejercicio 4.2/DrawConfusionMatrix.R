# Fuente: https://stackoverflow.com/questions/23891140/r-how-to-visualize-confusion-matrix-using-the-caret-package
# Modificado para ML MABA por Rodrigo de Marcos

draw_2D_confusion_matrix <- function(cm, caption) {
  
  total <- sum(cm$table)
  res <- as.numeric(cm$table)
  
  # set the basic layout
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title(caption, cex.main=2)
  
  # create the matrix 
  classes = colnames(cm$table)
  rect(150, 430, 240, 370, col='#17833E')
  text(195, 435, classes[1], cex=1.2)
  rect(250, 430, 340, 370, col='#83173E')
  text(295, 435, classes[2], cex=1.2)
  text(125, 370, 'Predicho', cex=1.3, srt=90, font=2)
  text(245, 450, 'Real', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#83173E')
  rect(250, 305, 340, 365, col='#17833E')
  text(140, 400, classes[1], cex=1.2, srt=90)
  text(140, 335, classes[2], cex=1.2, srt=90)
  
  # add in the cm results
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(120, 0), c(40, 0), type = "n", xlab="", ylab="", main = "Métricas", xaxt='n', yaxt='n')
  text(10, 30, "Sens.", cex=1.2, font=2)
  text(10, 15, round(as.numeric(cm$byClass[1]), 3) * 1, cex=1.2)
  text(30, 30, "Espec.", cex=1.2, font=2)
  text(30, 15, round(as.numeric(cm$byClass[2]), 3) * 1, cex=1.2)
  text(50, 30, "Prec.", cex=1.2, font=2)
  text(50, 15, round(as.numeric(cm$byClass[5]), 3) * 1, cex=1.2)
  text(70, 30, "V. pred. neg.", cex=1.2, font=2)
  text(70, 15, round(as.numeric(cm$byClass[4]), 3) * 1, cex=1.2)
  text(90, 30, "Exact.", cex=1.2, font=2)
  text(90, 15, round(as.numeric(cm$overall[1]), 3) * 1, cex=1.2)
  text(110, 30, "Valor F1", cex=1.2, font=2)
  text(110, 15, round(as.numeric(cm$byClass[7]), 3) * 1, cex=1.2)
  
}
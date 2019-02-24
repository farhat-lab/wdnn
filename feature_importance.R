library("UpSetR")
require(cowplot)
library(readr)

# Drugs
drugs = c('RIF', 'INH', 'PZA', 'EMB', 'STR', 'CIP', 'CAP', 'AMK', 'MOXI', 'OFLX', 'KAN')

# Get feature category names
feature_list <- read.csv("feature_names020318.csv", header=F, stringsAsFactors=FALSE)$V1

# Create dataframe of significant mutations
feature_res = as.data.frame(matrix(0, nrow=length(feature_list), ncol=11), row.names=feature_list)
colnames(feature_res) <- drugs
for (drug in drugs) {
  file <- paste("snpsF", drug, "020318.csv", sep="")
  df_drug <- as.data.frame(read_csv(file, col_names = TRUE))
  resistant <- df_drug[1][df_drug['S/R'] == 'resistant']
  feature_res[drug][resistant,] = 1
}

# Plot
feature_res_reorder <- feature_res[,c(2,1,4,3,5,8,11,7,10,6,9)]
feature_res_reorder$CIP <- NULL
pdf("feature_importance_021218.pdf", paper="a4r", width=10, height=7, onefile = FALSE)
upset(feature_res_reorder, sets = colnames(feature_res_reorder), 
      sets.bar.color = "#56B4E9",
      matrix.color = "#7A7775",
      mainbar.y.label = "Number of Feature Intersections", 
      sets.x.label = "Features per Drug", 
      order.by = c("freq"), 
      #nintersects=12,
      keep.order = TRUE)
dev.off()


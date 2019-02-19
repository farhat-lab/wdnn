library(readr)
library(dendextend)
library(RColorBrewer)
library(extrafont)
library(colorspace)
library(gplots)
library(grid)

# Get final layer and phenotypic labels
labels <- read_csv("labels.csv", col_names = FALSE)
features <- read.csv("X_features_with_names.csv")

# get features
features_small <- features[,colSums(features) >= 1]
features_small[1] <- NULL

xx <- colnames(features)
length_features <- length(features)
for (i in 1:length_features) {
  if (grepl("eis", xx[i])) {
    print(xx[i])
  }
}

lineage_snps <- c('ahpC_G88A','embA_P913S','embA_T608N','embA_V206M','embB_E378A','embB_Q139H',
                  'embC_N394D','embC_R567H','embC_R738Q','embC_T270I','embC_V104M','embC_V981L',
                  'embR_C110Y','embR_H124R','embR_L313R','embR_Y216H','gid_A119T','gid_E92D',
                  'gid_L16R','gid_S100F','gyrA_A384V','gyrA_E21Q','gyrA_G247S','gyrA_G668D',
                  'gyrA_S95T','gyrA_T80A','gyrB_A442S','gyrB_C48T','gyrB_M330I','inhA_V78A',
                  'iniA_H481Q','katG_R463L','manB_D152N','ndh_G70T','ndh_R284W','ndh_V18A',
                  'pncA_H57D','rmlD_S257P','rpoB_C61T','rpsA_A440T','rpsA_T459P','rrs_C492T')

# Colors: #9F7000, #228B00, #0082CE, #B646C7, #CC476B, #009681

all_snps <- colnames(features_small)
i <- 1
feasible_snps <- c()
for (all_snp in all_snps) {
  for (lin_snp in lineage_snps) {
    splitted <- strsplit(lin_snp, "_")[[1]]
    if (grepl(splitted[1], all_snp) & grepl(splitted[2], all_snp)) {
      feasible_snps[i] <- all_snp
      i <- i + 1
    }
  }
}

# Use only lineage mutations
final_features <- features_small[,which(names(features_small) %in% feasible_snps)]

# Get covariance matrix
cov_matrix = cov(t(final_features))

# Get distances
dists_row <- dist(cov_matrix)
dists_row <- dist(final_features)

# Get clusters
hc_features_row <- hclust(dists_row, method='ward.D2')
hc_features_col <- hclust(dists_row, method='ward.D2')

color_palette = c("#7FC97F", "#BEAED4", "#FDC086", "#D9D95A", "#386CB0")
# Get dendrograms
dend_row <- as.dendrogram(hc_features_row)
dend_row <- color_branches(dend_row, k=5, col=color_palette)
dend_row <- rev(dend_row)

dend_col <- as.dendrogram(hc_features_col)
dend_col <- color_branches(dend_col, k=5, col=color_palette)

# Colored dendrogram branches
color_rows <-get_leaves_branches_col(dend_row)
color_rows <- color_rows[order(order.dendrogram(dend_row))]
color_cols <- get_leaves_branches_col(dend_col)
color_cols <- color_cols[order(order.dendrogram(dend_col))]

# Heatmap colors
col_func <- colorRampPalette(rev(brewer.pal(9, "Greys")))(30)

# Plot heatmap
pdf("heatmap_cov_030218_test.pdf", paper="a4r", width=9, height=7, onefile = FALSE)
#gplots::heatmap.2(as.matrix(cov_matrix),
gplots::heatmap.2(as.matrix(dists_row),
                  #main = "Agglomarative clustering of MTB isolates by mutations",
                  srtCol = 20,
                  Rowv = dend_row,
                  Colv = dend_col,
                  key.xlab = "",
                  key.title = "NA",
                  key.xtickfun = function() {
                    breaks = pretty(parent.frame()$breaks)
                    breaks = breaks[c(1,length(breaks))]
                    list(at = parent.frame()$scale01(breaks),
                         labels = breaks)
                  },
                  symkey = F, symbreaks=F,symm=F,
                  labRow = "",
                  labCol = "",
                  density.info = 'none',
                  trace="none",          
                  margins =c(2,2),
                  RowSideColors = color_rows,
                  ColSideColors = color_cols,
                  col = col_func
)
dev.off()

write.csv(color_cols, file = "lin_colors_tsne_022118.csv", row.names=FALSE)

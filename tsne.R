library(ggplot2)
library(Rtsne)
require(cowplot)
library(extrafont)
library(readr)

# Get final layer and phenotypic labels
weights <- read_csv("embedding.csv", col_names = FALSE)
labels <- read_csv("labels.csv", col_names = FALSE)

# t-SNE analysis
tsne <- Rtsne(weights, check_duplicates = FALSE, pca = FALSE,
              perplexity=30, theta=0.5, dims=2, verbose = TRUE,
              max_iter=4000)

# Save t-SNE coordinates and label data
df_coords <- data.frame(D1=tsne$Y[,1], D2=tsne$Y[,2])
df_all <- as.data.frame(c(df_coords, labels))
colnames(df_all) <- c('D1', 'D2',
                    'Rifampicin', 'Isoniazid', 'Pyrazinamide', 'Ethambutol',
                    'Streptomycin', 'Ciprofloxacin', 'Capreomycin', 'Amikacin',
                    'Moxifloxacin', 'Ofloxacin', 'Kanamycin')

# Legend text
df_all[df_all == -1] <- "Unknown"
df_all[df_all == 0] <- "Resistant"
df_all[df_all == 1] <- "Sensitive"

#df_all <- read_csv("tsne_coords.csv")

# Plot
for (i in 3:13) {
  #df_all$Status <- as.factor(sub("Status", "", df_all[,i]))
  df_all$Status <- df_all[,i]
  p <- ggplot(df_all, aes(x=df_all[,1], y=df_all[,2], color=Status)) +
    geom_point(size=0.8) +
    guides(colour = guide_legend(override.aes = list(size=4))) +
    xlab("") + ylab("") +
    ggtitle(colnames(df_all[i])) +
    scale_color_manual(values=c("#E97A7A","#7A9DE9","lightgrey")) +
    theme_light(base_size=20) +
    theme(strip.background  = element_blank(),
          strip.text.x      = element_blank(),
          axis.text.x       = element_blank(),
          axis.text.y       = element_blank(),
          axis.ticks        = element_blank(),
          axis.line         = element_blank(),
          panel.border      = element_blank(),
          legend.title      = element_blank(),
          legend.position   = "none",
          legend.background = element_rect(linetype="solid", colour ="black"),
          legend.text       = element_text(family="Georgia", size=12),
          plot.title        = element_text(family="Georgia", hjust=0.5, size=16, vjust=0))
  name <- paste("p", i, sep="")
  assign(name, p)
}

legend <- get_legend(p3 + theme(legend.position="bottom"))
plot3by4 <- plot_grid(p3, p4, p5, p6, p7, p9, p10, p11, p12, p13, ncol = 4, align="h")
plot_all <- plot_grid(plot3by4, legend, ncol = 1, rel_heights = c(1, .05))
title <- ggdraw() + draw_label("t-SNE visualization for the WDNN's representation of drug resistance status", 
                               fontface='bold', size = 20, fontfamily='Georgia') 
final_plot <- plot_grid(title, plot_all, ncol=1, rel_heights = c(0.1, 1))
save_plot("plot_all.pdf", final_plot, ncol = 4, nrow = 3, base_aspect_ratio = 1.3)

write.csv(df_all, file = "tsne_coords.csv",row.names=FALSE, na="")
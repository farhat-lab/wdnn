library(ggplot2)
library(Rtsne)
require(cowplot)
library(extrafont)
library(readr)

# Get t-SNE coordinates and lineage clustering colors
df_all <- read_csv("tsne_coords.csv", col_names = TRUE)
color_col <- read_csv("lin_colors_tsne_022118.csv", col_names = TRUE)$x

# Plot
df_all$Status <- as.factor(sub("Status", "", color_col))
p <- ggplot(df_all, aes(x=df_all[,1], y=df_all[,2], color=Status)) +
  geom_point(size=0.1) +
  guides(colour = guide_legend(override.aes = list(size=4))) +
  xlab("") + ylab("") +
  ggtitle("t-SNE visualization colored by lineage clustering") +
  #scale_color_manual(values=c("#0082CE"="#58ABDB","#009681"="#59B7AA",
  #                            "#228B00"="#6DAF5A","#9F7000"="#BC9D58",
  #                            "#B646C7"="#CC85D5","#CC476B"="#D9859C")) +
  #scale_color_manual(values=c("#009232"="#4FB36F","#A352D1"="#C08ADE",
  #                            "#917600"="#B2A04F","#CC476B"="#DA7F97",
  #                            "#008FB7"="#53B3CE")) +
  scale_color_manual(values=c("#FDC086"="#FDC086","#386CB0"="#386CB0",
                              "#BEAED4"="#BEAED4","#7FC97F"="#7FC97F",
                              "#D9D95A"="#D9D95A")) +

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
        plot.title        = element_text(family="Georgia", hjust=0.5, size=8, vjust=0, face="bold"))

save_plot("lin_tsne_022118.pdf", p, base_aspect_ratio = 1.3)


library(ggplot2)
library(ggthemes)
library(dplyr)
library(tidyr)
library(cowplot)
library(readr)
library(Rtsne)
library(plotROC)
library(caret)
library(PRROC)
library(precrec)

## Resistance is currently y=0, but we want it to be the poisitive class

##------------ Create t-sne plot ------------##
data_file = '~/Downloads/tb_data_050818/X_features.csv'
weights <- read_csv(data_file,col_names = FALSE)
labels <- read_csv("~/Downloads/tb_data_050818/labels.csv", col_names = FALSE)


# t-SNE analysis
tsne <- Rtsne(weights, check_duplicates = FALSE, pca = TRUE,
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
          legend.text       = element_text(size=12),
          plot.title        = element_text(hjust=0.5, size=16, vjust=0))
  name <- paste("p", i, sep="")
  assign(name, p)
}

legend <- get_legend(p3 + theme(legend.position="bottom"))
plot3by4 <- plot_grid(p3, p4, p5, p6, p7, p9, p10, p11, p12, p13, ncol = 4, align="h")
plot_all <- plot_grid(plot3by4, legend, ncol = 1, rel_heights = c(1, .05))
title <- ggdraw() + draw_label("t-SNE Visualization of Input Markers", 
                               fontface='bold', size = 20) 
final_plot <- plot_grid(title, plot_all, ncol=1, rel_heights = c(0.1, 1))
save_plot("tsne_plot.pdf", final_plot, ncol = 4, nrow = 3, base_aspect_ratio = 1.3)


##------------ Create AUC plots for different input features ------------##
results <- read_csv('results.csv')
results <- rbind(results,read_csv('results_select_snps.csv'))
results <- rbind(results,read_csv('results_raw_features.csv'))
results <- rbind(results,read_csv('results_raw_features_rf_lm.csv'))
results <- rbind(results,read_csv('results_select_snps_rf_lm.csv'))

results <- results[results$Algorithm != "Random Forest Raw Features",]
results <- results[results$Algorithm != "Random Forest (Select Mutations)",]

first_line_drugs <- c('inh','rif','emb','pza')
second_line_drugs <- c('str','amk','kan','cap','oflx','moxi')
feature_models <- c('WDNN Single Task (Select Mutations)','WDNN Raw Features','WDNN',
                    'Logistic Regression (Select Mutations)','Logistic Regression Raw Features','Logistic Regression')
first_line_order <- c('inh','rif','emb','pza')

drugs_for_talk <- c('pza','cap','moxi')
drugs_for_talk_order <- c('pza','cap','moxi')

feature_comparison <- results %>%
  filter(Algorithm %in% feature_models) %>%
  group_by(Algorithm,Drug) %>%
  summarize(Mean_AUC = mean(AUC), SE = sd(AUC)/sqrt(n())) %>%
  mutate(CI_LOW = Mean_AUC - 1.96*SE, CI_HIGH = Mean_AUC + 1.96*SE)

feature_comparison$Algorithm[which(feature_comparison$Algorithm == 'WDNN Single Task (Select Mutations)')] <- 'WDNN (Preselected Mutations)'
feature_comparison$Algorithm[which(feature_comparison$Algorithm == 'WDNN Raw Features')] <- 'WDNN (Common Mutations)'
feature_comparison$Algorithm[which(feature_comparison$Algorithm == 'WDNN')] <- 'WDNN (Common + Derived)'

feature_comparison$Algorithm[which(feature_comparison$Algorithm == 'Logistic Regression (Select Mutations)')] <- 'LR (Preselected Mutations)'
feature_comparison$Algorithm[which(feature_comparison$Algorithm == 'Logistic Regression Raw Features')] <- 'LR (Common Mutations)'
feature_comparison$Algorithm[which(feature_comparison$Algorithm == 'Logistic Regression')] <- 'LR (Common + Derived)'
feature_comparison$Algorithm <- factor(feature_comparison$Algorithm, levels=c('WDNN (Preselected Mutations)','WDNN (Common Mutations)',
                                                                              'WDNN (Common + Derived)','LR (Preselected Mutations)',
                                                                              'LR (Common Mutations)','LR (Common + Derived)'))


feature_first_line <- feature_comparison %>%
                       filter(Drug %in% first_line_drugs) %>%
                       mutate(Drug = factor(Drug,levels=first_line_order)) %>%
                       ggplot(aes(y=Mean_AUC,x=Algorithm, color=Algorithm)) +
                        geom_point(size=1.5) +
                        geom_errorbar(aes(ymin=CI_LOW, ymax=CI_HIGH), width=.1) +
                        facet_grid(. ~ Drug) +
                        #ylim(c(0.9,1)) +
                        coord_flip() +
                        ggtitle("Effect of Different Features on Performance: First Line Drugs") +
                        ylab('Average AUC and 95% Confidence Intervals') +
                        xlab('Input Features') +
                        theme_bw() +
                        theme(legend.position="none") 

feature_second_line <- feature_comparison %>%
                        filter(Drug %in% second_line_drugs) %>%
                        #mutate(Drug = factor(Drug,levels=first_line_order)) %>%
                        ggplot(aes(y=Mean_AUC,x=Algorithm, color=Algorithm)) +
                        geom_point(size=1.5) +
                        geom_errorbar(aes(ymin=CI_LOW, ymax=CI_HIGH), width=.1) +
                        facet_grid(. ~ Drug) +
                        #ylim(c(0.9,1)) +
                        coord_flip() +
                        ggtitle("Effect of Different Features on Performance: Second Line Drugs") +
                        ylab('Average AUC and 95% Confidence Intervals') +
                        xlab('Input Features') +
                        theme_bw() +
                        theme(legend.position="none")

job_talk_plot <- feature_comparison %>%
                  filter(Drug %in% drugs_for_talk) %>%
                  mutate(Drug = factor(Drug,levels=drugs_for_talk_order)) %>%
                  ggplot(aes(y=Mean_AUC,x=Algorithm, color=Algorithm)) +
                  geom_point(size=1.5) +
                  geom_errorbar(aes(ymin=CI_LOW, ymax=CI_HIGH), width=.1) +
                  facet_grid(. ~ Drug, scales='free') +
                  #ylim(c(0.9,1)) +
                  coord_flip() +
                  ggtitle("Effect of Input Variables on Predictive Performance (cross-validation)") +
                  ylab('Average AUC and 95% Confidence Intervals') +
                  xlab('Input Features') +
                  theme_bw(base_size=14) +
                  theme(legend.position="none") 

p <- plot_grid(feature_first_line,feature_second_line,nrow=2)
ggsave('results_111918/feature_compare.pdf',p, width=10,height = 8)

##------------ Create AUC plots for different model types ------------##
results <- read_csv('results.csv')
results <- rbind(results,read_csv('results_select_snps.csv'))
results <- rbind(results,read_csv('results_raw_features.csv'))

first_line_drugs <- c('inh','rif','emb','pza')
second_line_drugs <- c('str','amk','kan','cap','oflx','moxi')
feature_models <- c('WDNN Single Task' ,'WDNN')
first_line_order <- c('inh','rif','emb','pza')

model_comparison <- results %>%
  filter(Algorithm %in% feature_models) %>%
  group_by(Algorithm,Drug) %>%
  summarize(Mean_AUC = mean(AUC), SE = sd(AUC)/sqrt(n())) %>%
  mutate(CI_LOW = Mean_AUC - 1.96*SE, CI_HIGH = Mean_AUC + 1.96*SE)

model_comparison$Algorithm[which(model_comparison$Algorithm == 'WDNN Single Task')] <- 'Single Drug'
model_comparison$Algorithm[which(model_comparison$Algorithm == 'WDNN')] <- 'Multi-Drug'
model_comparison$Algorithm <- factor(model_comparison$Algorithm, levels=c('Single Drug','Multi-Drug'))

model_first_line <- model_comparison %>%
                        filter(Drug %in% first_line_drugs) %>%
                        mutate(Drug = factor(Drug,levels=first_line_order)) %>%
                        ggplot(aes(y=Mean_AUC,x=Algorithm, color=Algorithm)) +
                        geom_point() +
                        geom_errorbar(aes(ymin=CI_LOW, ymax=CI_HIGH), width=.1) +
                        facet_grid(. ~ Drug) +
                        #ylim(c(0.9,1)) +
                        coord_flip() +
                        ggtitle("Effect of Single vs Multi-Drug Models: First Line Drugs") +
                        ylab('Average AUC and 95% Confidence Intervals') +
                        xlab('Model Type') +
                        theme_bw() +
                        theme(legend.position="none") 

model_second_line <- model_comparison %>%
                          filter(Drug %in% second_line_drugs) %>%
                          ggplot(aes(y=Mean_AUC,x=Algorithm, color=Algorithm)) +
                          geom_point() +
                          geom_errorbar(aes(ymin=CI_LOW, ymax=CI_HIGH), width=.1) +
                          facet_grid(. ~ Drug) +
                          #ylim(c(0.9,1)) +
                          coord_flip() +
                          ggtitle("Effect of Single vs Multi-Drug Models: Second Line Drugs") +
                          ylab('Average AUC and 95% Confidence Intervals') +
                          xlab('Model Type') +
                          theme_bw() +
                          theme(legend.position="none") 

p <- plot_grid(model_first_line,model_second_line,nrow=2)
ggsave('results/model_compare.pdf',p, width=12,height = 8)


##------------ Create AUC plots ------------##
results <- read_csv('results.csv')
results <- rbind(results,read_csv('results_select_snps.csv'))
results <- rbind(results,read_csv('results_raw_features.csv'))
results <- rbind(results,read_csv('results_raw_features_rf_lm.csv'))
results <- rbind(results,read_csv('results_select_snps_rf_lm.csv'))
results <- results[results$Algorithm != "Random Forest Raw Features",]
results <- results[results$Algorithm != "Random Forest (Select Mutations)",]

first_line_drugs <- c('inh','rif','emb','pza')
second_line_drugs <- c('str','amk','kan','cap','oflx','moxi')

# Get cross-validation table
cross_val_table <-  results %>%
                      group_by(Algorithm,Drug) %>%
                      summarize(Mean_AUC = mean(AUC), SE = sd(AUC)/sqrt(n())) %>%
                      mutate(CI_LOW = Mean_AUC - 1.96*SE, CI_HIGH = Mean_AUC + 1.96*SE) %>%
                      mutate(Display_String = paste0(round(Mean_AUC,3), ' ', '(',round(CI_LOW,3),' - ',round(CI_HIGH,3),')')) %>%
                      select(Algorithm,Drug,Display_String) %>%
                      spread(Drug,Display_String)
write.csv(cross_val_table,file='results_111918/cross_validation_table_111918.csv',row.names = F)

# Get first-line drugs average AUCs with confidence intervals
first_line_auc  <- results %>%
                        filter(Drug %in% first_line_drugs) %>%
                        group_by(Algorithm,Drug) %>%
                        summarize(Mean_AUC = mean(AUC), Var = var(AUC)) %>%
                        summarize(mean_first_line = mean(Mean_AUC), SE = sqrt(sum(Var)/(n())^2) / sqrt(n())) %>%
                        mutate(CI_LOW = mean_first_line - 1.96*SE, CI_HIGH = mean_first_line + 1.96*SE) %>%
                        mutate(Mean_AUC_First_Line = paste0(round(mean_first_line,3), ' ', '(',round(CI_LOW,3),' - ',round(CI_HIGH,3),')')) %>%
                        select(Algorithm,Mean_AUC_First_Line)

# Get second-line drugs average AUCs with confidence intervals
second_line_auc  <- results %>%
  filter(Drug %in% second_line_drugs) %>%
  group_by(Algorithm,Drug) %>%
  summarize(Mean_AUC = mean(AUC), Var = var(AUC)) %>%
  summarize(mean_second_line = mean(Mean_AUC), SE = sqrt(sum(Var)/(n())^2) / sqrt(n())) %>%
  mutate(CI_LOW = mean_second_line - 1.96*SE, CI_HIGH = mean_second_line + 1.96*SE) %>%
  mutate(Mean_AUC_Second_Line = paste0(round(mean_second_line,3), ' ', '(',round(CI_LOW,3),' - ',round(CI_HIGH,3),')')) %>%
  select(Algorithm,Mean_AUC_Second_Line)

# Get total AUC drug averages with confidence intervals
total_auc  <- results %>%
  group_by(Algorithm,Drug) %>%
  summarize(Mean_AUC = mean(AUC), Var = var(AUC)) %>%
  summarize(mean_tot_auc = mean(Mean_AUC), SE = sqrt(sum(Var)/(n())^2) / sqrt(n())) %>%
  mutate(CI_LOW = mean_tot_auc - 1.96*SE, CI_HIGH = mean_tot_auc + 1.96*SE) %>%
  mutate(Total_AUC = paste0(round(mean_tot_auc,3), ' ', '(',round(CI_LOW,3),' - ',round(CI_HIGH,3),')')) %>%
  select(Algorithm,Total_AUC)

summary_AUCs_out_table <- merge(merge(first_line_auc, second_line_auc), total_auc)
write.csv(summary_AUCs_out_table,file='results_111918/summary_auc_table_010819.csv',row.names = F)

# Get rank of first-line drugs
first_line_rank  <- results %>%
                          filter(Drug %in% first_line_drugs) %>%
                          group_by(Algorithm,Drug) %>%
                          summarize(Mean_AUC = mean(AUC)) %>%
                          spread(Drug,Mean_AUC)

first_line_rank_table = data.frame(Algorithm = first_line_rank$Algorithm,First_Line_Drugs = round(rowMeans(apply(-first_line_rank[,-1],2,rank,ties='average')),2))

# Get rank of second-line drugs
second_line_rank  <- results %>%
                      filter(Drug %in% second_line_drugs) %>%
                      group_by(Algorithm,Drug) %>%
                      summarize(Mean_AUC = mean(AUC)) %>%
                      spread(Drug,Mean_AUC)

second_line_rank_table = data.frame(Algorithm = second_line_rank$Algorithm,Second_Line_Drugs = round(rowMeans(apply(-second_line_rank[,-1],2,rank,ties='average')),2))

all_rank  <- results %>%
              filter(Drug != 'cip') %>%
              group_by(Algorithm,Drug) %>%
              summarize(Mean_AUC = mean(AUC)) %>%
              spread(Drug,Mean_AUC)

all_rank_table = data.frame(Algorithm = all_rank$Algorithm,All_Drugs = round(rowMeans(apply(-all_rank[,-1],2,rank,ties='average')),2))

rank_table <- data.frame(Algorithm = as.character(all_rank_table$Algorithm), 
                         First_Line_Drugs=first_line_rank_table$First_Line_Drugs, 
                         Second_Line_Drugs=second_line_rank_table$Second_Line_Drugs,
                         All_Drugs=all_rank_table$All_Drugs)
write.csv(rank_table, file='results_111918/cross_validation_rank_table_102618.csv', row.names = F)

first_line <- results %>%
                filter(Drug %in% first_line_drugs) %>%
                group_by(Algorithm,Drug) %>%
                summarize(Mean_AUC = mean(AUC), SE = sd(AUC)/sqrt(n())) %>%
                mutate(CI_LOW = Mean_AUC - 1.96*SE, CI_HIGH = Mean_AUC + 1.96*SE)

p <- ggplot(first_line,aes(y=Mean_AUC,x=Algorithm, color=Algorithm)) +
      geom_point() +
      geom_errorbar(aes(ymin=CI_LOW, ymax=CI_HIGH), width=.1) +
      facet_grid(. ~ Drug) +
      #ylim(c(0.9,1)) +
      coord_flip() +
      ggtitle("AUC for First Line Drugs: Cross-Validation") +
      ylab('Average AUC and 95% Confidence Intervals') +
      theme_bw() +
      theme(legend.position="none") 

ggsave('results_111918/first_line_auc_111918.pdf',p, width=12,height = 8)



second_line <- results %>%
                filter(Drug %in% second_line_drugs) %>%
                group_by(Algorithm,Drug) %>%
                summarize(Mean_AUC = mean(AUC), SE = sd(AUC)/sqrt(n())) %>%
                mutate(CI_LOW = Mean_AUC - 1.96*SE, CI_HIGH = Mean_AUC + 1.96*SE)

p <- ggplot(second_line,aes(y=Mean_AUC,x=Algorithm, color=Algorithm)) +
      geom_point() +
      geom_errorbar(aes(ymin=CI_LOW, ymax=CI_HIGH), width=.1) +
      facet_grid(. ~ Drug) +
      #ylim(c(0.9,1)) +
      coord_flip() +
      ggtitle("AUC for Second Line Drugs: Cross-Validation") +
      ylab('Average AUC and 95% Confidence Intervals') +
      theme_bw() +
      theme(legend.position="none") 

ggsave('results_111918/second_line_auc_111918.pdf',p, width=12,height = 8)


##------------ Create Precision Recall Cross-val table ------------##
#results <- read_csv('results.csv')
#results <- rbind(results,read_csv('results_select_snps.csv'))
#results <- rbind(results,read_csv('results_raw_features.csv'))
#results <- rbind(results,read_csv('results_raw_features_rf_lm.csv'))
#results <- rbind(results,read_csv('results_select_snps_rf_lm.csv'))
#results <- results[results$Algorithm != "Random Forest Raw Features",]
#results <- results[results$Algorithm != "Random Forest (Select Mutations)",]

results <- read_csv('results_020719/results_pr.csv')
results <- rbind(results,read_csv('results_020719/results_raw_features_pr.csv'))
results <- rbind(results,read_csv('results_020719/results_select_snps_pr.csv'))
results <- results[results$Algorithm != "Random Forest Raw Features",]
results <- results[results$Algorithm != "Random Forest (Select Mutations)",]

first_line_drugs <- c('inh','rif','emb','pza')
second_line_drugs <- c('str','amk','kan','cap','oflx','moxi')

# Get cross-validation table
cross_val_table <-  results %>%
  group_by(Algorithm,Drug) %>%
  summarize(Mean_PR = mean(AUC_PR), SE = sd(AUC_PR)/sqrt(n())) %>%
  mutate(CI_LOW = Mean_PR - 1.96*SE, CI_HIGH = Mean_PR + 1.96*SE) %>%
  mutate(Display_String = paste0(round(Mean_PR,3), ' ', '(',round(CI_LOW,3),' - ',round(CI_HIGH,3),')')) %>%
  select(Algorithm,Drug,Display_String) %>%
  spread(Drug,Display_String)
write.csv(cross_val_table,file='results_020719/cross_validation_table_pr_020719.csv',row.names = F)

# Get first-line drugs average AUCs with confidence intervals
first_line_pr  <- results %>%
  filter(Drug %in% first_line_drugs) %>%
  group_by(Algorithm,Drug) %>%
  summarize(Mean_PR = mean(AUC_PR), Var = var(AUC_PR)) %>%
  summarize(mean_first_line = mean(Mean_PR), SE = sqrt(sum(Var)/(n())^2) / sqrt(n())) %>%
  mutate(CI_LOW = mean_first_line - 1.96*SE, CI_HIGH = mean_first_line + 1.96*SE) %>%
  mutate(Mean_PR_First_Line = paste0(round(mean_first_line,3), ' ', '(',round(CI_LOW,3),' - ',round(CI_HIGH,3),')')) %>%
  select(Algorithm,Mean_PR_First_Line)

# Get second-line drugs average AUCs with confidence intervals
second_line_pr  <- results %>%
  filter(Drug %in% second_line_drugs) %>%
  group_by(Algorithm,Drug) %>%
  summarize(Mean_PR = mean(AUC_PR), Var = var(AUC_PR)) %>%
  summarize(mean_second_line = mean(Mean_PR), SE = sqrt(sum(Var)/(n())^2) / sqrt(n())) %>%
  mutate(CI_LOW = mean_second_line - 1.96*SE, CI_HIGH = mean_second_line + 1.96*SE) %>%
  mutate(Mean_PR_Second_Line = paste0(round(mean_second_line,3), ' ', '(',round(CI_LOW,3),' - ',round(CI_HIGH,3),')')) %>%
  select(Algorithm,Mean_PR_Second_Line)

# Get total AUC drug averages with confidence intervals
total_pr  <- results %>%
  group_by(Algorithm,Drug) %>%
  summarize(Mean_PR = mean(AUC_PR), Var = var(AUC_PR)) %>%
  summarize(mean_tot_pr = mean(Mean_PR), SE = sqrt(sum(Var)/(n())^2) / sqrt(n())) %>%
  mutate(CI_LOW = mean_tot_pr - 1.96*SE, CI_HIGH = mean_tot_pr + 1.96*SE) %>%
  mutate(Total_PR = paste0(round(mean_tot_pr,3), ' ', '(',round(CI_LOW,3),' - ',round(CI_HIGH,3),')')) %>%
  select(Algorithm,Total_PR)

summary_PR_out_table <- merge(merge(first_line_pr, second_line_pr), total_pr)
write.csv(summary_PR_out_table,file='results_020719/summary_pr_table_020719.csv',row.names = F)


##------------ Create Validation AUC plots ------------##
results <- read_csv('results_validation.csv')

first_line_drugs <- c('inh','rif','emb','pza','str')
second_line_drugs <- c('amk','kan','cap','oflx','moxi')

full_validation_table  <- results %>%
  group_by(Algorithm,Drug) %>%
  summarize(Mean_AUC = mean(AUC)) %>%
  spread(Drug,Mean_AUC)

validation_table <- results %>% filter(Algorithm == 'WDNN')
write.csv(full_validation_table,file='results_102618/validation_auc_table_102618.csv', row.names = F)

first_line <- results %>%
  filter(Drug %in% first_line_drugs) %>%
  group_by(Algorithm,Drug) %>%
  summarize(Mean_AUC = mean(AUC), SE = sd(AUC)/sqrt(n()))# %>%
  #mutate(CI_LOW = Mean_AUC - 1.96*SE, CI_HIGH = Mean_AUC + 1.96*SE)

p <- ggplot(first_line,aes(y=Mean_AUC,x=Algorithm, color=Algorithm)) +
      geom_point(size=2) +
      #geom_errorbar(aes(ymin=CI_LOW, ymax=CI_HIGH), width=.1) +
      facet_grid(. ~ Drug) +
      #ylim(c(0.9,1)) +
      coord_flip() +
      ggtitle("AUC for First Line Drugs: Validation Data") +
      ylab('Average AUC and 95% Confidence Intervals') +
      theme_bw() +
      theme(legend.position="none") 

ggsave('first_line_auc_validation.pdf',p, width=12,height = 8)

second_line <- results %>%
  filter(Drug %in% second_line_drugs) %>%
  group_by(Algorithm,Drug) %>%
  summarize(Mean_AUC = mean(AUC), SE = sd(AUC)/sqrt(n())) %>%
  mutate(CI_LOW = Mean_AUC - 1.96*SE, CI_HIGH = Mean_AUC + 1.96*SE)

p <- ggplot(second_line,aes(y=Mean_AUC,x=Algorithm, color=Algorithm)) +
  geom_point(size=2) +
  #geom_errorbar(aes(ymin=CI_LOW, ymax=CI_HIGH), width=.1) +
  facet_grid(. ~ Drug) +
  #ylim(c(0.9,1)) +
  coord_flip() +
  ggtitle("AUC for Second Line Drugs: Validation Data") +
  ylab('Average AUC and 95% Confidence Intervals') +
  theme_bw() +
  theme(legend.position="none") 

ggsave('second_line_auc_validation.pdf',p, width=12,height = 8)




##------------ Create ROC plots ------------##
probs <- read_csv('preds/val_probs.csv',col_names = FALSE)
labels <- read_csv('preds/y_val.csv',col_names = FALSE)
roc_df <- data.frame(Drug=NULL, Probability=NULL,Label=NULL)
drug_names <- c('Rifampicin', 'Isoniazid', 'Pyrazinamide', 'Ethambutol',
                'Streptomycin', 'Ciprofloxacin', 'Capreomycin', 'Amikacin',
                'Moxifloxacin', 'Ofloxacin', 'Kanamycin')
first_line_drug_names <- c('Rifampicin', 'Isoniazid', 'Pyrazinamide', 'Ethambutol')
second_line_drug_names <- c('Streptomycin','Capreomycin', 'Amikacin','Moxifloxacin', 'Ofloxacin', 'Kanamycin')

for(i in 1:ncol(probs)) {
  drug_probs = probs[,i]
  drug_labels = labels[,i]
  keep <- which(drug_labels != -1)
  drug_probs <- drug_probs[keep,]
  drug_labels <- drug_labels[keep,]
  drug_df <- data.frame(Drug=drug_names[i],Probability=as.numeric(unlist(drug_probs)),Label=as.numeric(unlist(drug_labels)))
  roc_df <- rbind(roc_df,drug_df)
}
first_line_roc_df <- roc_df %>% filter(Drug %in% first_line_drug_names)
second_line_roc_df <- roc_df %>% filter(Drug %in% second_line_drug_names)

first_line_roc <- ggplot(first_line_roc_df, aes(d = Label, m = Probability, color=Drug)) + 
                    geom_roc(n.cuts = 0) +
                    geom_abline(slope=1,color='grey') +
                    labs(title="ROC Curve for First Line Drugs: Cross-Validation",
                    x='1 - Specificity',
                    y='Sensitivity') +
                    theme_bw()


second_line_roc <- ggplot(second_line_roc_df, aes(d = Label, m = Probability, color=Drug)) + 
                    geom_roc(n.cuts = 0) +
                    geom_abline(slope=1,color='grey') +
                    labs(title="ROC Curve for Second Line Drugs: Cross-Validation",
                         x='1 - Specificity',
                         y='Sensitivity') +
                    theme_bw()


cross_val_roc <- plot_grid(first_line_roc,second_line_roc)

##------------ Create ROC plots for test data, WDNN ------------##
probs <- read_csv('results_111918/figure_roc_validation/test_probs_WDNN.csv',col_names = FALSE)
labels <- read_csv('results_111918/figure_roc_validation/test_labels.csv',col_names = FALSE)
roc_df <- data.frame(Drug=NULL, Probability=NULL,Label=NULL)
drug_names <- c('Rifampicin', 'Isoniazid', 'Pyrazinamide', 'Ethambutol',
                'Streptomycin', 'Ciprofloxacin', 'Capreomycin', 'Amikacin',
                'Moxifloxacin', 'Ofloxacin', 'Kanamycin')
first_line_drug_names <- c('Rifampicin', 'Isoniazid', 'Pyrazinamide', 'Ethambutol')
second_line_drug_names <- c('Streptomycin','Capreomycin', 'Amikacin','Moxifloxacin', 'Ofloxacin', 'Kanamycin')

for(i in 1:ncol(probs)) {
  cat(sprintf("%d",i))
  drug_probs = probs[,i]
  drug_labels = labels[,i]
  keep <- which(drug_labels != -1)
  drug_probs <- drug_probs[keep,]
  drug_labels <- abs(drug_labels[keep,])
  drug_df <- data.frame(Drug=drug_names[i],Probability=as.numeric(unlist(drug_probs)),Label=as.numeric(unlist(drug_labels)))
  roc_df <- rbind(roc_df,drug_df)
}
first_line_roc_df <- roc_df %>% filter(Drug %in% first_line_drug_names)
second_line_roc_df <- roc_df %>% filter(Drug %in% second_line_drug_names)

first_line_roc <- ggplot(first_line_roc_df, aes(d = Label, m = Probability, color=Drug)) + 
                    geom_roc(n.cuts = 0) +
                    geom_abline(slope=1,color='grey') +
                    labs(title="WDNN ROC Curve for First Line Drugs: Independent Validation",
                         x='1 - Specificity',
                         y='Sensitivity') +
                    theme_bw() +
                    theme(legend.position = c(0.8, 0.2)) +
                    scale_color_manual(labels = c("Rifampicin (AUC=0.98)", "Isoniazied (AUC=0.96)","Pyrazinamide (AUC=0.88)","Ethambutol (AUC=0.92)"),values = colorblind_pal()(4))


second_line_roc <- ggplot(second_line_roc_df, aes(d = Label, m = Probability, color=Drug)) + 
                    geom_roc(n.cuts = 0) +
                    geom_abline(slope=1,color='grey') +
                    labs(title="WDNN ROC Curve for Second Line Drugs: Independent Validation",
                         x='1 - Specificity',
                         y='Sensitivity') +
                    theme_bw() +
                    theme(legend.position = c(0.8, 0.2)) +
                    scale_color_manual(labels = c("Streptomycin (AUC=0.94)", "Capreomycin (AUC=0.81)","Amikacin (AUC=0.95)","Moxifloxacin (AUC=0.90)","Ofloxacin (AUC=0.87)","Kanamycin (AUC=0.88)"),
                                       values = colorblind_pal()(6))


test_roc <- plot_grid(first_line_roc,second_line_roc)
ggsave('results_111918/validation_roc_wdnn.pdf',test_roc, width=14,height = 6)

first_line_roc_wdnn <- first_line_roc
second_line_roc_wdnn <- second_line_roc

##------------ Create ROC plots for test data, LR ------------##
probs <- read_csv('results_111918/figure_roc_validation/test_probs_lr_111918.csv',col_names = FALSE)
labels <- read_csv('results_111918/figure_roc_validation/test_labels.csv',col_names = FALSE)
roc_df <- data.frame(Drug=NULL, Probability=NULL,Label=NULL)
drug_names <- c('Rifampicin', 'Isoniazid', 'Pyrazinamide', 'Ethambutol',
                'Streptomycin', 'Ciprofloxacin', 'Capreomycin', 'Amikacin',
                'Moxifloxacin', 'Ofloxacin', 'Kanamycin')
first_line_drug_names <- c('Rifampicin', 'Isoniazid', 'Pyrazinamide', 'Ethambutol')
second_line_drug_names <- c('Streptomycin','Capreomycin', 'Amikacin','Moxifloxacin', 'Ofloxacin', 'Kanamycin')

for(i in 1:ncol(probs)) {
  drug_probs = probs[,i]
  drug_labels = labels[,i]
  keep <- which(drug_labels != -1)
  drug_probs <- drug_probs[keep,]
  drug_labels <- abs(drug_labels[keep,])
  drug_df <- data.frame(Drug=drug_names[i],Probability=as.numeric(unlist(drug_probs)),Label=as.numeric(unlist(drug_labels)))
  roc_df <- rbind(roc_df,drug_df)
}
first_line_roc_df <- roc_df %>% filter(Drug %in% first_line_drug_names)
second_line_roc_df <- roc_df %>% filter(Drug %in% second_line_drug_names)

first_line_roc <- ggplot(first_line_roc_df, aes(d = Label, m = Probability, color=Drug)) + 
  geom_roc(n.cuts = 0) +
  geom_abline(slope=1,color='grey') +
  labs(title="LR ROC Curve for First Line Drugs: Independent Validation",
       x='1 - Specificity',
       y='Sensitivity') +
  theme_bw() +
  theme(legend.position = c(0.8, 0.2)) +
  scale_color_manual(labels = c("Rifampicin (AUC=0.98)", "Isoniazied (AUC=0.97)","Pyrazinamide (AUC=0.88)","Ethambutol (AUC=0.93)"),values = colorblind_pal()(4))


second_line_roc <- ggplot(second_line_roc_df, aes(d = Label, m = Probability, color=Drug)) + 
  geom_roc(n.cuts = 0) +
  geom_abline(slope=1,color='grey') +
  labs(title="LR ROC Curve for Second Line Drugs: Independent Validation",
       x='1 - Specificity',
       y='Sensitivity') +
  theme_bw() +
  theme(legend.position = c(0.8, 0.2)) +
  scale_color_manual(labels = c("Streptomycin (AUC=0.94)", "Capreomycin (AUC=0.82)","Amikacin (AUC=0.84)","Moxifloxacin (AUC=0.92)","Ofloxacin (AUC=0.86)","Kanamycin (AUC=0.89)"),
                     values = colorblind_pal()(6))


test_roc <- plot_grid(first_line_roc,second_line_roc)
ggsave('results_111918/validation_roc_lr.pdf',test_roc, width=14,height = 6)

final_test_roc <- plot_grid(first_line_roc_wdnn, second_line_roc_wdnn, first_line_roc,second_line_roc, ncol=2)
ggsave('results_111918/validation_roc_combined.pdf',final_test_roc, width=14,height = 14)

##------------ Create precision recall curve for WDNN ------------##

probs <- read_csv('results_111918/figure_roc_validation/test_probs_WDNN.csv',col_names = FALSE)
labels <- read_csv('results_111918/figure_roc_validation/test_labels.csv',col_names = FALSE)

pr_df <- NULL
for(i in 1:ncol(probs)) {
  drug_probs = 1-probs[,i]
  drug_labels = 1-labels[,i]
  keep <- which(drug_labels != 2)
  drug_probs <- drug_probs[keep,]
  drug_labels <- abs(drug_labels[keep,])
  probs_pos <- drug_probs[drug_labels == 1]
  probs_neg <- drug_probs[drug_labels == 0]
  curr_pr_obj <- pr.curve(scores.class0 = probs_pos, scores.class1 = probs_neg, curve = T)
  curr_pr_curve <- as.data.frame(curr_pr_obj$curve)
  colnames(curr_pr_curve) <- c("Recall", "Precision", "Threshold")
  curr_pr_curve$Drug <- rep(drug_names[i],nrow(curr_pr_curve))
  pr_df <- rbind(get0("pr_df"), get0("curr_pr_curve"))
  #cat(sprintf("%f\n", roc_pr_test))
}

first_line_pr_df <- pr_df %>% filter(Drug %in% first_line_drug_names)
second_line_pr_df <- pr_df %>% filter(Drug %in% second_line_drug_names)

first_line_pr_df$Drug <- factor(first_line_pr_df$Drug, levels=first_line_drug_names)
second_line_pr_df$Drug <- factor(second_line_pr_df$Drug, levels=second_line_drug_names)

first_line_pr_wdnn <- ggplot(first_line_pr_df, aes(x=Recall, y=Precision, color=Drug)) + 
  geom_line() +
  xlim(0, 1) + ylim(0,1) +
  labs(title="WDNN Precision-Recall Curve for First Line Drugs: Independent Validation",
       x='Recall',
       y='Precision') +
  theme_bw() +
  theme(legend.position = c(0.3, 0.2)) +
  scale_color_manual(labels = c("Rifampicin (AP=0.98)", "Isoniazid (AP=0.97)","Pyrazinamide (AP=0.77)","Ethambutol (AP=0.79)"),values = colorblind_pal()(4))

second_line_pr_wdnn <- ggplot(second_line_pr_df, aes(x=Recall, y=Precision, color=Drug)) +
  geom_line() +
  xlim(0, 1) + ylim(0,1) +
  labs(title="WDNN Precision-Recall Curve for Second Line Drugs: Independent Validation",
       x='Recall',
       y='Precision') +
  theme_bw() +
  theme(legend.position = c(0.3, 0.2)) +
  scale_color_manual(labels = c("Streptomycin (AP=0.88)", "Capreomycin (AP=0.50)","Amikacin (AP=0.74)","Moxifloxacin (AP=0.63)","Ofloxacin (AP=0.74)","Kanamycin (AP=0.73)"),
                     values = colorblind_pal()(6))

##------------ Create precision recall curve for LR ------------##

probs <- read_csv('results_111918/figure_roc_validation/test_probs_lr_111918.csv',col_names = FALSE)
labels <- read_csv('results_111918/figure_roc_validation/test_labels.csv',col_names = FALSE)

pr_df <- NULL
for(i in 1:ncol(probs)) {
  cat(sprintf("%d\n", i))
  if (drug_names[i] != "Ciprofloxacin") {
    drug_probs = 1-probs[,i]
    drug_labels = 1-labels[,i]
    keep <- which(drug_labels != 2)
    drug_probs <- drug_probs[keep,]
    drug_labels <- abs(drug_labels[keep,])
    probs_pos <- drug_probs[drug_labels == 1]
    probs_neg <- drug_probs[drug_labels == 0]
    curr_pr_curve <- as.data.frame(pr.curve(scores.class0 = probs_pos, scores.class1 = probs_neg, curve = T)$curve)
    colnames(curr_pr_curve) <- c("Recall", "Precision", "Threshold")
    curr_pr_curve$Drug <- rep(drug_names[i],nrow(curr_pr_curve))
    pr_df <- rbind(get0("pr_df"), get0("curr_pr_curve"))
  }
}

first_line_pr_df <- pr_df %>% filter(Drug %in% first_line_drug_names)
second_line_pr_df <- pr_df %>% filter(Drug %in% second_line_drug_names)

first_line_pr_df$Drug <- factor(first_line_pr_df$Drug, levels=first_line_drug_names)
second_line_pr_df$Drug <- factor(second_line_pr_df$Drug, levels=second_line_drug_names)

first_line_pr_lr <- ggplot(first_line_pr_df, aes(x=Recall, y=Precision, color=Drug)) + 
  geom_line() +
  xlim(0, 1) + ylim(0,1) +
  labs(title="LR Precision-Recall Curve for First Line Drugs: Independent Validation",
       x='Recall',
       y='Precision') +
  theme_bw() +
  theme(legend.position = c(0.3, 0.2)) +
  scale_color_manual(labels = c("Rifampicin (AP=0.98)", "Isoniazid (AP=0.98)","Pyrazinamide (AP=0.78)","Ethambutol (AP=0.81)"),values = colorblind_pal()(4))

second_line_pr_lr <- ggplot(second_line_pr_df, aes(x=Recall, y=Precision, color=Drug)) + 
  geom_line() +
  xlim(0, 1) + ylim(0,1) +
  labs(title="LR Precision-Recall Curve for Second Line Drugs: Independent Validation",
       x='Recall',
       y='Precision') +
  theme_bw() +
  theme(legend.position = c(0.3, 0.2)) +
  scale_color_manual(labels = c("Streptomycin (AP=0.89)", "Capreomycin (AP=0.45)","Amikacin (AP=0.64)","Moxifloxacin (AP=0.55)","Ofloxacin (AP=0.74)","Kanamycin (AP=0.71)"),
                     values = colorblind_pal()(6))

final_test_roc <- plot_grid(first_line_pr_wdnn, second_line_pr_wdnn, first_line_pr_lr,second_line_pr_lr, ncol=2)
ggsave('results_020719/Supp_Figure_S4.pdf',final_test_roc, width=14,height = 14)


##------------ Create ROC plots for test data, LR/WDNN ENSEMBLE ------------##
probs_lr <- read_csv('results_111918/figure_roc_validation/test_probs_lr_111918.csv',col_names = FALSE)
probs_wdnn <- read_csv('results_111918/figure_roc_validation/test_probs_WDNN.csv',col_names = FALSE)
probs <- as_data_frame((probs_lr + probs_wdnn) / 2)
labels <- read_csv('results_111918/figure_roc_validation/test_labels.csv',col_names = FALSE)
roc_df <- data.frame(Drug=NULL, Probability=NULL,Label=NULL)
drug_names <- c('Rifampicin', 'Isoniazid', 'Pyrazinamide', 'Ethambutol',
                'Streptomycin', 'Ciprofloxacin', 'Capreomycin', 'Amikacin',
                'Moxifloxacin', 'Ofloxacin', 'Kanamycin')
first_line_drug_names <- c('Rifampicin', 'Isoniazid', 'Pyrazinamide', 'Ethambutol')
second_line_drug_names <- c('Streptomycin','Capreomycin', 'Amikacin','Moxifloxacin', 'Ofloxacin', 'Kanamycin')

for(i in 1:ncol(probs)) {
  drug_probs = probs[,i]
  drug_labels = labels[,i]
  keep <- which(drug_labels != -1)
  drug_probs <- drug_probs[keep,]
  drug_labels <- abs(drug_labels[keep,])
  drug_df <- data.frame(Drug=drug_names[i],Probability=as.numeric(unlist(drug_probs)),Label=as.numeric(unlist(drug_labels)))
  roc_df <- rbind(roc_df,drug_df)
}

first_line_roc_df <- roc_df %>% filter(Drug %in% first_line_drug_names)
second_line_roc_df <- roc_df %>% filter(Drug %in% second_line_drug_names)

first_line_pr_df$Drug <- factor(first_line_pr_df$Drug, levels=first_line_drug_names)
second_line_pr_df$Drug <- factor(second_line_pr_df$Drug, levels=second_line_drug_names)

first_line_roc <- ggplot(first_line_roc_df, aes(d = Label, m = Probability, color=Drug)) + 
  geom_roc(n.cuts = 0) +
  geom_abline(slope=1,color='grey') +
  labs(title="LR ROC Curve for First Line Drugs: Independent Validation",
       x='1 - Specificity',
       y='Sensitivity') +
  theme_bw() +
  theme(legend.position = c(0.8, 0.2)) +
  scale_color_manual(labels = c("Rifampicin (AUC=0.98)", "Isoniazied (AUC=0.97)","Pyrazinamide (AUC=0.88)","Ethambutol (AUC=0.93)"),values = colorblind_pal()(4))


second_line_roc <- ggplot(second_line_roc_df, aes(d = Label, m = Probability, color=Drug)) + 
  geom_roc(n.cuts = 0) +
  geom_abline(slope=1,color='grey') +
  labs(title="LR ROC Curve for Second Line Drugs: Independent Validation",
       x='1 - Specificity',
       y='Sensitivity') +
  theme_bw() +
  theme(legend.position = c(0.8, 0.2)) +
  scale_color_manual(labels = c("Streptomycin (AUC=0.94)", "Capreomycin (AUC=0.82)","Amikacin (AUC=0.84)","Moxifloxacin (AUC=0.92)","Ofloxacin (AUC=0.86)","Kanamycin (AUC=0.89)"),
                     values = colorblind_pal()(6))


test_roc <- plot_grid(first_line_roc,second_line_roc)
ggsave('results_111918/validation_roc_lr_wdnn_ensemble.pdf',test_roc, width=14,height = 6)

final_test_roc <- plot_grid(first_line_roc_wdnn, second_line_roc_wdnn, first_line_roc,second_line_roc, ncol=2)
ggsave('results_111918/validation_roc_combined.pdf',final_test_roc, width=14,height = 14)


##------------ Compute cutpoints ------------##
cut_points <- seq(0,1,by=0.01)
threshold <- 0.90
high_sensitivity <- data.frame(Drug=NULL,Sensitivity=NULL,Specificity=NULL,Threshold=NULL)
# 0 = resistant (positive classe)
# 1 = sensitive (negative class)
for(i in 1:length(unique(roc_df$Drug))) {
  drug = unique(roc_df$Drug)[i]
  if(drug != 'Ciprofloxacin') {
    drug_data <- roc_df %>% filter(Drug == drug)
    labels = rep('sensitive',nrow(drug_data))
    labels[which(drug_data$Label == 0)] <- 'resistant'
    labels <- as.factor(labels)
    current_sensitivity = 0
    final_cut = 0
    for(j in 1:length(cut_points)) {
      p = cut_points[j]
      preds  = rep('sensitive',nrow(drug_data))
      # Convert to probability of resistance
      resistance_prob <- 1-drug_data$Probability
      preds[which(resistance_prob >= p)] <- 'resistant'
      preds <- as.factor(preds)
      current_sensitivity = sensitivity(preds,labels,positive='resistant')
      #print(paste0("Cut: ",p," Sens: ",current_sensitivity," Spec: ", specificity(preds,labels,positive='resistant')))
      if(current_sensitivity < threshold) {
        final_cut = cut_points[j-1]
        break
      }
    }
    preds  = rep('sensitive',nrow(drug_data))
    preds[which(1-drug_data$Probability >= final_cut)] <- 'resistant'
    preds <- as.factor(preds)
    sens = sensitivity(preds,labels,positive='resistant')
    spec = specificity(preds,labels,positive='resistant')
    high_sensitivity <- rbind(high_sensitivity,data.frame(Drug=drug,Sensitivity=sens,Specificity=spec,Threshold=final_cut))
  }
}


high_specificity <- data.frame(Drug=NULL,Sensitivity=NULL,Specificity=NULL,Threshold=NULL)

for(i in 1:length(unique(roc_df$Drug))) {
  drug = unique(roc_df$Drug)[i]
  if(drug != 'Ciprofloxacin') {
    drug_data <- roc_df %>% filter(Drug == drug)
    labels = rep('sensitive',nrow(drug_data))
    labels[which(drug_data$Label == 0)] <- 'resistant'
    labels <- as.factor(labels)
    # Convert to probability of resistance
    resistance_prob <- 1-drug_data$Probability
    current_sensitivity = 0
    final_cut = 0
    for(j in 1:length(cut_points)) {
      p = cut_points[j]
      preds  = rep('sensitive',nrow(drug_data))
      preds[which(resistance_prob >= p)] <- 'resistant'
      preds <- as.factor(preds)
      current_specificity = specificity(preds,labels,positive='resistant')
      #print(paste0("Cut: ",p," Spec: ",current_specificity," Sens: ", sensitivity(preds,labels,positive='resistant')))
      if(current_specificity > threshold) {
        final_cut = cut_points[j]
        break
      }
    }
    preds  = rep('sensitive',nrow(drug_data))
    preds[which(resistance_prob >= final_cut)] <- 'resistant'
    preds <- as.factor(preds)
    sens = sensitivity(preds,labels,positive='resistant')
    spec = specificity(preds,labels,positive='resistant')
    high_specificity <- rbind(high_specificity,data.frame(Drug=drug,Sensitivity=sens,Specificity=spec,Threshold=final_cut))
  }
}
write.csv(high_specificity, file='results_111918/max_sum_spec_greater_90_lrwdnn_ensemble.csv', row.names=FALSE)


max_sum_results <- data.frame(Drug=NULL,Sensitivity=NULL,Specificity=NULL,Threshold=NULL)

for(i in 1:length(unique(roc_df$Drug))) {
  drug = unique(roc_df$Drug)[i]
  if(drug != 'Ciprofloxacin') {
    drug_data <- roc_df %>% filter(Drug == drug)
    labels = rep('sensitive',nrow(drug_data))
    labels[which(drug_data$Label == 0)] <- 'resistant'
    labels <- as.factor(labels)
    # Convert to probability of resistance
    resistance_prob <- 1-drug_data$Probability
    max_sum = 0
    final_cut = 0
    for(j in 1:length(cut_points)) {
      p = cut_points[j]
      preds  = rep('sensitive',nrow(drug_data))
      preds[which(resistance_prob >= p)] <- 'resistant'
      preds <- as.factor(preds)
      current_specificity = specificity(preds,labels,positive='resistant')
      current_sensitivity = sensitivity(preds,labels,positive='resistant')
      current_sum  = current_specificity + current_sensitivity
      #print(paste0("Cut: ",p," Spec: ",current_specificity," Sens: ", sensitivity(preds,labels,positive='resistant')))
      if(current_sum > max_sum) {
        final_cut = cut_points[j]
        max_sum = current_sum
      }
    }
    preds  = rep('sensitive',nrow(drug_data))
    preds[which(resistance_prob >= final_cut)] <- 'resistant'
    preds <- as.factor(preds)
    sens = sensitivity(preds,labels,positive='resistant')
    spec = specificity(preds,labels,positive='resistant')
    max_sum_results <- rbind(max_sum_results,data.frame(Drug=drug,Sensitivity=sens,Specificity=spec,Threshold=final_cut))
  }
}

write.csv(max_sum_results, file='results_111918/max_sum_lrwdnn_ensemble.csv', row.names=FALSE)


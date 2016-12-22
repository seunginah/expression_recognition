library(dplyr)
library(ggplot2)
library(tidyr)

results <- read.csv('~/academic/cs670_compvision/expression_recognition/code/results/results.csv') %>%
  group_by(feature) %>%
  summarise(DT = max(DT),
            KNN = max(KNN), 
            SVM = max(SVM))

results <- results %>% gather(model, accuracy, DT:SVM)

f1 <- c('pixel_norm1','hog', 'hog_norm1', 'hog_norm2', 'lbp', 'gabor', 'gabor_norm1', 'pixel_sharpen', 'pixel_sharpen_norm1')
f2 <- c('fiducial_points', 'fiducial_points_n1','soft_clustering', 'soft_clustering_norm1', 'soft_clustering_norm2', 'raw pixel', 'pixel_gradient')

df1 <- results %>% filter(feature %in% f1)
df2 <- results %>% filter(feature %in% f2)

df1$feature <- factor(df1$feature, levels = f1)
df2$feature <- factor(df2$feature, levels = f2)

feats1 <- ggplot(df1, aes(x = feature, y = accuracy, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_bw() +
  ggtitle('Model performances with first group of features on CK') +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  scale_fill_manual(values=c( "#0000FF","#008000","#FF0000"))

feats2 <- ggplot(df2, aes(x = feature, y = accuracy, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_bw() +
  ggtitle('Model performances with second group of features on CK') +
  theme(axis.text.x = element_text(angle = 60, hjust = 1))+
  scale_fill_manual(values=c( "#0000FF","#008000","#FF0000"))

png(filename="./results/first_features.png")
feats1
dev.off()

png(filename="./results/second_features.png")
feats2
dev.off()
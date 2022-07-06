library(dplyr)
library(tidyr)
library(ggplot2)

basepath = "/u/sauves/leucegene-shared/RES/EMBEDDINGS"
dirs = list.dirs(basepath)
wd = dirs[length(dirs)]
emb1_filename = "output_embedding_1"
emb1_data = read.csv(paste (wd, emb1_filename, sep = "/"))

emb2_filename = "output_embedding_2"
emb2_data = read.csv(paste (wd,  emb2_filename, sep = "/"))
figsize = c(6,4) * 2
min_ = min(c(min(emb1_data[,1:2]), min(emb2_data[,1:2])))
max_ = max(c(max(emb1_data[,1:2]), max(emb2_data[,1:2])))
p = ggplot(emb1_data, aes(x = emb1, y = emb2))  + 
  geom_point(size = 3, aes(color = group2)) + 
  labs(title = "Patient embedding representation (replicate 1) on Leucegene", x = "Embedding 1", y = "Embedding 2") +
  coord_fixed() + 
  scale_x_continuous(limits= c(min_,max_)) + 
  scale_y_continuous(limits= c(min_,max_)) + theme_classic() 
svg(paste(wd, paste(emb1_filename,".svg", sep = ""), sep = "/"), width = figsize[1], height = figsize[2])
p
dev.off()

g = ggplot(emb2_data, aes(x = emb1, y = emb2))  + 
  geom_point(size = 3,aes(color = group2)) + 
  labs(title = "Patient embedding representation (replicate 2) on Leucegene", x = "Embedding 1", y = "Embedding 2") +
  coord_fixed() + 
  scale_x_continuous(limits= c(min_,max_)) + 
  scale_y_continuous(limits= c(min_,max_)) + theme_classic()

ggsave(paste(wd, paste(emb2_filename,".svg", sep = ""), sep = "/"), width = figsize[1], height = figsize[2])



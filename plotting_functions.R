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

p = ggplot(emb1_data, aes(x = emb1, y = emb2))  + geom_text(aes(label = index))+ 
  labs(title = "Patient embedding representation (replicate 1) on Leucegene", x = "Embedding 1", y = "Embedding 2") +
  coord_fixed() + 
  scale_x_continuous(limits= c(-5,5)) + 
  scale_y_continuous(limits= c(-5,5)) + theme_classic()
png(paste(wd, paste(emb1_filename,".png", sep = ""), sep = "/"))
p
dev.off()

g = ggplot(emb2_data, aes(x = emb1, y = emb2))  + geom_text(aes(label = index))+ 
  labs(title = "Patient embedding representation (replicate 2) on Leucegene", x = "Embedding 1", y = "Embedding 2") +
  coord_fixed() + 
  scale_x_continuous(limits= c(-5,5)) + 
  scale_y_continuous(limits= c(-5,5)) + theme_classic()

png(paste(wd, paste(emb2_filename,".png", sep = ""), sep = "/"))
g
dev.off()


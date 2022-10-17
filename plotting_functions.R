library(dplyr)
library(tidyr)
library(ggplot2)

basepath = "/u/sauves/leucegene-shared/RES/EMBEDDINGS"
dirs = list.dirs(basepath)
wd = dirs[length(dirs)]
emb1_filename = "output_embedding_1"
emb1_data = read.csv(paste (wd, emb1_filename, sep = "/"))

#emb2_filename = ""
#emb2_data = read.csv(paste (wd,  emb2_filename, sep = "/"))

#emb3_filename = "output_embedding_1_interpolated"
#emb3_data = read.csv(paste (wd,  emb3_filename, sep = "/"))

figsize = c(6,4) * 2
min_ = min(emb1_data[,1:2])
max_ = max(emb1_data[,1:2])

p = ggplot(emb1_data, aes(x = emb1, y = emb2))  + 
  geom_point(size = 3, aes(color = group2)) + 
  labs(title = "Patient embedding representation on Leucegene by subtype", x = "Embedding 1", y = "Embedding 2") +
  coord_fixed() + 
  scale_x_continuous(limits= c(min_,max_)) + 
  scale_y_continuous(limits= c(min_,max_)) + theme_classic() 
svg(paste(wd, paste(emb1_filename,".svg", sep = ""), sep = "/"), width = figsize[1], height = figsize[2])
p
dev.off()
test_pos = emb1_data %>% filter(group3 == 1)
posx = test_pos$emb1 
posy = test_pos$emb2
pos_label = paste("test (",round(posx, 3), ",", round(posy,3), ")", sep = "")

p2 = ggplot(emb1_data, aes(x = emb1, y = emb2))  + 
  geom_point(size = 3, aes(color = "grey")) + 
  labs(title = "Patient embedding representation on Leucegene by subtype", x = "Embedding 1", y = "Embedding 2") +
  coord_fixed() + 
  scale_x_continuous(limits= c(min_,max_)) + 
  scale_y_continuous(limits= c(min_,max_)) + theme_classic() +
  annotate("text", x = posx  + 2.5 , y = posy, label = pos_label) +
  annotate("segment", x = posx + 1, xend = posx, y = posy, yend = posy,
           arrow = arrow() )
svg(paste(wd, paste(emb1_filename,"_tst.svg", sep = ""), sep = "/"), width = figsize[1], height = figsize[2])
p2
dev.off()

traject_file = "interpolation_trajectories"
traject_data = read.csv(paste(wd, traject_file, sep = "/"))
traject_data %>% head()
color_hue = traject_data %>% group_by(init_pos) %>% summarise() %>% mutate(tmp = 1, id =cumsum(tmp)) %>% select(init_pos, id) %>% mutate(color_hue = id / length(unique(traject_data$init_pos)))
to_plot = traject_data %>% left_join(color_hue)
g = ggplot(emb1_data, aes(x = emb1, y = emb2))  + 
  geom_point(size = 3, col = "grey") + 
  geom_point(data = to_plot, aes(x = emb1, y = emb2, group = factor(color_hue), colour = factor(color_hue)), size = 0.5) + 
  labs(title = "Patient embedding representation on Leucegene with interpolation trajectories", x = "Embedding 1", y = "Embedding 2") +
  coord_fixed() + 
  scale_x_continuous(limits= c(min_,max_)) + 
  scale_y_continuous(limits= c(min_,max_)) + theme_classic() +
  #annotate("text", x = posx, x = posy, label = pos_label) 
  annotate("text", x = posx  + 2.5 , y = posy, label = pos_label) +
  annotate("segment", x = posx + 1, xend = posx, y = posy, yend = posy,
           arrow = arrow() )

svg(paste(wd, paste(emb1_filename,"_traject.svg", sep = ""), sep = "/"), width = figsize[1], height = figsize[2])
g
dev.off()

ggsave(paste(wd, paste(emb1_filename,"_traject.png", sep = ""), sep = "/"), width = figsize[1] , height = figsize[2], dpi = 300)
g
dev.off()



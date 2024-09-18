library(ggplot2)
library(dplyr)

### pos ###

pos_results <- read.csv('../new_pos_results.txt', header = T, sep = ' ', row.names = NULL)
head(pos_results)
pos_results <- subset(pos_results, Metric == 'f1')
pos_results$Total <- pos_results$Size + pos_results$Select_size

pos_results %>%
  ggplot(aes(Total, Value, group = Treebank, color = Treebank)) + 
  geom_point(aes(color = Treebank), alpha=.01) +
  scale_color_brewer(palette = "Accent") +
  #  scale_color_manual(values = wes_palette('Darjeeling2', n = 6)) + 
  #  scale_x_continuous(breaks=seq(0.5,5,1)) + 
  labs(x = "Training set size") +
  labs(y = 'F1 score') +
  geom_smooth(mapping = aes(Total, Value), method = 'gam') +
  facet_wrap(~ Treebank, ncol = 8) +
  theme_classic() + 
  theme(legend.position="top")  + 
  theme(text = element_text(size=20, family="Times")) +
  ylim(0, 1)

IE_results <- subset(pos_results, Language_Family == 'IE')

IE_results %>%
  ggplot(aes(Total, Value)) + #, group = Treebank, color = Treebank)) + 
  geom_point(aes(color = Treebank), alpha=.01) +
#  scale_color_brewer(palette = "Accent") +
  #  scale_color_manual(values = wes_palette('Darjeeling2', n = 6)) + 
  #  scale_x_continuous(breaks=seq(0.5,5,1)) + 
  labs(x = "Training set size") +
  labs(y = 'F1 score') +
  geom_smooth(mapping = aes(Total, Value), method = 'gam') +
  facet_wrap(~ Treebank, ncol = 8) +
  theme_classic() + 
  theme(legend.position="none")  + 
  theme(text = element_text(size=15, family="Times")) +
  ylim(0, 1)

select <- subset(pos_results, Treebank %in% c('UD_Latin-ITTB', 'UD_Wolof-WTB'))

select %>%
  ggplot(aes(Select_size, Value, group = Treebank, color = Treebank)) + 
  geom_point(aes(color = Treebank), alpha=.5, size=1) +
  scale_color_brewer(palette = "Accent") +
#    scale_color_manual(values = wes_palette('Darjeeling2', n = 2)) + 
  #  scale_x_continuous(breaks=seq(0.5,5,1)) + 
  labs(x = "Training set size") +
  labs(y = 'F1 score') +
  geom_smooth(method="gam", formula = y ~ s(x, bs = "cs", k=5)) +
 # geom_smooth(mapping = aes(Total, Value, fill = Select_size), method = 'gam') +
  #  facet_wrap(~ R, ncol = 3) +
  theme_classic() + 
  theme(legend.position="none")  + 
  theme(text = element_text(size=20, family="Times")) +
  ylim(0.65, 1) 


UD_Irish_IDT <- subset(pos_results, Treebank == 'UD_Irish-IDT')

UD_Irish_IDT %>%
  ggplot(aes(Total, Value, group = Treebank, color = Treebank)) + 
  geom_point(aes(color = Treebank)) +
  scale_color_brewer(palette = "Accent") +
  #  scale_color_manual(values = wes_palette('Darjeeling2', n = 6)) + 
  #  scale_x_continuous(breaks=seq(0.5,5,1)) + 
  labs(x = "Training set size") +
  labs(y = 'F1 score') +
  geom_smooth(mapping = aes(Total, Value), method = 'gam') +
#  facet_wrap(~ R, ncol = 3) +
  theme_classic() + 
  theme(legend.position="top")  + 
  theme(text = element_text(size=20, family="Times")) +
  ylim(0, 1) + 
  ggtitle('UD_Irish-IDT')

### Attempting to fit growth curve models
UD_Irish_IDT$Total_log = log(UD_Irish_IDT$Total)

UD_Irish_IDT_nonlinear <- brm(
  bf(Value  ~ upperAsymptote * exp(-exp(-growthRate*(Total_log-inflection))),
     upperAsymptote ~ 1,
     growthRate ~ 1,
     inflection ~ 1,
     nl = TRUE),
  data = UD_Irish_IDT,
  prior = c(
    prior(uniform(0, 1), nlpar = "upperAsymptote", lb = 0, ub = 1),
    prior(uniform(0, 3), nlpar = "growthRate", lb = 0, ub = 3),
    prior(uniform(0, 5), nlpar = "inflection", lb = 0, ub = 5)
  ),
  file = "UD_Irish-IDT_nonlinear",
  iter = 4000,
  chains = 2,
  cores = 2)

######## Older stuff ##########


pos_results$Select_size <- as.numeric(pos_results$Select_size)
pos_results$Value <- as.numeric(pos_results$Value)

subset(pos_results, Metric == 'f1' & Size == 500) %>% 
  ggplot(aes(Select_size, Value, group = Size, color = Size)) +
  geom_line() +
#  geom_point() +
  #  scale_shape_manual(values = c(3, 16, 3, 16)) +
#  scale_color_manual(values = c("steelblue",  "mediumpurple4")) +  
  facet_wrap( ~ Language) +
  scale_x_continuous(breaks=seq(0, 2000, 100)) +
  #  scale_y_continuous(breaks=seq(0, 1300, 100)) +
  theme_classic() + 
  theme(text = element_text(size=12)) + #, family="Times"
  theme(legend.position="top") +
  xlab("selection size") + 
  ylab("F1") + 
  guides(color = guide_legend(nrow = 2)) + 
  theme(legend.title = element_blank())

subset(pos_results, Metric == 'f1' & Size == 1000) %>% 
  ggplot(aes(Select_size, Value, group = Size, color = Size)) +
  geom_line() +
  #  geom_point() +
  #  scale_shape_manual(values = c(3, 16, 3, 16)) +
  #  scale_color_manual(values = c("steelblue",  "mediumpurple4")) +  
  facet_wrap( ~ Language) +
  scale_x_continuous(breaks=seq(0, 1500, 100)) +
  #  scale_y_continuous(breaks=seq(0, 1300, 100)) +
  theme_classic() + 
  theme(text = element_text(size=12)) + #, family="Times"
  theme(legend.position="top") +
  xlab("selection size") + 
  ylab("F1") + 
  guides(color = guide_legend(nrow = 2)) + 
  theme(legend.title = element_blank())

subset(pos_results, Metric == 'f1' & Size == 1500) %>% 
  ggplot(aes(Select_size, Value, group = Size, color = Size)) +
  geom_line() +
  #  geom_point() +
  #  scale_shape_manual(values = c(3, 16, 3, 16)) +
  #  scale_color_manual(values = c("steelblue",  "mediumpurple4")) +  
  facet_wrap( ~ Language) +
  scale_x_continuous(breaks=seq(0, 1000, 100)) +
  #  scale_y_continuous(breaks=seq(0, 1300, 100)) +
  theme_classic() + 
  theme(text = element_text(size=12)) + #, family="Times"
  theme(legend.position="top") +
  xlab("selection size") + 
  ylab("F1") + 
  guides(color = guide_legend(nrow = 2)) + 
  theme(legend.title = element_blank()) 

subset(pos_results, Metric == 'f1' & Size == 2000 & Select_size <= 500) %>% 
  ggplot(aes(Select_size, Value, group = Size, color = Size)) +
  geom_line() +
  #  geom_point() +
  #  scale_shape_manual(values = c(3, 16, 3, 16)) +
  #  scale_color_manual(values = c("steelblue",  "mediumpurple4")) +  
  facet_wrap( ~ Language) +
  scale_x_continuous(breaks=seq(0, 500, 100)) +
  #  scale_y_continuous(breaks=seq(0, 1300, 100)) +
  theme_classic() + 
  theme(text = element_text(size=12)) + #, family="Times"
  theme(legend.position="top") +
  xlab("selection size") + 
  ylab("F1") + 
  guides(color = guide_legend(nrow = 2)) + 
  theme(legend.title = element_blank()) 

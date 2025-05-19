library(ggplot2)
library(dplyr)

#### Full results
pos_1000 <- read.csv('../results/finalized_1000_pos_results.txt', header = T, sep = ' ', row.names = NULL)
pos_1000 <- subset(pos_1000, Language != "Latin" & Treebank != 'UD_German-HDT')
pos_1000$Total <- as.numeric(pos_1000$Total)
head(pos_1000)

pos_1000_average <- read.csv('../results/finalized_1000_pos_results_new.txt', header = T, sep = ' ', row.names = NULL)
pos_1000_average <- subset(pos_1000_average, Language != "Latin" & Treebank != 'UD_German-HDT')
pos_1000_average$Total <- as.numeric(pos_1000_average$Total)
pos_1000_average <- subset(pos_1000_average, Metric == 'f1'  & Seed == 'average')

#### Full results for individual tags
full_pos_1000_individual_tag <- read.csv('../results/finalized_1000_pos_results_individual_tag.txt', header = T, sep = ' ', row.names = NULL)
#pos_1000_individual_tag <- na.omit(pos_1000_individual_tag)
#pos_1000_individual_tag <- subset(pos_1000_individual_tag, as.integer(Support) != 0) # !is.na(pos_1000_individual_tag$Value))
head(full_pos_1000_individual_tag)

pos_1000_individual_tag <- subset(full_pos_1000_individual_tag, Metric == 'f1'  & Seed == '0')
pos_1000_individual_tag$Total <- as.numeric(pos_1000_individual_tag$Total)
pos_1000_individual_tag$Value <- as.numeric(pos_1000_individual_tag$Value)

### Irish Walk-through also plot for different treebanks 

irish_pos_1000 <- subset(pos_1000_average, Language == 'Irish' & Total <= 100000)

irish_IDT <- subset(irish_pos_1000, Treebank == 'UD_Irish-IDT')
irish_IDT$Method[irish_IDT$Method == 'al'] = 'AL'
irish_IDT$Method[irish_IDT$Method == 'random'] = 'Random'
irish_IDT_al <- subset(irish_pos_1000, Treebank == 'UD_Irish-IDT' & Method == 'al')
irish_IDT_random <- subset(irish_pos_1000, Treebank == 'UD_Irish-IDT' & Method == 'random')

irish_TwittIrish <- subset(irish_pos_1000, Treebank == 'UD_Irish-TwittIrish')
irish_TwittIrish_al <- subset(irish_pos_1000, Treebank == 'UD_Irish-TwittIrish' & Method == 'al')
irish_TwittIrish_random <- subset(irish_pos_1000, Treebank == 'UD_Irish-TwittIrish' & Method == 'random')

irish_IDT %>%
  ggplot(aes(Total, Value, group = Method, color = Method, linetype = Method)) + 
  geom_point(aes(color = Method), alpha=.01) +
  geom_line() +
  labs(x = "Training set size") +
  labs(y = 'F1 score') +
  #  geom_smooth(mapping = aes(Total, Value), method = 'gam') +
  facet_wrap(~ Treebank, ncol = 2) +
  theme_classic() + 
  theme(legend.position = 'top')  + 
  theme(text = element_text(size=17, family="Times")) +
  ylim(0.4, 1) #+
#  scale_x_continuous(breaks = seq(50, 100001)) #, by = 5000))


#### Plot of individual tags for irish

irish_pos_1000_individual_tag <- subset(pos_1000_individual_tag, Total <= 100000 & Treebank == 'UD_Irish-IDT' & Tag != 'kl')

subset(irish_pos_1000_individual_tag, Method == 'al') %>%
  ggplot(aes(Total, Value, group = Treebank, color = Treebank, linetype = Treebank)) + 
  #  geom_point(aes(color = Treebank), alpha=.01) +
  scale_color_manual(values = c("darkblue")) + 
  geom_line() +
  labs(x = "Training set size") +
  labs(y = 'F1 score') +
  #  geom_smooth(mapping = aes(Total, Value), method = 'gam') +
  facet_wrap(~ Tag, ncol = 9) +
  theme_classic() + 
  theme(legend.position="none")  + 
  theme(text = element_text(size=12, family="Times")) +
  ylim(0, 1) +
  theme( #axis.title.x=element_blank(),
    axis.text.x=element_blank(),
    axis.ticks.x=element_blank())

#### Comparing individual tags, when would the score reaches >= 0.85
tag_select<-subset(irish_pos_1000_individual_tag, Tag=='PUNCT')
max(tag_select$Value)
tag_select<-subset(tag_select, Value<0.85)
tail(tag_select, 1)$Total

### INTJ and NUM looked weird

num <- subset(irish_pos_1000_individual_tag, Tag == 'NUM')

num %>%
  ggplot(aes(Total, Value, group = Treebank, color = Treebank, linetype = Treebank)) + 
  #  geom_point(aes(color = Treebank), alpha=.01) +
  scale_color_manual(values = c("darkblue")) + 
  geom_line() +
  labs(x = "Training set size") +
  labs(y = 'F1 score') +
  #  geom_smooth(mapping = aes(Total, Value), method = 'gam') +
  #  facet_wrap(~ Tag, ncol = 6) +
  theme_classic() + 
  theme(legend.position="none")  + 
  theme(text = element_text(size=15, family="Times")) +
  ylim(0, 1) 

#### Plot of KL divergence for irish

for (family in as.vector(unique(pos_1000$Language_Family))){
  data=subset(pos_1000,Language_Family==family) 
  print(c(family, unique(data$Language)))}

irish_pos_1000_kl <- subset(pos_1000, Total <= 100000 & Seed == 'Irish' & Metric == 'kl') #column needs to be fixed in python script
irish_pos_1000_kl_IDT <- subset(irish_pos_1000_kl, Treebank=='UD_Irish-IDT')
irish_pos_1000_kl_TwittIrish <- subset(irish_pos_1000_kl, Treebank=='UD_Irish-TwittIrish')

irish_pos_1000_kl_IDT <- head(irish_pos_1000_kl_IDT, nrow(irish_pos_1000_kl_IDT)/3)
irish_pos_1000_kl_TwittIrish <- head(irish_pos_1000_kl_TwittIrish, nrow(irish_pos_1000_kl_TwittIrish)/3)
irish_pos_1000_kl_update <- rbind(irish_pos_1000_kl_IDT, irish_pos_1000_kl_TwittIrish)

subset(irish_pos_1000_kl_IDT, Method == 'al') %>%
  ggplot(aes(Total, Value, group = Treebank, color = Treebank, linetype = Treebank)) + 
  geom_line() +
  labs(x = "Training set size") +
  labs(y = 'KL divergence') +
  theme_classic() + 
  theme(legend.position="none")  + 
  theme(text = element_text(size=16, family="Times")) +
  facet_wrap(~ Treebank)

irish_IDT_kl <- subset(irish_pos_1000_kl_IDT, Method == 'al' & Value < 1  & Select_size <= 90000)
irish_IDT_f1 <- subset(irish_IDT_al, Total <= 100000 & Select_size <= 90000)
irish_IDT_f1 <- subset(irish_IDT_f1, Select_size %in% as.vector(irish_IDT_kl$Select_size))
irish_IDT_kl$f1 <- irish_IDT_f1$Value

summary(lm(Value ~ Total, data = irish_IDT_kl)) 
summary(lm(f1 ~ Value, data = irish_IDT_kl))
cor(irish_IDT_kl$Total, irish_IDT_kl$Value, method = "spearman")


### Analyzing relationship between KL divergence and training set sizes

kl_trainsize_corr = cor(irish_pos_1000_kl$Total, irish_pos_1000_kl$Value, method = "spearman")

irish_pos_1000_individual_tag_al <- subset(irish_pos_1000_individual_tag, Method == 'al')

#### Analyzing relationship between tag probability / word entropy and training set sizes for each tag
for (tag in as.vector(unique(irish_pos_1000_individual_tag_al$Tag))){
  print(tag)
  tag_subset <- subset(irish_pos_1000_individual_tag_al, Tag == tag & as.numeric(Tag_Prob) > 0)
  #  print(tag_subset)
  tagprob_trainsize_corr = cor(tag_subset$Total, as.numeric(tag_subset$Tag_Prob), method = 'spearman')
  tag_subset <- subset(tag_subset, Tag_word_entropy != 'None')
  wordentropy_trainsize_corr = cor(tag_subset$Total, as.numeric(tag_subset$Tag_word_entropy), method = 'spearman')
  print(c(tagprob_trainsize_corr, wordentropy_trainsize_corr))
}



### Some specific analysis about irish-SynTagRus; tag bigram entropy
irish_regression_data <- irish_pos_1000_individual_tag_al
irish_regression_data <- subset(irish_regression_data, Tag_word_entropy != 'None')
irish_regression_data$Tag_Prob <- as.numeric(irish_regression_data$Tag_Prob)
irish_regression_data$Tag_word_entropy <- as.numeric(irish_regression_data$Tag_word_entropy)
irish_regression_data$Tag_syntax_entropy <- as.numeric(irish_regression_data$Tag_syntax_entropy)

lm <- lmer(Value ~ Tag_Prob + Tag_word_entropy + Tag_syntax_entropy  + (1| Tag), data = irish_regression_data)
summary(lm)

### Fitting regression to every treebank
entropy_file <- read.csv('../results/50_pos_entropy.txt', header = T, sep = ' ', row.names = NULL)

treebank <- 'UD_irish-Poetry'

pos_1000_individual_tag <- subset(pos_1000_individual_tag, !(Language %in% c('irish', 'Latin')))

for (treebank in as.vector(unique(pos_1000_individual_tag$Treebank))){
  print(treebank)
  regression_data <- subset(pos_1000_individual_tag, Total <= 100000 & Treebank == treebank & Tag != 'kl')
  regression_data$Tag_syntax_entropy <- subset(entropy_file, Total <= 100000 & Treebank == treebank & Metric == 'f1')$Tag_syntax_entropy
  regression_data <- subset(regression_data, Word_entropy != 'None')
  regression_data$Tag_prob <- as.numeric(regression_data$Tag_prob)
  regression_data$Word_entropy <- as.numeric(regression_data$Word_entropy)
  regression_data$Tag_syntax_entropy <- as.numeric(regression_data$Tag_syntax_entropy)
  
  lm <- lmer(Value ~ Tag_prob + Word_entropy + Tag_syntax_entropy  + (1| Tag), data = regression_data)
  print(summary(lm))
  print(confint(lm))
  print('')
  print('')
}


### Fitting regression to individual tag

for (tag in as.vector(unique(irish_pos_1000_individual_tag$Tag))){
  print(tag)
  tag_subset <- subset(irish_pos_1000_individual_tag, Tag == tag & as.numeric(Tag_prob) > 0 &  Word_entropy != 'None')
  tag_subset$Tag_prob <- as.numeric(tag_subset$Tag_prob)
  tag_subset$Word_entropy <- as.numeric(tag_subset$Word_entropy)
  tag_subset$Tag_syntax_entropy <- subset(irish_syntax_entropy, Tag == tag & Select_size %in% as.vector(tag_subset$Select_size))$Value
  #  print(tag_subset)
  tag_lm = lm(Value ~ Tag_prob + Word_entropy + Tag_syntax_entropy, data = tag_subset)
  print(summary(tag_lm))
  print('')
  print('')
}


### Russian Walk-through also plot for different treebanks (add different initial sizes later)

russian_pos_1000 <- subset(pos_1000, Language == 'Russian' & Total <= 100000)

russian_syntagrus <- subset(russian_pos_1000, Treebank == 'UD_Russian-SynTagRus')

russian_pos_1000 %>%
  ggplot(aes(Total, Value, group = Treebank, color = Treebank, linetype = Treebank)) + 
  geom_point(aes(color = Treebank), alpha=.01) +
  geom_line() +
  labs(x = "Training set size") +
  labs(y = 'F1 score') +
  #  geom_smooth(mapping = aes(Total, Value), method = 'gam') +
  facet_wrap(~ Treebank, ncol = 2) +
  theme_classic() + 
  theme(legend.position="none")  + 
  theme(text = element_text(size=16, family="Times")) +
  ylim(0.4, 1) #+
#  scale_x_continuous(breaks = seq(50, 100001)) #, by = 5000))

### Russian random sampling

random_pos_1000 <- read.csv('../results/new_1000_pos_results_random.txt', header = T, sep = ' ', row.names = NULL)
random_pos_1000$Total <- as.numeric(random_pos_1000$Total)

russian_random_pos_1000 <- subset(random_pos_1000, Language == 'Russian' & Total <= 100000 & Metric=='f1')
russian_random_pos_1000$Language <- rep('Russian', nrow(russian_random_pos_1000))
russian_random_pos_1000$Language_Family <- rep('IE', nrow(russian_random_pos_1000))
russian_pos_1000$Method <- rep('AL', nrow(russian_pos_1000))
russian_random_pos_1000$Method <- rep('Random', nrow(russian_random_pos_1000))
together <- rbind(russian_pos_1000, russian_random_pos_1000)
together$Treebank[together$Treebank == 'UD_Russian-GSD'] = 'GSD'
together$Treebank[together$Treebank == 'UD_Russian-Poetry'] = 'Poetry'
together$Treebank[together$Treebank == 'UD_Russian-SynTagRus'] = 'SynTagRus'
together$Treebank[together$Treebank == 'UD_Russian-Taiga'] = 'Taiga'

together %>%
  ggplot(aes(Total, Value, group = Method, color = Method, linetype = Method)) + 
  geom_point(aes(color = Method), alpha=.01) +
  geom_line() +
  labs(x = "Training set size") +
  labs(y = 'F1 score') +
  #  geom_smooth(mapping = aes(Total, Value), method = 'gam') +
  facet_wrap(~ Treebank, ncol = 2) +
  theme_classic() + 
  theme(legend.position = 'top')  + 
  theme(text = element_text(size=17, family="Times")) +
  ylim(0.4, 1) #+
#  scale_x_continuous(breaks = seq(50, 100001)) #, by = 5000))


#### Plot of individual tags for Russian

russian_pos_1000_individual_tag <- subset(pos_1000_individual_tag, Total <= 100000 & Treebank == 'UD_Russian-SynTagRus' & Tag != 'kl')

russian_pos_1000_individual_tag %>%
  ggplot(aes(Total, Value, group = Treebank, color = Treebank, linetype = Treebank)) + 
#  geom_point(aes(color = Treebank), alpha=.01) +
  scale_color_manual(values = c("darkblue")) + 
  geom_line() +
  labs(x = "Training set size") +
  labs(y = 'F1 score') +
  #  geom_smooth(mapping = aes(Total, Value), method = 'gam') +
  facet_wrap(~ Tag, ncol = 9) +
  theme_classic() + 
  theme(legend.position="none")  + 
  theme(text = element_text(size=12, family="Times")) +
  ylim(0, 1) +
  theme( #axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

#### Comparing individual tags, when would the score reaches >= 0.85
tag_select<-subset(russian_pos_1000_individual_tag, Tag=='PUNCT')
max(tag_select$Value)
tag_select<-subset(tag_select, Value<0.85)
tail(tag_select, 1)$Total

### INTJ and NUM looked weird

num <- subset(russian_pos_1000_individual_tag, Tag == 'NUM')

num %>%
  ggplot(aes(Total, Value, group = Treebank, color = Treebank, linetype = Treebank)) + 
  #  geom_point(aes(color = Treebank), alpha=.01) +
  scale_color_manual(values = c("darkblue")) + 
  geom_line() +
  labs(x = "Training set size") +
  labs(y = 'F1 score') +
  #  geom_smooth(mapping = aes(Total, Value), method = 'gam') +
#  facet_wrap(~ Tag, ncol = 6) +
  theme_classic() + 
  theme(legend.position="none")  + 
  theme(text = element_text(size=15, family="Times")) +
  ylim(0, 1) 

#### Plot of KL divergence for Russian

for (family in as.vector(unique(pos_1000$Language_Family))){
  data=subset(pos_1000,Language_Family==family) 
  print(c(family, unique(data$Language)))}

#russian_pos_1000_kl <- subset(full_pos_1000_individual_tag, Total <= 100000 & Language == 'Russian' & Tag == 'kl')
russian_pos_1000_kl <- subset(full_pos_1000_individual_tag, Total <= 100000 & Tag_prob == 'Russian' & Tag == 'kl')
russian_pos_1000_kl$Treebank[russian_pos_1000_kl$Treebank == 'UD_Russian-GSD'] = 'GSD'
russian_pos_1000_kl$Treebank[russian_pos_1000_kl$Treebank == 'UD_Russian-Poetry'] = 'Poetry'
russian_pos_1000_kl$Treebank[russian_pos_1000_kl$Treebank == 'UD_Russian-SynTagRus'] = 'SynTagRus'
russian_pos_1000_kl$Treebank[russian_pos_1000_kl$Treebank == 'UD_Russian-Taiga'] = 'Taiga'

russian_pos_1000_kl %>%
  ggplot(aes(Total, Value, group = Treebank, color = Treebank, linetype = Treebank)) + 
  geom_line() +
  labs(x = "Training set size") +
  labs(y = 'KL divergence') +
  theme_classic() + 
  theme(legend.position="none")  + 
  theme(text = element_text(size=16, family="Times")) +
  facet_wrap(~ Treebank)

Russian_GSD_kl <- subset(russian_pos_1000_kl, Treebank=='UD_Russian-GSD' & Value < 1)
Russian_GSD_f1 <- subset(russian_pos_1000, Treebank == 'UD_Russian-GSD'&Total <= 100000)
Russian_GSD_f1 <- subset(Russian_GSD_f1, Total %in% as.vector(Russian_GSD_kl$Total))
Russian_GSD_kl$f1 <- Russian_GSD_f1$Value
summary(lm(f1 ~ Value, data = Russian_GSD_kl)) #-2.57
cor(Russian_GSD_kl$Total, Russian_GSD_kl$Value, method = "spearman")

Russian_Poetry_kl <- subset(russian_pos_1000_kl, Treebank=='UD_Russian-Poetry' & Value < 1)
Russian_Poetry_f1 <- subset(russian_pos_1000, Treebank == 'UD_Russian-Poetry'&Total <= 100000)
Russian_Poetry_f1 <- subset(Russian_Poetry_f1, Total %in% as.vector(Russian_Poetry_kl$Total))
Russian_Poetry_kl$f1 <- Russian_Poetry_f1$Value
summary(lm(f1 ~ Value, data = Russian_Poetry_kl)) #-3.76

Russian_SynTagRus_kl <- subset(russian_pos_1000_kl, Treebank=='UD_Russian-SynTagRus' & Value < 1)
Russian_SynTagRus_f1 <- subset(russian_pos_1000, Treebank == 'UD_Russian-SynTagRus'&Total <= 100000)
Russian_SynTagRus_f1 <- subset(Russian_SynTagRus_f1, Total %in% as.vector(Russian_SynTagRus_kl$Total))
Russian_SynTagRus_kl$f1 <- Russian_SynTagRus_f1$Value
summary(lm(f1 ~ Value, data = Russian_SynTagRus_kl)) #-2.16

Russian_Taiga_kl <- subset(russian_pos_1000_kl, Treebank=='UD_Russian-Taiga' & Value < 1)
Russian_Taiga_f1 <- subset(russian_pos_1000, Treebank == 'UD_Russian-Taiga'&Total <= 100000)
Russian_Taiga_f1 <- subset(Russian_Taiga_f1, Total %in% as.vector(Russian_Taiga_kl$Total))
Russian_Taiga_kl$f1 <- Russian_Taiga_f1$Value
summary(lm(f1 ~ Value, data = Russian_Taiga_kl)) #-2.90

### Analyzing relationship between KL divergence and training set sizes

kl_trainsize_corr = cor(russian_pos_1000_kl$Total, russian_pos_1000_kl$Value, method = "spearman")

#### Analyzing relationship between tag probability / word entropy and training set sizes for each tag
for (tag in as.vector(unique(russian_pos_1000_individual_tag$Tag))){
  print(tag)
  tag_subset <- subset(russian_pos_1000_individual_tag, Tag == tag & as.numeric(Tag_prob) > 0)
#  print(tag_subset)
  tagprob_trainsize_corr = cor(tag_subset$Total, as.numeric(tag_subset$Tag_prob), method = 'spearman')
  tag_subset <- subset(tag_subset, Word_entropy != 'None')
  wordentropy_trainsize_corr = cor(tag_subset$Total, as.numeric(tag_subset$Word_entropy), method = 'spearman')
  print(c(tagprob_trainsize_corr, wordentropy_trainsize_corr))
}



### Some specific analysis about Russian-SynTagRus; tag bigram entropy
russian_syntax_entropy <- read.csv('../results/russian_syntax_entropy.txt', header = T, sep = ' ', row.names = NULL)
russian_syntax_entropy <- subset(russian_syntax_entropy, Total <= 100000)

russian_regression_data <- russian_pos_1000_individual_tag
russian_regression_data$Tag_syntax_entropy <- russian_syntax_entropy$Value
russian_regression_data <- subset(russian_regression_data, Word_entropy != 'None')
russian_regression_data$Tag_prob <- as.numeric(russian_regression_data$Tag_prob)
russian_regression_data$Word_entropy <- as.numeric(russian_regression_data$Word_entropy)
russian_regression_data$Tag_syntax_entropy <- as.numeric(russian_regression_data$Tag_syntax_entropy)

lm <- lmer(Value ~ Tag_prob + Word_entropy + Tag_syntax_entropy  + (1| Tag), data = russian_regression_data)
summary(lm)

### Fitting regression to every treebank
entropy_file <- read.csv('../results/50_pos_entropy.txt', header = T, sep = ' ', row.names = NULL)

treebank <- 'UD_Russian-Poetry'

pos_1000_individual_tag <- subset(pos_1000_individual_tag, !(Language %in% c('Russian', 'Latin')))

for (treebank in as.vector(unique(pos_1000_individual_tag$Treebank))){
  print(treebank)
  regression_data <- subset(pos_1000_individual_tag, Total <= 100000 & Treebank == treebank & Tag != 'kl')
  regression_data$Tag_syntax_entropy <- subset(entropy_file, Total <= 100000 & Treebank == treebank & Metric == 'f1')$Tag_syntax_entropy
  regression_data <- subset(regression_data, Word_entropy != 'None')
  regression_data$Tag_prob <- as.numeric(regression_data$Tag_prob)
  regression_data$Word_entropy <- as.numeric(regression_data$Word_entropy)
  regression_data$Tag_syntax_entropy <- as.numeric(regression_data$Tag_syntax_entropy)
  
  lm <- lmer(Value ~ Tag_prob + Word_entropy + Tag_syntax_entropy  + (1| Tag), data = regression_data)
  print(summary(lm))
  print(confint(lm))
  print('')
  print('')
}


### Fitting regression to individual tag

for (tag in as.vector(unique(russian_pos_1000_individual_tag$Tag))){
  print(tag)
  tag_subset <- subset(russian_pos_1000_individual_tag, Tag == tag & as.numeric(Tag_prob) > 0 &  Word_entropy != 'None')
  tag_subset$Tag_prob <- as.numeric(tag_subset$Tag_prob)
  tag_subset$Word_entropy <- as.numeric(tag_subset$Word_entropy)
  tag_subset$Tag_syntax_entropy <- subset(russian_syntax_entropy, Tag == tag & Select_size %in% as.vector(tag_subset$Select_size))$Value
  #  print(tag_subset)
  tag_lm = lm(Value ~ Tag_prob + Word_entropy + Tag_syntax_entropy, data = tag_subset)
  print(summary(tag_lm))
  print('')
  print('')
}


### Snapshot of sample treebanks

select_treebanks = c('UD_English-EWT', 'UD_Hebrew-HTB', 'UD_Basque-BDT', 'UD_Estonian-EWT', 'UD_Wolof-WTB', 'UD_Turkish-BOUN', 
                     'UD_Naija-NSC', 'UD_Korean-Kaist', 'UD_Indonesian-GSD', 'UD_Telugu-MTG', 'UD_Japanese-GSDLUW', 'UD_Chinese-GSDSimp')

vip_pos_1000 <- subset(pos_1000_average, Treebank %in% select_treebanks)

vip_random_pos_1000 <- subset(random_pos_1000, Treebank %in% select_treebanks & Total <= 100000 & Metric=='f1')
vip_random_pos_1000$Method <- rep('Random', nrow(vip_random_pos_1000))

vip_al_pos_1000 <- subset(pos_1000, Treebank %in% select_treebanks & Total <= 100000)
vip_al_pos_1000$Method <- rep('AL', nrow(vip_al_pos_1000))

together <- rbind(vip_random_pos_1000, vip_al_pos_1000)

together %>%
  ggplot(aes(Total, Value, group = Method, color = Method, linetype = Method)) + 
  geom_point(aes(color = Method), alpha=.01) +
  geom_line() +
  labs(x = "Training set size") +
  labs(y = 'F1 score') +
  #  geom_smooth(mapping = aes(Total, Value), method = 'gam') +
  facet_wrap(~ Treebank, ncol = 6) +
  theme_classic() + 
  theme(legend.position="top")  + 
  theme(text = element_text(size=12.8, family="Times")) +
  ylim(0, 1)  +
  theme( #axis.title.x=element_blank(),
    axis.text.x=element_blank(),
    axis.ticks.x=element_blank())


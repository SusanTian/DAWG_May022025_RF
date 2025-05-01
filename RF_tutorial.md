# Random Forest Classifier in Sequencing data
Unlock the power of your sequencing data with Random Forest — a fast, powerful, and intuitive machine learning tool that thrives on complexity. In this workshop, we’ll explore how this algorithm turns raw reads into real biological insight.

## What is machine learning?
- Machine learning is a process that enables a **computer** to **LEARN** from the data
- regression and classification are types of **machine learning tasks**
- in **regression**: we are trying to predict a number (predict the price of eggs)
- in **classification**: we are trying to predict a class (distinguish cats and dogs)
![Regression vs Classification](https://www.sharpsightlabs.com/wp-content/uploads/2021/04/regression-vs-classification_simple-comparison-image_v3.png)


## Then what’s random forest?
- merges the outputs of multiple decision trees into one single result
- RF can be used for both classification and regression
- strength of RF: uses **ensemble** learning—> combines the individual trees that are uncorrelated to enhance predictive accuracy
![Random Forest Diagram](https://www.spotfire.com/content/dam/spotfire/images/graphics/inforgraphics/random-forest-diagram.svg)_A Random Forest uses multiple decision trees to make more accurate and robust predictions by combining their outputs._


## Now let’s add a little more details to RF
steps: 
- split sample into training and testing set (bagging and out-of-bag OOB sample)
- draw samples from the training set with replacement (bootstrap)—> **random**
- Build decision trees (**Forest**) and OOB serves as cross-validation
-regression: averaging the trees’ predictions
-Classification: a majority vote determines the class
![Random Forest Diagram](https://infoaryan.com/wp-content/uploads/2023/11/random-forest-diagram.jpg)

_This diagram illustrates how a Random Forest combines multiple decision trees to improve prediction accuracy and prevent overfitting._

## Pros and Cons about RF
### Pros:
![Random Forest Visualization](https://i.imgur.com/RRVuwSz.png)

_A visual representation of how Random Forest aggregates multiple decision trees for classification and regression tasks._
- reduced risk of overfitting by aggregating uncorrelated trees to lower the overall variance and prediction error
![High School Musical - We're All In This Together](https://media.tenor.com/hLyHxnDeP4gAAAAM/together-high-school-musical.gif)

_“We’re all in this together!” — A little Random Forest team spirit._


- flexible enough to do both regression and classification
- bagging also enables RF to estimate values accurately, maintaining performance even with incomplete data. Random forest considers only a subset of predictors at a split. This results in trees with different predictors at top split, thereby resulting in decorrelated trees and more reliable average output. That's why we say random forest is robust to correlated predictors.
- **provide feature importance (GINI importance)** which will be a important feature in our tutorial, and is essential for the tree to determine where to split. For a split to take place, the Gini index for a child node should be less than that for the parent node.

### Cons
cons: time-consuming and computationally intensive



## more towards the algorithm

Random forest has three primary hyperparameters: node size, number of trees, and the number of features sampled (bootstrap)
- **node size**: the number of samples required to split an internal node. Higher values make the model more conservative (not always a good thing and needs to be tested)
- **number of trees**: is the number of trees in the forest. More trees usually improve performance (up to a point) but increase computation time.
- **bootstrap**: Whether to use bootstrap samples (bagging the samples) when building trees. Remembers this is where randomness will be generated. True (default) gives classic Random Forest behavior. False builds trees on the whole dataset (like bagging without resampling).
![Decision Tree Node Splitting](https://miro.medium.com/v2/resize:fit:1200/format:webp/0*dRrspv4FJVH2SnOO.png)

_This diagram illustrates how decision trees split nodes based on feature thresholds and stopping_

## Let's get into a real example with RF classification with sequencing data
the goal is to identify the features are most important in predicting _C. difficile_ colonized microbiome. You can access the reference paper here: [Susan Tian - DAWG RandomForest - Meta-analysis]
![Figure 1: Meta-analysis](https://github.com/SusanTian/DAWG_May022025_RF/blob/main/Screenshot%202025-05-01%20at%2012.21.22.png?raw=true)
As Panel C-H showed, the microbiome diversity has gone down in _C. difficile_ colonized microbiomes and the microbiota composition shifted from healthy counterparts. Will the missing microbes essential to providing colonization resistance to _C. difficile_? Can we predict _C. difficile_ infection (CDI) status using these microbes? Let's see if we can use random forest to predict _C. difficile_ colonization status. 

### Loading packages
```{R}
# Suppress messages for cleaner output
{

  # List of CRAN packages
  cran_packages <- c("tidyverse", "readxl", "randomForest", "ROCR")

  # Install missing CRAN packages
  installed <- rownames(installed.packages())
  for (pkg in cran_packages) {
    if (!pkg %in% installed) {
      install.packages(pkg, dependencies = TRUE)
    }
  }

  # Install qiime2R from GitHub if not already installed
  if (!"qiime2R" %in% installed) {
    if (!require("devtools")) install.packages("devtools")
    devtools::install_github("jbisanz/qiime2R", quiet = TRUE)
  }

  # Load all packages
  all_packages <- c(cran_packages, "qiime2R")
  invisible(lapply(all_packages, function(pkg) library(pkg, character.only = TRUE, quietly = TRUE)))
  
  message("All packages are installed and loaded.")
}
```

### Next load the datasets and metadata
to make the computation faster, I filtered out the sequences from three studies out of 12 studies (59 out of 899 samples). What do you think this is going to affect the result than running the full meta-analysis dataset? 
```
metadata <- read_csv("https://raw.githubusercontent.com/SusanTian/DAWG_May022025_RF/main/metadata.csv")
metadata %>% interactive_table()

otu_table <- readRDS(url("https://raw.githubusercontent.com/SusanTian/DAWG_May022025_RF/main/OTU_table.RDS"))
otu_table_subsampled <- otu_table %>% subsample_table()

otu_taxonomy <- read_csv("https://raw.githubusercontent.com/SusanTian/DAWG_May022025_RF/main/OTU_taxonomy.csv")
```
Different samples often have different numbers of sequencing reads due to technical variation. Subsampling brings all samples to the same read count, so comparisons aren’t biased by depth. Subsampling also ensures that diversity differences reflect biology, not sequencing effort (alpha and beta diversity are sensitive to sequencing depth. Many statistical models assume equal observation effort across groups—subsampling helps meet that assumption.

### Building the model
Each time, we will exclude one study to do external validation (this is a new dataset that the model has never seen) to increase robustness. We will also write this into a loop so that we are not manually doing it for each study. 

#### getting ready by defining tables
```
predictions <- list()
importance <- list()
auc <- list()
ROC_Table <- list()
tables <- list()
predictions_ext_study <- list()
auc_ext_study <- list()
ROC_Table_ext_study <- list()
ext_study_list <- metadata$StudyID %>% unique()

tables[["OTU_clr_withCD"]] <- make_clr(otu_table)
tables[["Species_clr_withCD"]] <- make_clr(summarize_taxa(otu_table,otu_taxonomy %>% tibble::column_to_rownames("FeatureID"))$Species)
tables[["Species_proportion_withCD"]] <- make_proportion(summarize_taxa(otu_table,otu_taxonomy %>% 
  tibble::column_to_rownames("FeatureID"))$Species)

# to further increase the robustness of our model, we will test the model performance after C. difficile OTUs so that the model doesn't cheat!

no_cd_taxonomy <- filter(otu_taxonomy,Genus %in% "Peptoclostridium")
otu_table_nocd <- otu_table[!rownames(otu_table) %in% no_cd_taxonomy$FeatureID,]
temp_otutable <- filter_features(otu_table,minsamples = 3, minreads = 3) %>% t()
temp_otutable <- temp_otutable[,!colnames(temp_otutable)%in%no_cd_taxonomy$FeatureID]

tables[["OTU_clr_withoutCD"]] <- make_clr(otu_table_nocd)
tables[["Species_clr_withoutCD"]] <- make_clr(summarize_taxa(otu_table_nocd,otu_taxonomy %>% 
  tibble::column_to_rownames("FeatureID"))$Species)
tables[["Species_proportion_withoutCD"]] <- make_proportion(summarize_taxa(otu_table_nocd,otu_taxonomy %>% 
  tibble::column_to_rownames("FeatureID"))$Species)
```

#### Define Do_RF
```
dir.create("figures")
dir.create("figures/validation")

# define the function
Do_RF <- function(a,training_meta,test_meta,ext_meta, label){
  ROC<-list()

# training the model
ROC$Model<-randomForest(x=a[,training_meta$SampleID] %>% t(), y=training_meta$Cdifficile, importance=TRUE)

# Extracts the MeanDecreaseGini importance score for each feature.
# also add ranking to the importance on descending order
ROC$Importance<-
  ROC$Model$importance %>% 
  as.data.frame() %>% 
  tibble::rownames_to_column("FeatureID") %>% 
  arrange(desc(MeanDecreaseGini)) %>%
  mutate(Rank=1:nrow(.))

# Saves two feature importance plots: one full, one zoomed into the top 500
p <- 
ROC$Importance %>%
  ggplot(aes(x=Rank, y=MeanDecreaseGini)) +
  geom_line() +
  geom_vline(xintercept=500, linetype="dashed", color="grey50")
ggsave(paste0("figures/validation/rf_importance_",label,".pdf"),p, height=3, width=3)

p <- 
ROC$Importance %>%
  filter(Rank<500) %>%
  ggplot(aes(x=Rank, y=MeanDecreaseGini)) +
  geom_line()
ggsave(paste0("figures/validation/rf_importance_top_",label,".pdf"),p, height=3, width=3)

# Exports the importance table to a .tsv file
ROC$Importance %>%
  readr::write_tsv(paste0("figures/validation/rf_importance_table_",label,".tsv"))

# Gets probability predictions (for class = 1) for both test datasets
# Computes ROC curve (TPR vs FPR) 
ROC$Predictions<-predict(ROC$Model, newdata=a[,test_meta$SampleID] %>% t(), type="prob")[,2] %>%
             prediction(., test_meta$Cdifficile) %>%
             performance(., "tpr","fpr")
ROC$Predictions_ext_study<-predict(ROC$Model, newdata=a[,ext_meta$SampleID] %>% t(), type="prob")[,2] %>%
             prediction(., ext_meta$Cdifficile) %>%
             performance(., "tpr","for")

# Calculates Area Under the Curve (AUC) for both test sets
ROC$AUC<-
predict(ROC$Model, newdata=a[,test_meta$SampleID] %>% t(), type="prob")[,2] %>%
             prediction(., test_meta$Cdifficile) %>%
             performance(., "auc") %>%
             .@y.values %>%
             as.numeric()
ROC$AUC_ext_study<-
predict(ROC$Model, newdata=a[,ext_meta$SampleID] %>% t(), type="prob")[,2] %>%
             prediction(., ext_meta$Cdifficile) %>%
             performance(., "auc") %>%
             .@y.values %>%
             as.numeric()

# Converts the ROC object into a tidy tibble for plotting
ROC$ROC_Table<-tibble(FPR=unlist(ROC$Predictions@x.values), TPR=unlist(ROC$Predictions@y.values)) %>% 
  mutate(Data_Type=label)
ROC$ROC_Table_ext_study<-tibble(FPR=unlist(ROC$Predictions_ext_study@x.values), TPR=unlist(ROC$Predictions_ext_study@y.values)) %>% 
  mutate(Data_Type=label)

# Generates and saves ROC plots for internal and external validations
p <- 
ROC$ROC_Table %>%
  ggplot(aes(x=FPR, y=TPR)) +
  geom_line() +
  theme_q2r() +
  geom_abline(linetype="dashed", color="grey50") +
  ylab("True Positive Rate") +
  xlab("False Positive Rate") +
  ggtitle(paste0(label, " AUC=", ROC$AUC))
ggsave(paste0("figures/validation/ROC_plot_",label,".pdf"),p, height=3, width=3)

p <- 
ROC$ROC_Table_ext_study %>%
  ggplot(aes(x=FPR, y=TPR)) +
  geom_line() +
  theme_q2r() +
  geom_abline(linetype="dashed", color="grey50") +
  ylab("True Positive Rate") +
  xlab("False Positive Rate") +
  ggtitle(paste0(label, " AUC=", ROC$AUC_ext_study))
ggsave(paste0("figures/validation/ROC_plot__ext_study_",label,".pdf"),p, height=3, width=3)

# Returns the ROC list containing model, importance, predictions, AUC, and plots.
return(ROC)
}
```

#### Running Random Forest 
Setting a seed in random forest ensures reproducibility by making the random processes (like bootstrapping and feature selection) generate the same results each time the code is run. If you want every run to be different, a tip would be using today's date as a seed. Whereas you want to have reproducible runs, set the seed and reuse it. 
```
for (i in (1:3)){
  set.seed(i)
  message(i)
  ext_study <- sample(ext_study_list,1)
  ext_study_list <- ext_study_list[!ext_study_list %in% ext_study]
  message(ext_study)
  message(ext_study_list)
  rf_metadata <- metadata %>% mutate(Cdifficile=ifelse(Cdifficile,"Positive","Negative") %>% as.factor())
  ext_meta <- rf_metadata %>% filter(StudyID==ext_study)
  rf_metadata <- rf_metadata %>% filter(StudyID!=ext_study)
  training_meta <- rf_metadata %>% sample_n(round(2/3*nrow(rf_metadata))) # training the model with 2/3 of the samples
  test_meta <- rf_metadata %>% filter(!SampleID %in% training_meta$SampleID) # test the model with the rest

  
  for (a in names(tables)){
  message(a)
    result <- Do_RF(tables[[a]],training_meta,test_meta, ext_meta, paste0(a,i))
    importance[[paste(a,i, sep = "_")]] <- result$Importance
    predictions[[paste(a,i, sep = "_")]] <- result$Predictions
    auc[[paste(a,i, sep = "_")]] <- result$AUC
    ROC_Table[[paste(a,i, sep = "_")]] <- result$ROC_Table 
    
    message(ext_study)
    predictions_ext_study[[paste(a,i, sep = "_")]] <- result$Predictions_ext_study
    auc_ext_study[[paste(a,i, sep = "_")]] <- result$AUC_ext_study
    ROC_Table_ext_study[[paste(a,i, sep = "_")]] <- result$ROC_Table_ext_study %>% mutate(StudyID=ext_study)
  }
}
```

#### Now let's examine the model! 
```
auc_plot <- auc %>% unlist() %>% data.frame(AUC=.) %>% 
  tibble::rownames_to_column("Name") %>% 
  separate(.,Name,into = c("Feature_Type","Normalization","Cdiff Inclusion","Iteration"),sep="_",remove=FALSE)

auc_plot_ext <- auc_ext_study %>% unlist() %>% data.frame(AUC=.) %>% 
  tibble::rownames_to_column("Name") %>% 
  separate(.,Name,into = c("Feature_Type","Normalization","Cdiff Inclusion","Iteration"),sep="_",remove=FALSE)

auc_plot_ext <- 
auc_plot_ext %>% 
  mutate(StudyID=case_when(Iteration==1~"Nygren_2023",
                   Iteration==2~"Seekatz_2018",
                   Iteration==3~"Weingarden_2015"))
auc_plot_ext <- 
  auc_plot_ext %>% 
  left_join(metadata %>%
  group_by(StudyID) %>%
  summarize(Nsamples=n()) %>%
  arrange(desc(Nsamples)) %>%
  ungroup() %>%
  mutate(StudyID=factor(StudyID, levels=unique(StudyID))))

merged_boxplot <- 
  bind_rows(auc_plot,auc_plot_ext) %>% 
  mutate(External_Study=if_else(!is.na(StudyID),"External Studies","Combined Analysis"))

merged_boxplot %>% 
  mutate(Normalization=if_else(Normalization=="proportion","Proportion","Log Ratio")) %>% 
  ggplot(aes(x=Feature_Type,y=AUC,fill=Normalization))+
  geom_boxplot()+
  theme_q2r() +
  xlab("Feature Type") +
  ylab("AUC")+
  facet_grid(External_Study~`Cdiff Inclusion`)

rm(auc_plot)
rm(auc_plot_ext)
rm(merged_boxplot)
```
looks like species proportion gives the highest AUC. 

#### Mean Decrease GINI figure
```{r}
importance[grep("Species_proportion_withoutCD",names(importance))] %>% do.call(bind_rows,.) %>%
  group_by(FeatureID) %>%
  summarize(Mean=mean(MeanDecreaseGini), sd=sd(MeanDecreaseGini)) %>%
  arrange(desc(Mean)) %>%
  mutate(Rank=1:nrow(.)) %>%
  ggplot(aes(x=log10(Rank), y=Mean, ymin=Mean-sd, ymax=Mean+sd)) +
  geom_ribbon(fill="#BDBDBD", alpha=0.8) +
  geom_line(linetype="dashed", color="grey40") +
  geom_vline(xintercept = log10(500), linetype=2,color="grey40")+
  ylab("Mean Decrease Gini")+
  #coord_cartesian(ylim=c(0,2.8))+
  theme_q2r()
```


#### AUROC curves
```
bind_rows(
  ROC_Table[grep("Species_proportion_withCD",names(ROC_Table))] %>% 
    do.call(bind_rows,.) %>% 
    mutate(Group="Cd_included"),
  ROC_Table[grep("Species_proportion_withoutCD",names(ROC_Table))] %>% 
    do.call(bind_rows,.) %>% 
    mutate(Group="Cd_excluded")) %>% 
  mutate(Data_Type=gsub("\\d","",Data_Type)) %>% 
  separate(Data_Type, c("Type","Normalization","CD"), sep="_") %>%
  filter(Type=="Species" & Normalization=="proportion") %>%
  mutate(FPRr=if_else(FPR>0 & TPR>0, plyr::round_any(FPR, 0.1, ceiling) ,0)) %>%#round to nearest 0.05
  mutate(TPRr=if_else(TPR>0 & TPR>0, plyr::round_any(TPR, 0.1, ceiling) ,0)) %>%#round to nearest 0.05
  mutate(rFPR=round(FPR/0.05)*0.05) %>%
  ggplot(aes(x=rFPR, y=TPRr, fill=CD, group=CD)) +
  stat_summary(geom="ribbon", alpha=0.5, fun.data=mean_se) +
  stat_summary(geom="line", aes(color=CD), linetype="dashed") +
  #geom_point(shape=21) +
  #geom_line() +
  geom_abline(linetype="dashed", color="grey50") +
  #geom_smooth(aes(color=CD)) +
  coord_cartesian(ylim=c(0,1), xlim=c(0,1)) +
  theme_q2r() +
  scale_fill_manual(values=c("indianred","cornflowerblue")) +
  scale_color_manual(values=c("indianred","cornflowerblue")) +
  theme(legend.position="none")
```

#### How does the result of these 3 dataset compared to the full dataset? 
![Figure 1: Meta-analysis](https://github.com/SusanTian/DAWG_May022025_RF/blob/main/Screenshot%202025-05-01%20at%2012.21.22.png?raw=true)
![ROC Curve](https://raw.githubusercontent.com/SusanTian/DAWG_May022025_RF/main/Screenshot%202025-05-01%20at%2019.40.48.png)

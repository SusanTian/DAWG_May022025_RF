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
```
# Suppress messages for cleaner output
suppressMessages({

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
})
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

tables[["OTU_proportion_withCD"]] <- make_proportion(otu_table)
tables[["OTU_clr_withCD"]] <- make_clr(otu_table)
tables[["Species_clr_withCD"]] <- make_clr(summarize_taxa(otu_table,otu_taxonomy %>% tibble::column_to_rownames("FeatureID"))$Species)
tables[["Species_proportion_withCD"]] <- make_proportion(summarize_taxa(otu_table,otu_taxonomy %>% 
  tibble::column_to_rownames("FeatureID"))$Species)

# to further increase the robustness of our model, we will test the model performance after _C. difficile_ OTUs so that the model doesn't cheat!

no_cd_taxonomy <- filter(otu_taxonomy,Genus %in% "Peptoclostridium")
otu_table_nocd <- otu_table[!rownames(otu_table) %in% no_cd_taxonomy$FeatureID,]
temp_otutable <- filter_features(otu_table,minsamples = 3, minreads = 3) %>% t()
temp_otutable <- temp_otutable[,!colnames(temp_otutable)%in%no_cd_taxonomy$FeatureID]

```

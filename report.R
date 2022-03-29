# These are all the non-base packages used. May have to install them one-by-one if in RStudio.
# ggplot2 is used for all plots except the correlation plot. 
# mlr3verse (and friends) are used for the machine learning.
install.packages("skimr")
install.packages("readr")
install.packages("tidyverse")
install.packages("ggplot2")
install.packages("ggcorrplot")
install.packages("data.table")
install.packages("mlr3verse")
install.packages("mlr3learners")
install.packages("mlr3proba")
install.packages("mlr3tuning")
install.packages("reshape2")
library("tidyverse")
library("ggplot2")
library("dplyr")
library("ggcorrplot")
library("data.table")
library("mlr3verse")
library("mlr3learners")
library("mlr3proba")
library("mlr3tuning")
library("reshape2")

heart <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv")
# Check for missing values:
skimr::skim(heart)$n_missing
# Check for any obvious errors in the data (e.g. age = 500) by finding mins and maxs:
apply(heart,2,min)
apply(heart,2,max)

plot_data = heart %>% as_tibble()
plot_data <- plot_data %>% mutate("fatal_mi" = ifelse(fatal_mi == 1, "Yes", "No"))
plot_data <- plot_data %>% mutate("sex" = ifelse(sex == 1, "Male", "Female"))
plot_data <- plot_data %>% mutate("high_blood_pressure" = ifelse(high_blood_pressure == 1, "Yes", "No"))

ggplot(plot_data, aes(x=time, fill=fatal_mi)) +
  geom_histogram(colour="black",alpha=0.5, position="identity", bins=10) +
  guides(fill=guide_legend(title="Fatal MI")) +
  theme_bw() + ## theme_bw() MUST COME BEFORE ALL theme() CALLS!! Or else future theme calls in next plots won't work.
  theme(legend.position=c(0.94,0.9),
        legend.background = element_rect(fill="white",color="black"),
        legend.key.size = unit(6, "mm"),
        legend.title=element_text(size=14),
        legend.text=element_text(size=14),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15),  
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16)) +
  labs(x="Follow-up time period (days)", y="Count")

ggplot(plot_data, aes(x=high_blood_pressure, fill=fatal_mi)) +
  geom_bar(colour="black",alpha=0.5, position="dodge") +
  guides(fill=guide_legend(title="Fatal MI")) +
  theme_bw() +
  theme(legend.position=c(0.94,0.9),
        legend.background = element_rect(fill="white",color="black"),
        legend.key.size = unit(6, "mm"),
        legend.title=element_text(size=14),
        legend.text=element_text(size=14),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15),  
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16)) +
  labs(x="High Blood Pressure (Hypertension)", y="Count")

corr_matrix = round(cor(heart), 1)
ggcorrplot(corr_matrix, lab=TRUE,lab_size=4, type="lower",tl.cex = 14,tl.srt=45)

# In MLR3, the target must be a factor, not a numeric
heart$fatal_mi = factor(heart$fatal_mi)

set.seed(1337)

heart_task <- TaskClassif$new(id = "heart",
                              backend = heart,
                              target = "fatal_mi")

# Step 2: create resampling strategy (error estimation)
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(heart_task)

# Step 3: Create learners
# First try with default parameters:
lrn_logreg = lrn("classif.log_reg",predict_type="prob")
lrn_forest = lrn("classif.ranger", importance='impurity',predict_type="prob")
lrn_xgbooster = lrn("classif.xgboost",predict_type="prob")

set.seed(1337)
# Step 4: fit the models
res <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_logreg,
                    lrn_forest,
                    lrn_xgbooster),
  resampling = list(cv5)
), store_models = TRUE)

# Recall is important for us
results = res$aggregate(list(msr("classif.acc"),
                             msr("classif.auc"),
                             msr("classif.precision"),
                             msr("classif.recall")))
results

# Tuning:
lrn_xgbooster = lrn("classif.xgboost",nrounds=to_tune(10,100), eta=to_tune(0.05,0.5),
                    lambda=to_tune(1,10))
set.seed(1337)
## TAKES FEW MINS + SOME MEMORY; UNCOMMENT TO RUN TUNING:

# instance = tune(
#   method = "grid_search",
#   task = heart_task,
#   learner = lrn_xgbooster,
#   resampling = rsmp("cv", folds = 5),
#   measure = msr("classif.recall"),
#   allow_hotstart = FALSE,
#   resolution = 10, # resolution is how the range in to_tune(a,b) is split up; 10 means we have 10 values of the parameter, starting from a and finishing at b.
#   batch_size = 10 # batch_size is number of calculations in each batch
# )

# best performing hyperparameter configuration
# instance$result

# See results with tuned parameters:
lrn_xgbooster = lrn("classif.xgboost",predict_type="prob", eta=0.5, lambda=6,nrounds=100)
set.seed(1337)
res <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_logreg,
                    lrn_forest,
                    lrn_xgbooster),
  resampling = list(cv5)
), store_models = TRUE)

results = res$aggregate(list(msr("classif.acc"),
                             msr("classif.auc"),
                             msr("classif.precision"),
                             msr("classif.recall")))
results

# Feature importance:
forest_filter = flt("importance", learner=lrn_forest)
forest_filter$calculate(heart_task)
ggplot(as.data.table(forest_filter), aes(x=reorder(feature,score), y=score))+
  geom_bar(stat="identity", position="dodge",fill="#87CEEB")+ coord_flip() +
  labs(x="", y="Mean Decrease in Gini Impurity")+
  theme_bw() +
  theme(axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15),  
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16))

# Reduced feature set:
heart_reduced <- heart %>% 
  select(-diabetes, -sex, -anaemia, -high_blood_pressure, -smoking)

heart_task_red <- TaskClassif$new(id = "heart_red",
                                  backend = heart_reduced,
                                  target = "fatal_mi")
set.seed(1337)
cv5red <- rsmp("cv", folds = 5)
cv5red$instantiate(heart_task_red)

lrn_logreg_red = lrn("classif.log_reg",predict_type="prob")
lrn_forest_red = lrn("classif.ranger", importance='impurity',predict_type="prob")
lrn_xgbooster_red = lrn("classif.xgboost",predict_type="prob",eta=0.5,nrounds=100,lambda=3)
set.seed(1337)
res <- benchmark(data.table(
  task       = list(heart_task_red),
  learner    = list(lrn_logreg_red,
                    lrn_forest_red,
                    lrn_xgbooster_red),
  resampling = list(cv5red)
), store_models = TRUE)

results_red = res$aggregate(list(
  msr("classif.acc"),
  msr("classif.auc"),
  msr("classif.precision"),
  msr("classif.recall")))
results_red

# Tuning for reduced features 
lrn_xgbooster_red = lrn("classif.xgboost",nrounds=100,eta=to_tune(0.1,0.5),lambda=to_tune(1,5))
set.seed(1337)

## This tuning is quicker:
# reduced_tune = tune(
#   method = "grid_search",
#   task = heart_task_red,
#   learner = lrn_xgbooster_red,
#   resampling = rsmp("cv", folds = 5),
#   measure = msr("classif.recall"),
#   allow_hotstart = FALSE,
#   resolution = 5, # resolution is how the range in to_tune(a,b) is split up; 10 means we have 10 values of the parameter, starting from a and finishing at b.
#   batch_size = 5 # batch_size is number of calculations in each batch
# ) 
# reduced_tune$result

# Performance measures:
measure_vals = as.data.frame(results[,7:10])
colnames(measure_vals) = c("Accuracy","AUC","Precision","Recall")
measure_vals=cbind(Learner=c("Logistic Regression","Random Forest", "XGBoost"),measure_vals)
measure_vals<-melt(measure_vals)
ggplot(measure_vals, aes(x=reorder(variable,value),y=value, fill=Learner))+
  geom_bar(stat="identity", position="dodge")+ 
  labs(x="Measure", y="Value")+
  theme_bw() +
  theme(legend.position=c(0.28,0.96),
        legend.direction="horizontal",
        legend.background = element_rect(fill="white",color="black"),
        legend.title=element_blank(),
        legend.key.size = unit(4, "mm"),
        legend.text=element_text(size=12),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15),  
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16))

# Train final model on entire dataset before deployment:
lrn_forest$train(heart_task)
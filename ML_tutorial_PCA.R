#This code is created to use PCAs to predict the Sex of participants from the HCP dataset
#Produced by Mohan Gupta (https://mohanwugupta.com)
#Github: mohanwugupta
#If you find this tutorial and code useful, please share it with your students and colleagues!

HCP = read.csv(file="/Users/CCNLAB/Box Sync/Education_Youtube/Machine Learning/Code/HCP_morph_tutorial_dat.csv", header = TRUE)

library(e1071)
library(caret)
library(dplyr)
library(gridExtra)
library(ggbiplot)
#To install: 
#library(devtools)
#install_github("vqv/ggbiplot")

####################################################################################################
####################################################################################################
HCP$SUBJ = NULL
HCP$Age = NULL
anyNA(HCP)
HCP = na.omit(HCP)
#Preprocess Data - must do seperately for some reason...
preProcess_missingdata_model <- preProcess(HCP, method='center')
preProcess_missingdata_model
HCP <- predict(preProcess_missingdata_model, newdata = HCP)

preProcess_missingdata_model <- preProcess(HCP, method='scale')
preProcess_missingdata_model
HCP <- predict(preProcess_missingdata_model, newdata = HCP)

####################################################################################################
####################################################################################################
#PCA
HCP_pca = prcomp(HCP[,2:length(HCP)], center = F, scale. = F)
#Store for plotting later
HCP_Group = factor(HCP$Sex)
#plot PCAs
ggbiplot(HCP_pca, lables = rownames(HCP))

#Determine # of PCs
PVE = HCP_pca$sdev^2/sum(HCP_pca$sdev^2)
round(PVE,2)
# PVE (aka scree) plot
PVEplot <- qplot(c(1:138), PVE) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab("PVE") +
  ggtitle("Scree Plot") +
  ylim(0, 1)
# Cumulative PVE plot
cumPVE <- qplot(c(1:138), cumsum(PVE)) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab(NULL) + 
  ggtitle("Cumulative Scree Plot") +
  ylim(0,1)

grid.arrange(PVEplot, cumPVE, ncol = 2)

#Extract PC Scores
HCP2 = cbind(HCP, HCP_pca$x[,1:10]) #Need to determine how many PCs to keep

#Plot 
ggbiplot(HCP_pca, obs.scale = 1, var.scale = 1,
         groups = HCP_Group, ellipse = TRUE, circle = FALSE) +
  scale_color_discrete(name = '') +
  theme(legend.direction = 'horizontal', legend.position = 'top')

#Correlations between vars and PCs
cor(HCP[,-1], HCP2[,140:149], method = "pearson")
View(HCP_pca$rotation)
####################################################################################################
####################################################################################################
#CrossValidation
fitControl <- trainControl(
  method = 'repeatedcv',            # k-fold cross validation
  number = 10,                      # number of folds
  savePredictions = 'final',        # saves predictions for optimal tuning parameter
  classProbs = T,                   # should class probabilities be returned
  summaryFunction=twoClassSummary  # results summary function
)

set.seed(100)
#Ensure factors
HCP2$Sex = factor(HCP2$Sex)

#Train model and cv tuning parameters
model_svm2 = train(Sex ~ ., data=HCP2[,c(1,2:139)], 
                   method='svmLinear',
                   metric='ROC', 
                   verbose="FALSE", 
                   trControl = fitControl)
model_svm2

varimp_svm <- varImp(model_svm2)
plot(varimp_svm, main="Variable Importance with SVM")

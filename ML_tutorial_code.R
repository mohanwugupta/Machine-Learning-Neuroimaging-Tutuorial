#This code is for the ML for neuroimagers tutorial
#Produced by Mohan Gupta (https://mohanwugupta.com)
#Github: mohanwugupta

####################################################################################################
####################################################################################################
#SETTING UP R ENVIRONMENT

#Install packages - uncomment if this is your first time
# install.packages("e1071")
# install.packages("caret")
# install.packages("dplyr")
# install.packages("randomForest")
# install.packages("caretEnsemble")
# install.packages("skimr")
# install.packages("RANN")
# install.packages("LiblineaR")
# install.packages("corrplot")
# install.packages("glmnet")

#Load libraries 
library(e1071) # SVM models and more
library(caret) #Implementation of ML methods
library(dplyr) #data formatting
library(randomForest) #RF ML models
library(caretEnsemble) #Allows for stacking ML models
library(skimr) #feature visualization
library(RANN) #ML models
library(LiblineaR) #ML models
library(corrplot) #feature visualization 
library(glmnet) #Other ML models and crossover features of caret

####################################################################################################
####################################################################################################
#DEMOGRAPHICS AND FILTERING 

#Load in data - CHANGE THIS PATH TO WHERE YOU DOWNLOADED OR MOVED THE CSV FILE FROM MY GITHUB
HCP = read.csv(file="/Users/CCNLAB/Box Sync/Education_Youtube/Machine Learning/Code/HCP_morph_tutorial_dat.csv", header = TRUE)

#Get Demographics
#Make Female dataframe
F_HCP = HCP%>%
  filter(!Sex == "Male")
F_Age_avg = mean(F_HCP$Age)

#Make Male dataframe
M_HCP = HCP%>%
  filter(!Sex == "Female")
M_Age_avg = mean(M_HCP$Age)

#t.test age to see if there are sig diffs
t.test(M_HCP$Age, F_HCP$Age, var.equal = T)

#Get rid of unwanted features
HCP$SUBJ <- NULL
HCP$Age <- NULL
####################################################################################################
#Small Note: There cannot be any NAs or empty cells for this method to work
#Fix missing data points - there aren't any missing data points for morphemtry 
anyNA(HCP)
HCP = na.omit(HCP)
anyNA(HCP)
####################################################################################################
####################################################################################################
#Preprocess - scale and demean = standard deviation = 1 and mean = 0
y=HCP$Sex
preProcess_range_model <- preProcess(HCP[,2:length(HCP)], method=c('center'))
HCP = predict(preProcess_range_model, HCP[,2:length(HCP)])

preProcess_range_model <- preProcess(HCP[,1:length(HCP)], method=c('scale'))
HCP = predict(preProcess_range_model, HCP[,1:length(HCP)])
HCP$Sex=y

#Move Sex to the front
HCP = HCP[,c(ncol(HCP),1:(ncol(HCP)-1))]

####################################################################################################
####################################################################################################
#Feature Plotting and Selection
#Plot Features - helps tell us which variables may be important
featurePlot(x = HCP[, 2:18], 
            y = HCP$Sex, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))


#LASSO Feature Selection
#alpha controls how much penalty is applied and lambda controls the overall magnitude of the penalty
set.seed(30)
#CV which lambda to use for lasso
cvfit <- glmnet::cv.glmnet(as.matrix(HCP[,2:length(HCP)]), factor(HCP$Sex), type.measure='mse',
                           nfolds=10,alpha=.1, family = "binomial")

#Lasso feature selection
sel_feats = as.matrix(coef(cvfit, s = "lambda.1se"))

#Get list of non-zero features
features_lasso = which(sel_feats!=0, arr.ind = T)
features_lasso = rownames(features_lasso)
features_lasso = features_lasso[2:length(features_lasso)]

HCP = HCP%>%
  select(Sex, features_lasso)
#Plot MSE
plot(cvfit, ylab = "Mean-Squared Error", xlab = "log(Lambda)") 

#Correlation Matrix of kept features
#Correlation Matrix - Need short names in order to fit them on the plot!
# M = cor(HCP[,2:length(HCP)])
# res1 = cor.mtest(HCP[, 2:length(HCP)], conf.level = .95)
# col = colorRampPalette(c("blue", "red"))
# Feature.matrix <- corrplot(M, p.mat = res1$p, sig.level = c(.001, .01, .05), pch.cex = .5, 
#                        insig = "label_sig", method = "circle", tl.pos = "y", type = "upper",
#                        col = col(10), addrect = 3)

#Plot Selected Features
featurePlot(x = HCP[, 2:18], 
            y = HCP$Sex, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

#Look at mins and maxes of features
apply(HCP[, 2:18], 2, FUN=function(x){c('min'=min(x), 'max'=max(x))})
####################################################################################################
####################################################################################################
#Data splitting 
#IT IS ALWAYS BEST TO HAVE AN INDEPENDENT DATASET!!!!!
#Create Training and testing data - 80% training, 20% testing
#Group based on sex
set.seed(100)
trainIndex <- createDataPartition(as.factor(HCP$Sex), p = .8, list= FALSE)
traindata = HCP[trainIndex,]
testdata = HCP[-trainIndex,]

#Descriptive Stats and visualization 
skimmed <- skim_to_wide(traindata)
skimmed[, c(1:5, 9:11, 13)]

####################################################################################################
####################################################################################################
#Cross Validation of ML parameters
set.seed(102)
fitControl <- trainControl(
  method = 'repeatedcv',            # k-fold cross validation
  number = 10,                      # number of folds
  savePredictions = 'final',        # saves predictions for optimal tuning parameter
  classProbs = T,                   # should class probabilities be returned
  summaryFunction=twoClassSummary  # results summary function
)


#Ensure factors
traindata$Sex = factor(traindata$Sex)
####################################################################################################
####################################################################################################
#ML Models
#Train model and cv tuning parameters
model_svm2 = train(Sex ~ ., data=traindata, 
                   method='svmPoly', 
                   tuneLength = 5, metric='ROC',
                   preProc = c("center","scale"),
                   verbose="FALSE", 
                   trControl = fitControl)
model_svm2

plot(model_svm2, main="Model Accuracies with SVM")

varimp_svm <- varImp(model_svm2)
plot(varimp_svm, main="Variable Importance with SVM")

####################################################################################################
####################################################################################################
#Prediction and Confusion Matrices
predicted <- predict(model_svm2, testdata)

testdata$Sex = factor(testdata$Sex) #Make factors in data
levels(testdata$Sex) <- c("Female", "Male") #Ensure names

u=union(predicted, testdata$Sex)
t = table(factor(predicted, u), factor(testdata$Sex, u))
confusionMatrix(t)

####################################################################################################
####################################################################################################
#Stacking
#TAKES A LONG TIME TO RUN - DO WITH CAUTION
# Stacking Algorithms - Voting method - most votes classify
#setup CV for models
set.seed(56)
trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE,
                             classProbs=TRUE)

#algos to use in stack
algorithmList <- c('rf', 'xgbDART','knn','mlp')


#run algos
models <- caretList(Sex ~ ., data=traindata, 
                    trControl=trainControl, 
                    tuneLength = 3,
                    preProc = c("center", "scale"),
                    methodList=algorithmList)

results <- resamples(models)
summary(results)

set.seed(101)
#setup CV for stack
stackControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             preProc = c("center", "scale"),
                             classProbs=TRUE)

stack.glm <- caretStack(models, method="glm", metric="ROC", trControl=stackControl)
print(stack.glm)

#Compare algorithms
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

splom(results)

# Step 2: Predict on testdata and Compute the confusion matrix
predicted3 <- predict(stack.glm, testdata)
testdata$Sex = factor(testdata$Sex) #Make factors in data

u=union(predicted3, testdata$Sex)
t = table(factor(predicted3, u), factor(testdata$Sex, u))
confusionMatrix(t)

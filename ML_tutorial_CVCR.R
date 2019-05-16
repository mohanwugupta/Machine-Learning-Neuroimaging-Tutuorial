#This code is for it perform "cross-validated confound regression" (CVCR) proposed by Snoek, Miletic, and Scholte (2018) 
#for confound regression in machine learning models
#Produced by Mohan Gupta (https://mohanwugupta.com)
#Github: mohanwugupta
#I graciously thank Dr. Jeanette Mumford for her input on the code and my incessent questions
#And Lukas Snoek for confirming my code and making a correction
#If you find this tutorial and code useful, please share it with your students and colleagues!
####################################################################################################
####################################################################################################
#SETTING UP R ENVIRONMENT

#Load libraries 
library(caret)
library(dplyr)

#Load in data - CHANGE THIS PATH TO WHERE YOU DOWNLOADED OR MOVED THE CSV FILE FROM MY GITHUB
HCP = read.csv(file="/Users/CCNLAB/Box Sync/Education_Youtube/Code/Machine Learning/HCP_morph_tutorial_dat.csv", header = TRUE)
#Remove NA values from dataset
HCP = na.omit(HCP)
#Define dfs for later
whichTwoPct = data_frame()
resul=list()
final=data_frame()
all_conf=data_frame()
current = data_frame()

#Create 10 equally size folds
k = 10 #Number of folds
set.seed(1234)

#Segement your data by fold
fold_inner <- cut(seq(1,nrow(HCP)),breaks=k,labels=FALSE)

#Define confound
conf = as.numeric(unclass(factor(HCP[,2])))
conf = cbind(1, conf)

#Start CV
for (j in 1:k) {
  #divide test subset up into k-fold train and validation
  Indexes <- which(fold_inner==j,arr.ind=TRUE)
  
  X_trainData = HCP[-Indexes, ]
  X_testData = HCP[Indexes, ]
  
  #Define training set
  X_train =as.matrix(X_trainData[,4:length(X_trainData)])
  X_test = as.matrix(X_testData[,4:length(X_testData)])

  # Define confound 
  C_train = conf[-Indexes, ]
  C_test = conf[Indexes, ]
  
  #Estimate Betas
  beta_est = solve(t(C_train)%*%C_train)%*%t(C_train)%*%X_train
  X_train_cor = X_train - C_train%*%beta_est
  X_test_cor = X_test - C_test%*%beta_est
  
  #insert Sex back
  X_test_cor= as.data.frame(X_test_cor)
  X_train_cor = as.data.frame(X_train_cor)
  X_test_cor$Sex= X_testData$Sex
  X_train_cor$Sex= X_trainData$Sex
  
  #run model
  X_train_cor$Sex = factor(X_train_cor$Sex)
  levels(X_train_cor$Sex) <- c("Female", "Male") #Ensure names
  model = e1071::svm(Sex ~ ., data = X_train_cor,
                     type = "C",
                     kernel = "linear",
                     preProc = c("scale", "center"),
                     cost = 1,
                     probability= T,
                     fitted = T)
  #Create confusion
  pred = predict(model, X_test_cor)
  u=union(pred, X_test_cor$Sex)
  t = table(factor(pred, u), factor(X_test_cor$Sex, u))
  resu=confusionMatrix(t) #store confusion matrix
  #Store confusion
  resul=cbind(resul,resu) #make data frame to grab metrics from
  current = as.data.frame(t(resul[[3,j]])) #Grab accuracy confusion metrics, will add sensitivity and specificity
  all_conf = rbind(all_conf,current) #Final confusion metrics for FwCR

}





rm(list=ls())
#Set the current working directory
setwd("C:/Users/VB018797/Documents/Bike Rental Project")

#Get the current working directory
getwd()

#Install packages

install.packages(c("dplyr","plyr","reshape","ggplot2","data.table","NISTunits", "geosphere", "spatial","schoolmath","gridExtra", "DMwR", "corrgram", "caret","usdm", "rpart","MASS","randomForest"), repos = "http://cran.us.r-project.org")
library(NISTunits)
library(ggplot2)
library(gridExtra)
library(DMwR)
library(corrgram)
library(rpart)
library(MASS)
library(usdm)
library(caret)
library(randomForest)


#Load the train data for the bike prediction from the working directory
bike_data = read.csv("day.csv", header=T, stringsAsFactors = FALSE)

str(bike_data)
View(head(bike_data))


#Pre Processing ##

#the variable instant is just carrying an index value and it is not adding anything meaningful to the data so we can drop the variable.#

bike_data$instant = NULL
str(bike_data)
View(head(bike_data))

#Variables which are categorical data are currently of type int hence changing the data type of those variables##
str(bike_data)


bike_data$season = as.factor(bike_data$season)
bike_data$yr = as.factor(bike_data$yr)
bike_data$mnth = as.factor(bike_data$mnth)
bike_data$holiday = as.factor(bike_data$holiday)
bike_data$weekday = as.factor(bike_data$weekday)
bike_data$workingday = as.factor(bike_data$workingday)
bike_data$weathersit = as.factor(bike_data$weathersit)

str(bike_data)


####Converting dteday into date variable with only date as the value and converting into factor from 1 to 31###
bike_data$dteday = as.factor(substr(bike_data$dteday,9,10))

unique(bike_data$dteday)

### Variable casual and registered are removed as their summation values are stored in cnt variable##
bike_data$casual = NULL
bike_data$registered = NULL

str(bike_data)

##Missing value analysis##

#Find the no of missing values in the dataset

View(sum(is.na(bike_data)))

##Outlier Analysis##

##Plotting boxplot for Outlier analysis###

boxplot(bike_data$atemp , Horizontal = TRUE)
boxplot(bike_data$temp , Horizontal = TRUE)
boxplot(bike_data$hum , Horizontal = TRUE)
boxplot(bike_data$windspeed , Horizontal = TRUE)
boxplot(bike_data$cnt , Horizontal = TRUE)

##Outliers are replaced with NA imputed using  KNN method######

####Selecting only 2 variables hum and windspeedbecause remaining are not having any outliers###

bike_outlier = colnames(bike_data)[colnames(bike_data) %in% c("hum", "windspeed")]

for(i in bike_outlier){
  val=bike_data[,i][bike_data [,i] %in% boxplot.stats(bike_data[,i])$out]
  print(length(val))
  bike_data[,i][bike_data [,i] %in% val] =NA
}

sum(is.na(bike_data))


########imputing the missing values of NA with KNN imputation method####

bike_data = knnImputation(bike_data, k=5)
sum(is.na(bike_data))
str(bike_data)



##Feature Selection###

#Plotting the co variance between variables##
corrgram::corrgram(bike_data, order=F, lower.panel = panel.shade,
                   upper.panel = panel.pie, text.panel=panel.txt, main = "Correlation Plot of Bike Data")


#Based on correlation plot removing atemp variable as it is highly correlated with temp variable##
bike_data$atemp = NULL

str(bike_data)

#Analysis of dependent variable cnt on continuous variables##

plot(bike_data$hum, bike_data$cnt , xlab = "Humidity" , ylab = "Count of Bike", main = "Scatter plot of bike count against Humidity" , pch=20, col="red")
plot(bike_data$hum, bike_data$cnt , xlab = "Wind Speed" , ylab = "Count of Bike", main = "Scatter plot of bike count against Windspeed" , pch=20, col="blue")
plot(bike_data$temp, bike_data$cnt, xlab = "Temperature", ylab = "Count of Bike", main = "Scapter plot of bike count against Temperature", pch =20, col = "green")


##########

#Model Evaluation

##########

#Linear Regression Model

##Sampling the data

train_index=sample(1:nrow(bike_data) , 0.8*nrow(bike_data))
model_train = bike_data[train_index,]
model_test = bike_data[-train_index,]

dim(model_train)
dim(model_test)

############Linear regression#################



linear_reg_model = lm(cnt ~., data = model_train)
summary(linear_reg_model)

##Predict test data####

predictions_LRM = predict(linear_reg_model, model_test[,-12])


#Calculate MAPE###

####First create MAPE function#######

MAPE = function(y, z){
  
  mean(abs((y - z)/y))
}

MAPE_LR = MAPE(model_test[,12], predictions_LRM) 
print(MAPE_LR)

X = postResample(model_test[,12], predictions_LRM)
str(X)
RMSE_LR = X["RMSE"]
Rsquared_LR = X["Rsquared"]
MAE_LR = X["MAE"]
print(MAE_LR)

############Decision Tree Model#############

install.packages("rpart.plot")
library("rpart.plot")
##Decision tree with cp value of 0.2###
Decision_Tree_Model = rpart(cnt ~.,data = model_train, method = "anova",control = rpart.control(cp=0.2,minsplit = 5, minbucket = 5, maxdepth=10))
rpart.plot(Decision_Tree_Model)

##Decision tree with cp value of 0.002##

Decision_Tree_Model = rpart(cnt ~.,data = model_train, method = "anova",control = rpart.control(cp=0.002,minsplit = 5, minbucket = 5, maxdepth=10))
rpart.plot(Decision_Tree_Model)

##Plot the cp value ##

plotcp(Decision_Tree_Model)
View(printcp(Decision_Tree_Model))

##Pruning the tree

tree.fit = prune(Decision_Tree_Model, cp = 0.0102805)
rpart.plot(tree.fit)

###Choosing the optimal cp value after Pruning the tree and Decision tree model is created##

Decision_Tree_Model = rpart(cnt ~.,data = model_train, method = "anova",control = rpart.control(cp=0.0102805,minsplit = 5, minbucket = 5, maxdepth=10))

# predict test data##
predict_DTM = predict(Decision_Tree_Model,model_test[,-12])

#Calculate MAPE###
MAPE_DTM = MAPE(model_test[,12], predict_DTM)


print(MAPE_DTM)


###Calculate other paraemeters

Y = postResample(model_test[,12], predict_DTM)

RMSE_DTM = Y["RMSE"]
Rsquared_DTM = Y["Rsquared"]
MAE_DTM = Y["MAE"]

####Random Forest Model####

Random_Forest_model = randomForest(cnt ~., data=model_train, importance=TRUE, ntree=500)
print(Random_Forest_model)
summary(Random_Forest_model)

##Predict test data####

predictions_RF = predict(Random_Forest_model, model_test[,-12])


MAPE_RF = MAPE(model_test[,12], predictions_RF) 
print(MAPE_RF)


###Hyper tuning of parameters in Random Forest Model using Random search CV###

#1. Evaluating the default score using train model in caret package##

Control_train = trainControl(method = "cv", number = 10, search ="grid")

RF_Model = train(cnt ~., data = model_train,method = "rf",trControl=Control_train)
print(RF_Model)

#2. Search for the best mtry

set.seed(1234)
tune_grid=expand.grid(.mtry = c(1:50))
RF_Model_mtry = train(cnt ~., data = model_train,method = "rf",trControl=Control_train,tuneGrid = tune_grid,nodesize=10,ntree=500)
print(RF_Model_mtry)

#The best value of mtry is stored in a variable##
best_mtry = RF_Model_mtry$bestTune$mtry

max(RF_Model_mtry$results$Rsquared)

#3. Find the best maxnode #

store_maxnode=list()
tune_grid = expand.grid(.mtry = best_mtry)
for (maxnodes in c(5:15)) {
  set.seed(1234)
  RF_Model_maxnode = train(cnt ~., data = model_train,method = "rf",trControl=Control_train,tuneGrid = tune_grid,nodesize=14,maxnodes=maxnodes,ntree=500)
  current_iteration = toString(maxnodes)
  store_maxnode[[current_iteration]] = RF_Model_maxnode
  
}
results_mtry = resamples(store_maxnode)
summary(results_mtry)

##Trying again with increased range for maxnode##
store_maxnode=list()
tune_grid = expand.grid(.mtry = best_mtry)
for (maxnodes in c(20:30)) {
  set.seed(1234)
  RF_Model_maxnode = train(cnt ~., data = model_train,method = "rf",trControl=Control_train,tuneGrid = tune_grid,nodesize=14,maxnodes=maxnodes,ntree=500)
  current_iteration = toString(maxnodes)
  store_maxnode[[current_iteration]] = RF_Model_maxnode
  
}
results_mtry = resamples(store_maxnode)
summary(results_mtry)

##4. Search for best ntree###
store_maxtrees = list()
for (ntree in c(300,350,400,450,500,600,700,800,900,1000,2000)) 
  {
  set.seed(5678)
  RF_Model_maxtrees = train(cnt ~., data = model_train,method = "rf",trControl=Control_train,tuneGrid = tune_grid,nodesize=14,maxnodes=29,importance = TRUE , ntree=ntree)
  key = toString(ntree)
  store_maxtrees[[key]] = RF_Model_maxtrees
  
}
results_tree = resamples(store_maxtrees)
summary(results_tree)

##5. Train the model with the best results obtained from 2. 3 and 4 above##

RF_fit = train(cnt ~., data = model_train, method = "rf",trControl=Control_train,tuneGrid = tune_grid,nodesize=14,maxnodes=29,importance = TRUE , ntree= 600)

##Predict test data####

predictions_RF_fit = predict(RF_fit, model_test[,-12])


#Calculate MAPE###

MAPE_RF = MAPE(model_test[,12], predictions_RF_fit) 
print(MAPE_RF)


###Calculate other parameters

Z = postResample(model_test[,12], predictions_RF)
str(Z)
RMSE_RF = Z["RMSE"]
Rsquared_RF = Z["Rsquared"]
MAE_RF = Z["MAE"]



#####Combine all the parameters calculated and add it in variable and create a list for comparison#####

Linear_Regression = c( "RMSE" = RMSE_LR, "Rsquared" = Rsquared_LR, "MAE"= MAE_LR )

Decision_Tree = c("RMSE" = RMSE_DTM, "Rsquared" = Rsquared_DTM, "MAE"= MAE_DTM )

Random_Forest = c("RMSE" = RMSE_RF, "Rsquared" = Rsquared_RF, "MAE"= MAE_RF)

Comparison_Model = cbind(Linear_Regression, Decision_Tree, Random_Forest)

t(Comparison_Model)

##Put the results back into 




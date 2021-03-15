

#import data  
setwd("C:/Users/Rs_Wi1Liam/Desktop/HZW-Classification") #(set your own pwd)
df <- read.csv("bank_personal_loan.csv",header=T)
head(df)

df$Personal.Loan<-factor(df$Personal.Loan, levels=c(0,1),labels=c("No", "Yes")) 
head(df)
table(df[,'Personal.Loan'])
summary(df)
barplot( table(df[,'Personal.Loan']))
set.seed(1234) 
train <- sample(nrow(df), 0.7*nrow(df)) 
df.train <- df[train,] 
df.validate <- df[-train,] 
table(df.train$Personal.Loan) 
table(df.validate$Personal.Loan)


#####Logit
fit.logit <- glm(Personal.Loan~., data=df.train, family=binomial()) #Logit
summary(fit.logit)

fit.logit <- glm(Personal.Loan~Income+Family+Education+CD.Account+CreditCard, data=df.train, family=binomial())
summary(fit.logit)

prob <- predict(fit.logit, df.validate, type="response") 
logit.pred <- factor(prob > .5, levels=c(FALSE, TRUE), 
                     labels=c("benign", "malignant"))
logit.perf <- table(df.validate$Personal.Loan, logit.pred,
                    dnn=c("Actual", "Predicted")) 
logit.perf




#####decision tree
library(rpart) 
set.seed(1234) 
dtree <- rpart(Personal.Loan ~ ., data=df.train, method="class",  parms=list(split="information"))   
print(dtree)
summary(dtree)
dtree$cptable

plotcp(dtree) 
dtree.pruned <- prune(dtree, cp=.0213) 
dtree.pruned$cptable
library(rpart.plot) 
prp(dtree, type = 2, extra = 104, fallen.leaves = TRUE, main="Decision Tree") 
prp(dtree.pruned, type = 2, extra = 104, fallen.leaves = TRUE, main="Decision Tree") 
dtree.pred <- predict(dtree.pruned, df.validate, type="class")
dtree.perf <- table(df.validate$Personal.Loan, dtree.pred, dnn=c("Actual", "Predicted")) 
dtree.perf

#####Bagging 
library(adabag)
bagging.model <- bagging(Personal.Loan~.,data= df.train)
bagging.pred <- predict(bagging.model,newdata=df.validate)$class
bagging.perf <- table(df.validate$Personal.Loan,bagging.pred)
bagging.perf


#####Boosting
boosting.model <- boosting(Personal.Loan~.,data=df.train)
boosting.pred <- predict(boosting.model,newdata=df.validate)$class
boosting.perf <- table(df.validate$Personal.Loan,boosting.pred)
boosting.perf

#####random forest
library(randomForest) 
set.seed(1234) 
fit.forest <- randomForest(Personal.Loan~., data=df.train,
                           importance=TRUE) 
fit.forest


forest.pred <- predict(fit.forest, df.validate) 
forest.perf <- table(df.validate$Personal.Loan, forest.pred,
                     dnn=c("Actual", "Predicted")) 
forest.perf


#####SVM 
library(e1071) 
set.seed(1234) 
fit.svm <- svm(Personal.Loan~., data=df.train)  
fit.svm


svm.pred <- predict(fit.svm, na.omit(df.validate))
svm.perf <- table(na.omit(df.validate)$Personal.Loan, 
                  svm.pred, dnn=c("Actual", "Predicted")) 
svm.perf

tuned <- tune.svm(Personal.Loan~., data=df.train, 
                  gamma=10^(-3:1),
                  cost=10^(-3:3)) 
tuned

fit.svm <- svm(Personal.Loan~., data=df.train, gamma=.01, cost=1000)
svm.pred <- predict(fit.svm, na.omit(df.validate)) 
svm.perf <- table(na.omit(df.validate)$Personal.Loan, 
                  svm.pred, dnn=c("Actual", "Predicted"))
svm.perf


#####Bayes
library(e1071)

nb.model <- naiveBayes(Personal.Loan~.,data = df.train)
nb.pred <- predict(nb.model, newdata=df.validate)
nb.perf <- table(df.validate$Personal.Loan, nb.pred, 
                 dnn=c("Actual", "Predicted")) 
nb.perf



performance <- function(table, n=2){  
  if(!all(dim(table) == c(2,2)))
    stop("Must be a 2 x 2 table") 
  tn = table[1,1]
  fp = table[1,2] 
  fn = table[2,1] 
  tp = table[2,2] 
  sensitivity = tp/(tp+fn) 
  specificity = tn/(tn+fp)
  ppp = tp/(tp+fp)
  npp = tn/(tn+fn)
  hitrate = (tp+tn)/(tp+tn+fp+fn)
  result <- paste("Sensitivity = ", round(sensitivity, n) , 
                  "\nSpecificity = ", round(specificity, n), 
                  "\nPositive Predictive Value = ", round(ppp, n),
                  "\nNegative Predictive Value = ", round(npp, n),
                  "\nAccuracy = ", round(hitrate, n), "\n", sep="") 
  cat(result) 
}

performance(logit.perf)
performance(dtree.perf)
performance(forest.perf)
performance(svm.perf)
performance(nb.perf)
performance(bagging.perf)
performance(boosting.perf)

par(mfrow=c(3,3))

#Logistic ROC
library(AUC)
plot(roc(logit.pred,as.factor(df.validate$Personal.Loan)),col=rainbow(10),main='ROC curve for the Logistic Regression Model')
lr.auc<-auc(roc(logit.pred,as.factor(df.validate$Personal.Loan)))
text(0.8,0.2,labels=paste('AUC=',round(lr.auc,3),sep=''),cex=1.2)

#Dtree ROC
library(AUC)
plot(roc(dtree.pred,as.factor(df.validate$Personal.Loan)),col=rainbow(10),main='ROC curve for the Dtree Model')
lr.auc<-auc(roc(dtree.pred,as.factor(df.validate$Personal.Loan)))
text(0.8,0.2,labels=paste('AUC=',round(lr.auc,3),sep=''),cex=1.2)

#Random forest ROC
library(AUC)
plot(roc(forest.pred,as.factor(df.validate$Personal.Loan)),col=rainbow(10),main='ROC curve for the Random Forest Model')
lr.auc<-auc(roc(forest.pred,as.factor(df.validate$Personal.Loan)))
text(0.8,0.2,labels=paste('AUC=',round(lr.auc,3),sep=''),cex=1.2)

#sVM ROC
library(AUC)
plot(roc(svm.pred,as.factor(df.validate$Personal.Loan)),col=rainbow(10),main='ROC curve for the SVM Model')
lr.auc<-auc(roc(svm.pred,as.factor(df.validate$Personal.Loan)))
text(0.8,0.2,labels=paste('AUC=',round(lr.auc,3),sep=''),cex=1.2)

#Nb ROC
library(AUC)
plot(roc(nb.pred,as.factor(df.validate$Personal.Loan)),col=rainbow(10),main='ROC curve for the Bayesian Model')
lr.auc<-auc(roc(nb.pred,as.factor(df.validate$Personal.Loan)))
text(0.8,0.2,labels=paste('AUC=',round(lr.auc,3),sep=''),cex=1.2)


#Bagging ROC
library(AUC)
plot(roc(bagging.pred,as.factor(df.validate$Personal.Loan)),col=rainbow(10),main='ROC curve for the Bagging Model')
lr.auc<-auc(roc(bagging.pred,as.factor(df.validate$Personal.Loan)))
text(0.8,0.2,labels=paste('AUC=',round(lr.auc,3),sep=''),cex=1.2)

#Boosting ROC
library(AUC)
plot(roc(boosting.pred,as.factor(df.validate$Personal.Loan)),col=rainbow(10),main='ROC curve for the Boosting Model')
lr.auc<-auc(roc(boosting.pred,as.factor(df.validate$Personal.Loan)))
text(0.8,0.2,labels=paste('AUC=',round(lr.auc,3),sep=''),cex=1.2)





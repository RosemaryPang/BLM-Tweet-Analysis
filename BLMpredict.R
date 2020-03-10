################################################
########       BLM prediction II        ########
########          Rosemary Pang         ########
########     Bayes Lemma Prediction     ########
########            Oct.8. 2019         ########
################################################

#### Seting work directory and load packages ####
setwd("/Users/mpang/Desktop")
library(wordcloud)
library(tm)
library(SnowballC)
library(syuzhet)
library(e1071)
library(dplyr)
library(caret)
library(doMC)
library(quanteda)
library(data.table)
library(text2vec)


##### Load Data #####
### Lemma
train_lemma <- read.csv("lemma_train1000.csv",stringsAsFactors = FALSE)
train_lemma$label <- as.factor(train_lemma$label) # 1000 labeled tweet for training (lemma)
testlb_lemma <- read.csv("lemma_test200.csv",stringsAsFactors = FALSE)
testlb_lemma$label <- as.factor(testlb_lemma$label) # 200 labeled tweet for testing (lemma)
testulb_lemma <- read.csv("lemma_unlabel20860.csv",stringsAsFactors = FALSE) # unlabeled data for testing (stem)
Anntweet_lemma <- rbindlist(list(train_lemma,testlb_lemma)) 

#### Bag of Words (Get Ready for Data Analysis) ####
corpus.train_lemma <- Corpus(VectorSource(train_lemma$clean_tweet)) ## build corpus for further analysis
corpus.train_lemma# Inspect the corpus
inspect(corpus.train_lemma[1:3])

corpus.testlb_lemma <- Corpus(VectorSource(testlb_lemma$clean_tweet)) 
corpus.testulb_lemma <- Corpus(VectorSource(testulb_lemma$clean_tweet))
corpus.anntweet_lemma <- Corpus(VectorSource(Anntweet_lemma$clean_tweet))

dtm.train_lemma <- DocumentTermMatrix(corpus.train_lemma)# build document term matrix
inspect(dtm.train_lemma[40:50,1:10])
dtm.testlb_lemma <- DocumentTermMatrix(corpus.testlb_lemma)
dtm.testulb_lemma <- DocumentTermMatrix(corpus.testulb_lemma)
dtm.anntweet_lemma <- DocumentTermMatrix(corpus.anntweet_lemma)


##### Bayes #####
dim(dtm.train_lemma)
fivefreq_lemma <- findFreqTerms(dtm.train_lemma, 5)
length((fivefreq_lemma))
dtm.train.nb_lemma <- DocumentTermMatrix(corpus.train_lemma, control=list(dictionary=fivefreq_lemma))
dim(dtm.train.nb_lemma)
dtm.testulb.nb_lemma <- DocumentTermMatrix(corpus.testulb_lemma, control=list(dictionary=fivefreq_lemma))
dim(dtm.testulb.nb_lemma)
# convert word frequencies to yes and no
convert_count <- function(x) {
  y <- ifelse(x>0, 1, 0)
  y <- factor(y,levels=c(0,1),labels=c("No","Yes"))
  y
}
trainNB_lemma <- apply(dtm.train.nb_lemma, 2, convert_count)
testulbNB_lemma <- apply(dtm.testulb.nb_lemma, 2, convert_count)

# Train the classifier (Accuracy 0.725)
system.time(classifier <- naiveBayes(trainNB_lemma,train_lemma$label))
system.time(pred <- predict(classifier,newdata=testulbNB_lemma))
# Predict unlabled tweets
testulb_lemma$label<- pred #20860 total

# All tweets 22060 total
FullTweet <- rbindlist(list(train_lemma,testlb_lemma,testulb_lemma))

##### Merging Label and Geolocation #####
location <- read.csv("tweets_state_county.csv",colClasses=c("NULL",NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA))

LocLabel <- merge(location,FullTweet,by="id")
write.csv(LocLabel,"LocLabel.csv")

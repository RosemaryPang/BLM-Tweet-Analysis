################################################
########       BLM prediction II        ########
########          Rosemary Pang         ########
########    Bayes Lemma Stem Original   ########
########          Sept.19. 2019         ########
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
### Original
train_original <- read.csv("original_train1000.csv",colClasses=c("NULL",NA,NA,NA,NA),stringsAsFactors = FALSE)
train_original$label <- as.factor(train_original$label) # 1000 labeled tweet for training (original)
testlb_original <- read.csv("original_test200.csv",colClasses=c("NULL",NA,NA,NA,NA),stringsAsFactors = FALSE)
testlb_original$label <- as.factor(testlb_original$label) # 200 labeled tweet for testing (original)
testulb_original <- read.csv("original_unlabel20860.csv",colClasses=c("NULL",NA,NA,NA),stringsAsFactors = FALSE) # unlabeled data for testing (original)
Anntweet_original <- rbindlist(list(train_original,testlb_original)) 

### Stem
train_stem <- read.csv("stem_train1000.csv",colClasses=c("NULL",NA,NA,NA,NA),stringsAsFactors = FALSE)
train_stem$label <- as.factor(train_stem$label) # 1000 labeled tweet for training (stem)
testlb_stem <- read.csv("stem_test200.csv",colClasses=c("NULL",NA,NA,NA,NA),stringsAsFactors = FALSE)
testlb_stem$label <- as.factor(testlb_stem$label) # 200 labeled tweet for testing (stem)
testulb_stem <- read.csv("stem_unlabel20860.csv",colClasses=c("NULL",NA,NA,NA),stringsAsFactors = FALSE) # unlabeled data for testing (stem)
Anntweet_stem <- rbindlist(list(train_stem,testlb_stem)) 

### Lemma
train_lemma <- read.csv("lemma_train1000.csv",colClasses=c("NULL",NA,NA,NA,NA),stringsAsFactors = FALSE)
train_lemma$label <- as.factor(train_lemma$label) # 1000 labeled tweet for training (lemma)
testlb_lemma <- read.csv("lemma_test200.csv",colClasses=c("NULL",NA,NA,NA,NA),stringsAsFactors = FALSE)
testlb_lemma$label <- as.factor(testlb_lemma$label) # 200 labeled tweet for testing (lemma)
testulb_lemma <- read.csv("lemma_unlabel20860.csv",colClasses=c("NULL",NA,NA,NA),stringsAsFactors = FALSE) # unlabeled data for testing (stem)
Anntweet_lemma <- rbindlist(list(train_lemma,testlb_lemma)) 


##### Original #####

#### Bag of Words (Get Ready for Data Analysis) ####
corpus.train_original <- Corpus(VectorSource(train_original$clean_tweet)) ## build corpus for further analysis
corpus.train_original # Inspect the corpus
inspect(corpus.train_original[1:3])

corpus.testlb_original <- Corpus(VectorSource(testlb_original$clean_tweet)) 
corpus.testulb_original <- Corpus(VectorSource(testulb_original$clean_tweet))
corpus.anntweet_original <- Corpus(VectorSource(Anntweet_original$clean_tweet))

dtm.train_original <- DocumentTermMatrix(corpus.train_original)# build document term matrix
inspect(dtm.train_original[40:50,1:10])
dtm.testlb_original <- DocumentTermMatrix(corpus.testlb_original)
dtm.testulb_original <- DocumentTermMatrix(corpus.testulb_original)
dtm.anntweet_original <- DocumentTermMatrix(corpus.anntweet_original)

#### Sentiment ####
word.df_original=as.vector(testlb_original$clean_tweet)
emotion.df_original <- get_nrc_sentiment(word.df_original) #this gives us the different emotions present in each of the tweets
emotion.df2_oritinal <- cbind(testlb_original$clean_tweet, emotion.df_original) #this shows us the tweets and their emotions

#Now we need to extract the sentiment score for each of the tweets 
sentvalue_original <- get_sentiment(word.df_original) #these are the sentiment values for the tweets
sentiment_original <- cbind(testlb_original$clean_tweet,testlb_original$label,sentvalue_original)

# Set sentiment category and cross matrix 
sentiment_original <- as.data.frame(sentiment_original)
sentiment_original$sentvalue2 <- as.numeric(sentvalue_original)
sentiment_original$category[sentiment_original$sentvalue2 > 0] <- 1
sentiment_original$category[sentiment_original$sentvalue2 < 0] <- 2
sentiment_original$category[sentiment_original$sentvalue2 == 0] <- 3

# Accuracy
table <-table(sentiment_original$V2,sentiment_original$category)
n <- sum(table)
nc <- nrow(table)
diag = diag(table)
rowsums <- apply(table,1,sum)
colsums <- apply(table,2,sum)
p <- rowsums/n
q <- colsums/n

accuracy <- sum(diag)/n
precision <- diag/colsums
recall <- diag/rowsums
f1 <- 2*precision*recall/(precision+recall)

#### Bayes ####
dim(dtm.train_original)
fivefreq_original <- findFreqTerms(dtm.train_original, 5)
length((fivefreq_original))
dtm.train.nb_original <- DocumentTermMatrix(corpus.train_original, control=list(dictionary=fivefreq_original))
dim(dtm.train.nb_original)
dtm.testlb.nb_original <- DocumentTermMatrix(corpus.testlb_original, control=list(dictionary=fivefreq_original))
dim(dtm.testlb.nb_original)
# convert word frequencies to yes and no
convert_count <- function(x) {
  y <- ifelse(x>0, 1, 0)
  y <- factor(y,levels=c(0,1),labels=c("No","Yes"))
  y
}
trainNB_original <- apply(dtm.train.nb_original, 2, convert_count)
testlbNB_original <- apply(dtm.testlb.nb_original, 2, convert_count)

# Train the classifier (Accuracy 0.745)
system.time(classifier <- naiveBayes(trainNB_original,train_original$label))
system.time(pred <- predict(classifier,newdata=testlbNB_original))

# Accuracy
table <- table("Actual"=testlb_original$label,"Predictions"=pred)
n <- sum(table)
nc <- nrow(table)
diag = diag(table)
rowsums <- apply(table,1,sum)
colsums <- apply(table,2,sum)
p <- rowsums/n
q <- colsums/n

accuracy <- sum(diag)/n
precision <- diag/colsums
recall <- diag/rowsums
f1 <- 2*precision*recall/(precision+recall)

#Most Positive Negative Words
dfm.train_original <- dfm(train_original$clean_tweet)
nb.mod_original <- textmodel_nb(dfm.train_original,train_original$label,distribution = "Bernoulli")
sort(nb.mod_original$PcGw[1,],dec=T)[1:10] # Most Positive words
sort(nb.mod_original$PcGw[2,],dec=T)[1:10] # Most Negative words
sort(nb.mod_original$PcGw[3,],dec=T)[1:10] # Most Neutural words



##### Stem #####

#### Bag of Words (Get Ready for Data Analysis) ####
corpus.train_stem <- Corpus(VectorSource(train_stem$clean_tweet)) ## build corpus for further analysis
corpus.train_stem # Inspect the corpus
inspect(corpus.train_stem[1:3])

corpus.testlb_stem <- Corpus(VectorSource(testlb_stem$clean_tweet)) 
corpus.testulb_stem <- Corpus(VectorSource(testulb_stem$clean_tweet))
corpus.anntweet_stem <- Corpus(VectorSource(Anntweet_stem$clean_tweet))

dtm.train_stem <- DocumentTermMatrix(corpus.train_stem)# build document term matrix
inspect(dtm.train_stem[40:50,1:10])
dtm.testlb_stem <- DocumentTermMatrix(corpus.testlb_stem)
dtm.testulb_stem <- DocumentTermMatrix(corpus.testulb_stem)
dtm.anntweet_stem <- DocumentTermMatrix(corpus.anntweet_stem)

#### Sentiment ####
word.df_stem <- as.vector(testlb_stem$clean_tweet)
emotion.df_stem <- get_nrc_sentiment(word.df_stem) #this gives us the different emotions present in each of the tweets
emotion.df2_stem <- cbind(testlb_stem$clean_tweet, emotion.df_stem) #this shows us the tweets and their emotions

#Now we need to extract the sentiment score for each of the tweets 
sentvalue_stem <- get_sentiment(word.df_stem) #these are the sentiment values for the tweets
sentiment_stem <- cbind(testlb_stem$clean_tweet,testlb_stem$label,sentvalue_stem)

# Set sentiment category and cross matrix 
sentiment_stem <- as.data.frame(sentiment_stem)
sentiment_stem$sentvalue2 <- as.numeric(sentvalue_stem)
sentiment_stem$category[sentiment_stem$sentvalue2 > 0] <- 1
sentiment_stem$category[sentiment_stem$sentvalue2 < 0] <- 2
sentiment_stem$category[sentiment_stem$sentvalue2 == 0] <- 3

# Accuracy
table <-table(sentiment_stem$V2,sentiment_stem$category)
n <- sum(table)
nc <- nrow(table)
diag = diag(table)
rowsums <- apply(table,1,sum)
colsums <- apply(table,2,sum)
p <- rowsums/n
q <- colsums/n

accuracy <- sum(diag)/n
precision <- diag/colsums
recall <- diag/rowsums
f1 <- 2*precision*recall/(precision+recall)

#### Bayes ####
dim(dtm.train_stem)
fivefreq_stem <- findFreqTerms(dtm.train_stem, 5)
length((fivefreq_stem))
dtm.train.nb_stem <- DocumentTermMatrix(corpus.train_stem, control=list(dictionary=fivefreq_stem))
dim(dtm.train.nb_stem)
dtm.testlb.nb_stem <- DocumentTermMatrix(corpus.testlb_stem, control=list(dictionary=fivefreq_stem))
dim(dtm.testlb.nb_stem)
# convert word frequencies to yes and no
convert_count <- function(x) {
  y <- ifelse(x>0, 1, 0)
  y <- factor(y,levels=c(0,1),labels=c("No","Yes"))
  y
}
trainNB_stem <- apply(dtm.train.nb_stem, 2, convert_count)
testlbNB_stem <- apply(dtm.testlb.nb_stem, 2, convert_count)

# Train the classifier (Accuracy 0.725)
system.time(classifier <- naiveBayes(trainNB_stem,train_stem$label))
system.time(pred <- predict(classifier,newdata=testlbNB_stem))

# Accuracy
table <- table("Actual"=testlb_stem$label,"Predictions"=pred)
n <- sum(table)
nc <- nrow(table)
diag = diag(table)
rowsums <- apply(table,1,sum)
colsums <- apply(table,2,sum)
p <- rowsums/n
q <- colsums/n

accuracy <- sum(diag)/n
precision <- diag/colsums
recall <- diag/rowsums
f1 <- 2*precision*recall/(precision+recall)

#Most Positive Negative Words
dfm.train_stem <- dfm(train_stem$clean_tweet)
nb.mod_stem <- textmodel_nb(dfm.train_stem,train_stem$label,distribution = "Bernoulli")
sort(nb.mod_stem$PcGw[1,],dec=T)[1:10] # Most Positive words
sort(nb.mod_stem$PcGw[2,],dec=T)[1:10] # Most Negative words
sort(nb.mod_stem$PcGw[3,],dec=T)[1:10] # Most Neutural words



##### Lemma #####

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

#### Sentiment ####
word.df_lemma <- as.vector(testlb_lemma$clean_tweet)
emotion.df_lemma <- get_nrc_sentiment(word.df_lemma) #this gives us the different emotions present in each of the tweets
emotion.df2_lemma <- cbind(testlb_lemma$clean_tweet, emotion.df_lemma) #this shows us the tweets and their emotions

#Now we need to extract the sentiment score for each of the tweets 
sentvalue_lemma <- get_sentiment(word.df_lemma) #these are the sentiment values for the tweets
sentiment_lemma <- cbind(testlb_lemma$clean_tweet,testlb_lemma$label,sentvalue_lemma)

# Set sentiment category and cross matrix 
sentiment_lemma <- as.data.frame(sentiment_lemma)
sentiment_lemma$sentvalue2 <- as.numeric(sentvalue_lemma)
sentiment_lemma$category[sentiment_lemma$sentvalue2 > 0] <- 1
sentiment_lemma$category[sentiment_lemma$sentvalue2 < 0] <- 2
sentiment_lemma$category[sentiment_lemma$sentvalue2 == 0] <- 3

# Accuracy
table <-table(sentiment_lemma$V2,sentiment_lemma$category)
n <- sum(table)
nc <- nrow(table)
diag = diag(table)
rowsums <- apply(table,1,sum)
colsums <- apply(table,2,sum)
p <- rowsums/n
q <- colsums/n

accuracy <- sum(diag)/n
precision <- diag/colsums
recall <- diag/rowsums
f1 <- 2*precision*recall/(precision+recall)

#### Bayes ####
dim(dtm.train_lemma)
fivefreq_lemma <- findFreqTerms(dtm.train_lemma, 5)
length((fivefreq_lemma))
dtm.train.nb_lemma <- DocumentTermMatrix(corpus.train_lemma, control=list(dictionary=fivefreq_lemma))
dim(dtm.train.nb_lemma)
dtm.testlb.nb_lemma <- DocumentTermMatrix(corpus.testlb_lemma, control=list(dictionary=fivefreq_lemma))
dim(dtm.testlb.nb_lemma)
# convert word frequencies to yes and no
convert_count <- function(x) {
  y <- ifelse(x>0, 1, 0)
  y <- factor(y,levels=c(0,1),labels=c("No","Yes"))
  y
}
trainNB_lemma <- apply(dtm.train.nb_lemma, 2, convert_count)
testlbNB_lemma <- apply(dtm.testlb.nb_lemma, 2, convert_count)

# Train the classifier (Accuracy 0.725)
system.time(classifier <- naiveBayes(trainNB_lemma,train_lemma$label))
system.time(pred <- predict(classifier,newdata=testlbNB_lemma))

# Accuracy
table <- table("Actual"=testlb_lemma$label,"Predictions"=pred)
n <- sum(table)
nc <- nrow(table)
diag = diag(table)
rowsums <- apply(table,1,sum)
colsums <- apply(table,2,sum)
p <- rowsums/n
q <- colsums/n

accuracy <- sum(diag)/n
precision <- diag/colsums
recall <- diag/rowsums
f1 <- 2*precision*recall/(precision+recall)

#Most Positive Negative Words
dfm.train_lemma <- dfm(train_lemma$clean_tweet)
nb.mod_lemma <- textmodel_nb(dfm.train_lemma,train_lemma$label,distribution = "Bernoulli")
sort(nb.mod_lemma$PcGw[1,],dec=T)[1:10] # Most Positive words
sort(nb.mod_lemma$PcGw[2,],dec=T)[1:10] # Most Negative words
sort(nb.mod_lemma$PcGw[3,],dec=T)[1:10] # Most Neutural words

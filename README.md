# BLM-Tweet-Analysis

This is a project written with Hyunjoon Park, Sara Francisco, and Lulu Peng. This project use geo-located tweets in 2014 and 2015 to identify stances towards BLM movement on regional level across the United State, and then explore how stances in turn is associated with the obeisity of black population. We annotated 1200 tweets and use different machine learning methods (Sentimental analysis, Naive Bayes, SVM, and LSTM) to predict the stance of unannotated tweets. 

# BLM_Pang.R

This R file analyze the accuracy of machine learning using sentimental analysis and naive Bayes methods. It also compares the results using different format of tweet data, including original data, stem and lemma. 

# BLMpredict.R

After comparing the accuracy, precision and recall of different models. Naive Bayes model using lemma data has the best result. This R file predicts unannotated tweets using naive Bayes model and merges stance to geolocation for further analysis. 

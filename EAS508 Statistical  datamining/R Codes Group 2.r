#GroupNumber:2
#project: Product recommender system
#Datasource: Amazon ratings data (http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Digital_Music.csv)
#Loading required packages for recommend system
library(recommenderlab)
library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(DT)
library(knitr)
library(grid)
library(gridExtra)
library(corrplot)
library(methods)
library(Matrix)
library(stringi)
library(reshape2)
library(recommenderlab)
#setwd("")
#reading data from csv file
#datasource: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Digital_Music.csv
#Magazine_Subscriptions(only ratings) from amazon
#prod_ratings=fread("Magazine_Subscriptions.csv")
#Digital Music from Amazon datasets
#prod_ratings=fread("ratings_Digital_Music.csv")
prod_ratings=fread("Data Group 2.csv")
#Rename the column names
#prod_ratings = rename(prod_ratings,productId = V1, userId = V2,rating = V3,timestamp = V4)#for ms
prod_ratings = rename(prod_ratings, userId = V1, productId = V2,rating = V3,timestamp = V4)#for music
head(prod_ratings,10)

#To remove duplicate reviews (if any)
prod_ratings[duplicated(prod_ratings[,1:2]),]
prod_ratings = prod_ratings[!duplicated(prod_ratings[,1:2]),]
kable(head(prod_ratings))

#Amazon digital music data set
#Form subset of data to avoid sparse issues and find good number of review-product combination

set.seed(101)
sample_1 = data.frame(head(sort(table(prod_ratings$userId), decreasing = T), 1835))
colnames(sample_1) = c("userId", "count")

sub_users_merge = merge(prod_ratings, sample_1, by = "userId")

sample_2 = data.frame(head(sort(table(sub_users_merge$productId), decreasing = T), 988))
colnames(sample_2) = c("productId", "count")

sub_products_merge = merge(sub_users_merge, sample_2, by = "productId")

final_subset_data = subset(sub_products_merge, select = c("userId", "productId", "rating"))


#Create ratings matrix with rows as users and columns as products, We are ignoring timestamp here.
ratingmat = dcast(final_subset_data, userId~productId, value.var = "rating", na.rm=FALSE)
#Remove user ids
ratingmat = as.matrix(ratingmat[,-1])
#Conversion of ratings matrix to real rating matrix to obtain dense matrix
ratingmat = as(ratingmat, "realRatingMatrix")

#Data Analysis and plots

#Generate heatmap of ratings
dataMetric1 = ratingmat[rowCounts(ratingmat) > 25, colCounts(ratingmat) > 25]
image(dataMetric1, main = "Heatmap of Users and Products")
#Histogram of ratings
qplot(final_subset_data$rating, geom="histogram", main = "Histogram of Ratings", xlab = "Rating Scores", binwidth = 0.5, fill=I("sky blue"),col=I("black"))


# Ratings per Product mean
final_subset_data %>% 
  group_by(productId) %>% 
  summarize(mean_product_rating = mean(rating)) %>% 
  ggplot(aes(mean_product_rating)) + geom_histogram(fill = "orange", color = "grey20") + coord_cartesian(c(1,5)) + labs( title= "Histogram of Product Ratings (Mean)", y="Product", x = "mean product rating")

# Ratings per Product count
final_subset_data %>% 
  group_by(productId) %>% 
  summarize(number_of_ratings_per_product = n()) %>% 
  ggplot(aes(number_of_ratings_per_product)) + 
  geom_bar(fill = "#FF6666", color = "grey20", width = 1) + coord_cartesian(c(0,40)) + labs( title= "Histogram of Product Ratings (Count)", y="Product", x = "Rating Count")

# Ratings per user mean
final_subset_data %>% 
  group_by(userId) %>% 
  summarize(mean_user_rating = mean(rating)) %>% 
  ggplot(aes(mean_user_rating)) +
  geom_histogram(fill = "light green", color = "black",bins = 15) +
  labs( title= "Histogram of user given Ratings (Mean)", y="Users", x = "Mean user rating")


# Ratings per user count
final_subset_data %>% 
  group_by(userId) %>% 
  summarize(number_of_ratings_per_user = n()) %>% 
  ggplot(aes(number_of_ratings_per_user)) + 
  geom_bar(fill = "Dark green", color = "black") + coord_cartesian(c(3, 50))+
  labs( title= "Histogram of user given Ratings (Count)", y="Users", x = "Rating count")



set.seed(101)
#Create Evaluation scheme
evaluation = evaluationScheme(ratingmat, method="split", train=0.8, given=1, goodRating=5)

#Evaluation datasets 
prod_ratings_train = getData(evaluation, "train")
prod_ratings_known = getData(evaluation, "known")
prod_ratings_unknown = getData(evaluation, "unknown")

#Recommender Model creation for UBCF using Cosine similarity. We take 10 nearest neighbours
ubcf_rec_1 = Recommender(prod_ratings_train, method = "UBCF",param=list(method="Cosine",nn=10)) 
ubcf_pred_1 = predict(ubcf_rec_1, prod_ratings_known[1], n=5,type="ratings")
List_cos = as(ubcf_pred_1, "list")
#Shows the recommendations and predictions
List_cos
calcPredictionAccuracy(ubcf_pred_1, prod_ratings_unknown[1])

#Recommender Model creation for UBCF using Jaccard similarity. We take 10 nearest neighbours
ubcf_rec_2 = Recommender(prod_ratings_train, method = "UBCF", param=list(method="Jaccard",nn=10)) 
ubcf_pred_2 = predict(ubcf_rec_2, prod_ratings_known[1], n=5,type="ratings")
List_jac = as(ubcf_pred_2, "list")
#Shows the recommendations and predictions
List_jac
calcPredictionAccuracy(ubcf_pred_2, prod_ratings_unknown[1])

#Recommender Model creation for UBCF using Pearson similarity. We take 10 nearest neighbours
#ubcf_rec_3 = Recommender(prod_ratings_train, method = "UBCF", param=list(method="Pearson",nn=10)) 
#ubcf_pred_3 = predict(ubcf_rec_3, prod_ratings_known[1], n=5,type="ratings")
#Shows the recommendations and predictions
#List_pearson = as(ubcf_pred_3, "list")
#List_pearson

#Recommender Model creation for UBCF using Manhattan similarity. We take 10 nearest neighbours
ubcf_rec_4 = Recommender(prod_ratings_train, method = "UBCF", param=list(method="Manhattan",nn=10)) 
ubcf_pred_4 = predict(ubcf_rec_4, prod_ratings_known[1], n=5,type="ratings")
#Shows the recommendations and predictions
List_Man = as(ubcf_pred_4, "list")
List_Man
calcPredictionAccuracy(ubcf_pred_2, prod_ratings_unknown[1])


#Summarising the accuracies
accuracy_models = rbind(
  UBCF_Cosine = calcPredictionAccuracy(ubcf_pred_1, prod_ratings_unknown[1]),
  UBCF_Jaccard = calcPredictionAccuracy(ubcf_pred_2, prod_ratings_unknown[1]),
  UBCF_Manhattan = calcPredictionAccuracy(ubcf_pred_4, prod_ratings_unknown[1])
  #UBCF_Pearson = calcPredictionAccuracy(ubcf_pred_3, prod_ratings_unknown[1]),
)

accuracy_dataframe = round(as.data.frame(accuracy_models), 3)
kable(accuracy_dataframe[order(accuracy_dataframe$RMSE),])
plot(accuracy_dataframe)

#Summarising the top results for all algorithms
pred_top = predict(ubcf_rec_1, prod_ratings_known[1], n=5)
Top_res_cos = as(pred_top, "list")
Top_res_cos
pred_top = predict(ubcf_rec_2, prod_ratings_known[1], n=5)
Top_res_jac = as(pred_top, "list")
Top_res_jac
#pred_top = predict(ubcf_rec_3, prod_ratings_known[1], n=5)
#Top_res_pearson = as(pred_top, "list")
#Top_res_pearson
pred_top = predict(ubcf_rec_4, prod_ratings_known[1], n=5)
Top_res_man = as(pred_top, "list")
Top_res_man


Algorithms_used = c ('UBCF_Cosine','UBCF_Jaccard','UBCF_Manhattan')
Top_Results_for_UserID_1 = c(Top_res_cos,Top_res_jac,Top_res_man)

Summary_DT = data.table(
  Algorithms_used,
  Top_Results_for_UserID_1
)
kable(Summary_DT)



                                     
                                     
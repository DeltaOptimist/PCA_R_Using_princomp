# Removes List of data 
rm(list = ls())

# About PCA

# PCA is a very popular method of dimensionality reduction because it provides a way to easily 
# reduce the dimensions and is easy to understand.

# The princomp() function in R calculates the principal components of any data. 
# We will also compare our results by calculating eigenvectors and eigenvalues separately. 
# Let’s use the IRIS dataset.





# Taking the numeric part of the IRIS data for PCA
data_iris <- iris[1:4]
# The iris dataset having 150 observations (rows) with 4 features.
#Let’s use the cov() function to calculate the covariance matrix of the loaded iris data set.

# Calculating the covariance matrix
Cov_data <- cov(data_iris)

# The next step is to calculate the eigenvalues and eigenvectors. 
# We can use the eigen() function to do this automatically for us.
# Find out the eigenvectors and eigenvalues using the covariance matrix
Eigen_data <- eigen(Cov_data)
# We have calculated the Eigen values from the data. 

# We will now look at the PCA function princomp() which automatically calculates these values.
# Let’s calculate the components and compare the values.

#using inbuilt function 
PCA_data <- princomp(data_iris, cor = "False")


#Compare the output variance
Eigen_data$values

PCA_data$sdev^2

# There is a slight difference due to squaring in 
# PCA_data but the outputs are more or less simil ar. 
# We can also compare the eigenvectors of both models.


PCA_data$loadings[,1:4]

Eigen_data$vectors

# This time the eigenvectors calculated are same and there is no difference.
# -------------------------------------------------------------------------------
# Let us now understand our model. We transformed our 4 features into 4 new orthogonal components.
# To know the importance of the first component, we can view the summary of the model.
# -------------------------------------------------------------------------------
  
summary(PCA_data)

# From the Proportion of Variance, we see that the first component has an importance of 92.5% 
# in predicting the class while the second principal component has an importance of 5.3% and so on. 
# This means that using just the first component instead of all the 4 features will make our model accuracy 
# to be about 92.5% while we use only one-fourth of the entire set of features.


# If we want the higher accuracy, we can take the first two components together and obtain a cumulative accuracy 
# of up to 97.7%. We can also understand how our features are transformed by using the biplot function on our model.

biplot(PCA_data)

screeplot(PCA_data, type="lines")

# This plot shows the bend at the second principal component. 
# Let us now fit two naive Bayes models.
# 1. one over the entire data.
# 2. The second on the first principal component.

# We will calculate the difference in accuracy between these two models.

# Select the first principal component for the second model
model2 = PCA_data$loadings[,1]


#For the second model, we need to calculate scores by multiplying our loadings with the data.
model2_scores <- as.matrix(data_iris) %*% model2

# Loading libraries for naiveBayes model
library(class)
# SVM library
install.packages("e1071")

library(e1071)

# Fitting the first model over the entire data
mod1<-naiveBayes(iris[,1:4], iris[,5])

# Fitting the second model using the first principal component
mod2<-naiveBayes(model2_scores, iris[,5])

# Accuracy for the first model
table(predict(mod1, iris[,1:4]), iris[,5])

# Accuracy for the second model 
table(predict(mod2, model2_scores), iris[,5])

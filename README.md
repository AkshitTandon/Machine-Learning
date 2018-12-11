# Machine-Learning

What is a Model Parameter?
A model parameter is a configuration variable that is internal to the model and whose value can be estimated from data.
They are required by the model when making predictions.
They values define the skill of the model on your problem.
They are estimated or learned from data.
They are often not set manually by the practitioner.
They are often saved as part of the learned model.
Parameters are key to machine learning algorithms. They are the part of the model that is learned from historical training data.
In classical machine learning literature, we may think of the model as the hypothesis and the parameters as the tailoring of the hypothesis to a specific set of data.
What is a Model Hyperparameter?
A model hyperparameter is a configuration that is external to the model and whose value cannot be estimated from data.
They are often used in processes to help estimate model parameters.
They are often specified by the practitioner.
They can often be set using heuristics.
They are often tuned for a given predictive modeling problem.
We cannot know the best value for a model hyperparameter on a given problem. We may use rules of thumb, copy values used on other problems, or search for the best value by trial and error.
When a machine learning algorithm is tuned for a specific problem, such as when you are using a grid search or a random search, then you are tuning the hyperparameters of the model or order to discover the parameters of the model that result in the most skillful predictions.
Some examples of model hyperparameters include:
The learning rate for training a neural network.
The C and sigma hyperparameters for support vector machines.
The k in k-nearest neighbors.
------->REGRESSION
Regression is basically of three types: Linear, Logistic and Polynomial Regression. 

A Linear Regression model can be represented by a straight line where as a Logistic Regression uses Sigmoid function. The output of Linear regression model is the value of the variable whereas for Logistic Regression it is the probability of occurrence of event. 
A Linear Regression maps a continuous X to a continuous Y. Logistic Regression maps continuous X to binary Y. We can use Logistic Regression to predict true false from the data or categories. 

In linear regression we find a line y=m*x+c that best fits the given data points, i.e.  a line with least error. In the first iteration we predict the values and check for errors between actual value and predicted value by computing the distance between them, we keep on modifying our line till the time we get least error.
To check the goodness of check we can use R square method in Linear Regression. It is the measure of how close the data are to the fitted regression line. It is also called as coefficient of determination.     
R square is the (sum)(Y(predicted)- mean)^2 / (sum)(Y(actual)-mean)^2.  R square value is equal to 1 when actual values lies on the regression line itself. Most of the times, the smaller he value of R^2, the farther away the actual points are from the regression line. But in some fields like psychology, the R^2 values are mostly less. 

Batch Gradient Descent and Stochastic Gradient Descent:

Lets say you are about to start a business that sells t-shirts, but you are unsure what are the best measures for a medium sized one for males. Luckily you have gathered a group of men that have all stated they tend to buy medium sized t-shirts. Now you figure you're going to use a gradient descent type method t get the size just right.

Batch Gradient Descent
Tailor makes initial estimate.
Each person in the batch gets to try the t-shirt and write down feedback.
Collect and summarize all feedback.
If the feedback suggests a change, let the tailor adjust the t-shirt and go to 2.
Stochastic Gradient Descent
Tailor makes initial estimate.
A random guy (or a subset of the full group) tries the t-shirt and gives feedback.
Make a small adjustment according to feedback.
While you still have time for this, go to 2.

Highlighting the differences
Batch gradient descent needs to collect lots of feedback before making adjustments, but needs to do fewer adjustments.
Stochastic gradient descent makes many small adjustments, but spends less time collecting feedback in between.
Batch gradient descent preferable if the full population is small, stochastic gradient descent preferable if the full population is very large.
Batch gradient descent methods can be made parallel if you have access to more hardware (in this case, more tailors and materials) as you can collect all feedback in parallel.
Stochastic gradient descent does not readily lend itself to parallelization as the you need the feedback from one iteration to proceed with the next iteration.


LOGISTIC REGRESSION:
We use this when the predicted  output needs to be in the binary format like 0 &1, True or False, High or low etc . We can only have 2 output values in Logistic Regression as shown in the examples. The Sigmoid function is used to normalize any value from -infinity to +infinity  to 0 and 1. There is a threshold line that depicts the probability of one class over other like 0 over 1 or 1 over 0 etc. 

If I have two class 0 and 1. My threshold value is 0.5. So any value above 0.5 is classified as 1 and any value below 0.5 is 0. 
To form the Sigmoid curve, we need to solve equations. 



Regression Tree(Decision Tree):

Regression trees (a.k.a. decision trees) learn in a hierarchical fashion by repeatedly splitting your dataset into separate branches that maximize the information gain of each split. This branching structure allows regression trees to naturally learn non-linear relationships.
Random Forest:
Random Forest is a supervised learning algorithm. It creates a forest and makes it somehow random. The „forest“ it builds, is an ensemble of Decision Trees, most of the time trained with the “bagging” method. The general idea of the bagging method is that a combination of learning models increases the overall result.
We can use both Regression and classification using Random Forest method.
We can say that the Random Forest are collection of Decision trees with some differences. 
The hyperparameter in Random forest used are:
n_estimators: It tells number of tress to be build before taking the predication or average value.
max_features:  It is the maximum number of features random forest is allowed to try in individual tree.
min_sample_leaf: The minimum number of leafs that are required to split an internal node.
The main limitation of Random Forest is that a large number of trees can make the algorithm to slow and ineffective for real-time predictions. In general, these algorithms are fast to train, but quite slow to create predictions once they are trained.
Deep Learning:
Deep learning refers to multi-layer neural networks that can learn extremely complex patterns. They use "hidden layers" between inputs and outputs in order to model intermediary representations of the data that other algorithms cannot easily learn.
It perform very well with image, text or audio data.But it is not suitable for general purpose data as it requires large dataset to perform and train.

Nearest Mean:
Nearest neighbors algorithms are "instance-based," which means that that save each training observation. They then make predictions for new observations by searching for the most similar training observations and pooling their values.

-------> Classification:
Classification is the supervised learning task for modeling and predicting categorical variables. Examples include predicting employee churn, email spam, financial fraud, or student letter grades.

It can also have algorithms like Logistic Regression. Decision tree and deep learning in addition to the ones mentioned below.

Support Vector Machine:

Support vector machines (SVM) use a mechanism called kernels, which essentially calculate distance between two observations. The SVM algorithm then finds a decision boundary that maximizes the distance between the closest members of separate classes.
SVM with Linear kernel is same as Logistic Regression. 
SVM can handle nonlinear decision boundaries and there are many kernels to choose from. 
Naive Bayes:
Naive Bayes (NB) is a very simple algorithm based around conditional probability and counting. Essentially, your model is actually a probability table that gets updated through your training data. To predict a new observation, you'd simply "look up" the class probabilities in your "probability table" based on its feature values.It's called "naive" because its core assumption of conditional independence (i.e. all input features are independent from one another) rarely holds true in the real world.
-------->CLUSTERING
Clustering is an unsupervised learning task for finding natural groupings of observations (i.e. clusters) based on the inherent structure within your dataset. Examples include customer segmentation, grouping similar items in e-commerce, and social network analysis
K-Means Clustering
K-Means is a general purpose algorithm that makes clusters based on geometric distances (i.e. distance on a coordinate plane) between points. The clusters are grouped around centroids, causing them to be globular and have similar sizes.
It is easy to do but we have to define K initially which is difficult to estimate. 

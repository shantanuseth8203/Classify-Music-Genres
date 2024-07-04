import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

## PREPARE DATASET ##

# Import track metadata with genre labels
tracks = pd.read_csv('datasets/fma-rock-vs-hiphop.csv')
# Import track metrics with the features
metrics = pd.read_json('datasets/echonest-metrics.json', precise_float=True)

# Print datasets
print("Track metadata (tracks):\n", tracks.head())
print("\nTrack metrics (metrics):\n", metrics.head())

# Merge datasets on track_id to add the genre to the metrics dataframe
genre_metrics = pd.merge(left=tracks[['track_id', 'genre_top']], right=metrics, on='track_id')

# Inspect merged dataset
print("\nMerged dataset info (genre_metrics):")
genre_metrics.info()
print("\nMerged dataset (genre_metrics):\n", genre_metrics.head())


## EXPLORE CORRELATIONS ##

# Exclude non-numeric columns before creating the correlation matrix
numeric_metrics = genre_metrics.select_dtypes(include=['number'])

# Create a correlation matrix
corr_metrics = numeric_metrics.corr()

# Visualize the correlation matrix 
plt.figure(figsize=(12, 10))
sns.heatmap(corr_metrics, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Correlation Matrix of Track Metrics')
plt.xticks(rotation=30)
plt.show()  # No strong correlations are found


## NORMALIZE FEATURE DATA ##

# It can be particularly useful to simplify our models and use as few features as necessary 
# to achieve the best result. 
# Since we didn't find any particular strong correlations between our features,
# we can instead use a common approach to reduce the number of features called principal component analysis (PCA)

# Define features by dropping non-feature columns
features = genre_metrics.drop(['genre_top', 'track_id'], axis=1)

# Define labels
labels = genre_metrics['genre_top']

# Import the StandardScaler
scaler = StandardScaler()

# Scale the features and set the values to a new variable
scaled_train_features = scaler.fit_transform(features)


## PRINCIPAL COMPONENT ANALYSIS ON SCALED DATA ##

# Import PCA class
from sklearn.decomposition import PCA

# Get explained variance ratios from PCA using all features
pca = PCA()
pca.fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_
print("\nExplained variance ratios from PCA:\n", exp_variance)

# Plot the explained variance using a bar plot
fig, ax = plt.subplots()
ax.bar(range(pca.n_components_), exp_variance)
ax.set_xlabel('Principal Component #')
plt.title('Explained Variance per Principal Component')
plt.show()

# Unfortunately, there does not appear to be a clear elbow in this scree plot, 
# which means it is not straightforward to find the number of intrinsic dimensions using this method.
# Instead, we can also look at the cumulative explained variance plot to determine how many features are required to explain 
# about 90% of the variance. 
# Once we determine the appropriate number of components, we can perform PCA with that many components, 
# ideally reducing the dimensionality of our data.

# Calculate the cumulative explained variance
cum_exp_variance = np.cumsum(exp_variance)
print("\nCumulative explained variance:\n", cum_exp_variance)

# Plot the cumulative explained variance and draw a dashed line at 0.90.
fig, ax = plt.subplots()
ax.plot(range(len(cum_exp_variance)), cum_exp_variance)
ax.axhline(y=0.9, linestyle='--')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.show()

# Select number of components to retain based on the cumulative variance plot
n_components = 7

# Perform PCA with the chosen number of components and project data onto components
pca = PCA(n_components, random_state=10)
pca.fit(scaled_train_features)
pca_projection = pca.transform(scaled_train_features)
print("\nShape of PCA projection with {} components:".format(n_components), pca_projection.shape)


## TRAIN A DECISION TREE TO CLASSIFY GENRE ##

# Now we can use the lower dimensional PCA projection of the data to classify songs into genres.

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(pca_projection, labels, stratify=labels, random_state=10)

# Train Decision Tree
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_features, train_labels)

# Predict the labels for the test data
pred_labels_tree = tree.predict(test_features)

# Measure accuracy of the Decision Tree model
print("\nDecision Tree accuracy on test data:", accuracy_score(test_labels, pred_labels_tree))

# Count labels to inspect class imbalance
print("\nLabel counts:\n", labels.value_counts())

# The data is imbalanced, so here accuracy is not really a good metric.
# Let's look at the confusion matrix to understand misclassifications.
print("\nConfusion matrix (Decision Tree):\n", confusion_matrix(test_labels, pred_labels_tree))


## COMPARE DECISION TREE TO A LOGISTIC REGRESSION ##

# Let's test to see if logistic regression model can perform better than the decision tree.

# Train a logistic regression model and predict labels for the test set
logreg = LogisticRegression(random_state=10)
logreg.fit(train_features, train_labels)
pred_labels_logit = logreg.predict(test_features)

# Create the classification report for both models
class_rep_tree = classification_report(test_labels, pred_labels_tree)
class_rep_log = classification_report(test_labels, pred_labels_logit)

print("\nDecision Tree Classification Report:\n", class_rep_tree)
print("\nLogistic Regression Classification Report:\n", class_rep_log)


## BALANCE DATA FOR GREATER PERFORMANCE ##

# Both models perform similarly well with an avg precision of 87% each
# However, Hip-Hop songs are disproportionately misclassified as rock songs.
# That is because there are far more data points for rock than hip-hop.
# To account for this, we can balance the number of data points of the two genres.

# Separate the two genres into two datasets
hop_only = genre_metrics.loc[genre_metrics['genre_top'] == 'Hip-Hop']
rock_only = genre_metrics.loc[genre_metrics['genre_top'] == 'Rock']

# Count data points before balancing
print("\nData points before balancing (Rock, Hip-Hop):", rock_only.shape, hop_only.shape)

# Make the rock songs the same number as the hip-hop songs
rock_only = rock_only.sample(n=hop_only.shape[0])

# Count data points after balancing
print("\nData points after balancing (Rock, Hip-Hop):", rock_only.shape, hop_only.shape)

# Concatenate the balanced rock and hip-hop dataframes
rock_hop_bal = pd.concat([rock_only, hop_only])

# Create features, labels, and PCA projection for the balanced dataframe
features = rock_hop_bal.drop(['genre_top', 'track_id'], axis=1) 
labels = rock_hop_bal['genre_top']
pca_projection = pca.fit_transform(scaler.fit_transform(features))

# Redefine the train and test sets with the PCA projection from the balanced data
train_features, test_features, train_labels, test_labels = train_test_split(pca_projection,
                                                                            labels,
                                                                            stratify=labels,
                                                                            random_state=10)

## TEST IF MODEL BIAS PROBLEM IS SOLVED ##

# Train decision tree on the balanced data
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_features, train_labels)
pred_labels_tree = tree.predict(test_features)

# Train logistic regression on the balanced data
logreg = LogisticRegression(random_state=10)
logreg.fit(train_features, train_labels)
pred_labels_logit = logreg.predict(test_features)

# Compare the models
print("\nDecision Tree (Balanced) Classification Report:\n", classification_report(test_labels, pred_labels_tree))
print("\nLogistic Regression (Balanced) Classification Report:\n", classification_report(test_labels, pred_labels_logit))


## USING CROSS-VALIDATION TO EVALUATE MODELS ##

# Set up K-fold cross-validation with shuffling
kf = KFold(n_splits=10, shuffle=True, random_state=10)

# Initialize the models
tree = DecisionTreeClassifier(random_state=10)
logreg = LogisticRegression(random_state=10)

# Perform cross-validation for both models
tree_score = cross_val_score(tree, pca_projection, labels, cv=kf)
logit_score = cross_val_score(logreg, pca_projection, labels, cv=kf)

# Print the mean of each array of scores
print("\nCross Validation Score :")
print("Decision Tree:", np.mean(tree_score))
print("Logistic Regression:", np.mean(logit_score))


## We can conclude that the logistic regression model performs better at predicting song genre
## than the decision tree.

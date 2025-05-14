import pandas as pandas
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sb
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

NUM_CLASSES = 3
EPOCHS = 10000

# read data
red_data = pandas.read_csv("datasets/winequality-red.csv", sep=';').to_numpy(dtype=np.float64)
white_data = pandas.read_csv("datasets/winequality-white.csv", sep=';').to_numpy(dtype=np.float64)

df = pandas.read_csv("datasets/winequality-white.csv", sep=';')
print(df.columns)


# normalize features
red_scaler = StandardScaler()
red_data[:, :-1] = red_scaler.fit_transform(red_data[:, :-1])

white_scaler = StandardScaler()
white_data[:, :-1] = white_scaler.fit_transform(white_data[:, :-1])

# shuffle rows to prevent bias ordering
np.random.shuffle(red_data)
np.random.shuffle(white_data)

rng     = np.random.default_rng(12345)
theta   = rng.random((12,NUM_CLASSES)) #weight array. each class is a row and each feature is a column

def bucketize(y): #reduces data to 3 classes
    y = np.asarray(y, dtype=int)
    # low: 3-5, medium: 6, high: 7-9
    # np.digitize splits at 4, 6, 8 and produces 0, 1, 2, 3
    #return np.digitize(y, bins=[4,6,8], right=False)
    return np.digitize(y, bins=[5, 6], right=True)

# Red wine
red_targets_raw     = red_data[:, -1]  # last column = quality score  
print("Original Quality Score Values, # Samples/Score")
print(np.unique(red_targets_raw, return_counts=True))

red_targets         = bucketize(red_targets_raw) 
print("Bucket Quality Classes, # Samples/bucket")
print(np.unique(red_targets, return_counts=True))
hot_red_targets     = np.eye(NUM_CLASSES)[red_targets]

red_samples         = np.ones((red_data.shape[0], red_data.shape[1]))
red_samples[:, 1:]  = red_data[:, :-1]

#Splits into testing and training data
red_testing         = red_samples[:500, :]
red_training        = red_samples[500:,:]

hot_red_testing     = hot_red_targets[:500,:]
hot_red_training    = hot_red_targets[500:,:]

# White wine
white_targets_raw   = white_data[:, -1]
print("Original Quality Score Values, # Samples/Score")
print(np.unique(white_targets_raw, return_counts=True))

white_targets       = bucketize(white_targets_raw)
print("Bucket Quality Classes, # Samples/bucket")
print(np.unique(white_targets, return_counts=True))

hot_white_targets   = np.eye(NUM_CLASSES)[white_targets]

white_samples       = np.ones((white_data.shape[0], white_data.shape[1]))
white_samples[:, 1:] = white_data[:, :-1]

# splitting into testing and trainging data
white_testing       = white_samples[:1000, :]
white_training      = white_samples[1000:, :]

hot_white_testing   = hot_white_targets[:1000, :]
hot_white_training  = hot_white_targets[1000:, :]



def soft_max(weights, samples): #computes the soft max
    scores          = samples @ weights
    scores          -= scores.max(axis=1, keepdims=True) #
    e               = np.exp(scores)

    return e / e.sum(axis=1,keepdims=True)


# training loop
def regression_loop(weights, samples, targets): 
    lambda_reg = 0.1 # add regularization
    mu = 0.005 # learning rate
    n = samples.shape[0]

    for i in range(EPOCHS):
        y           = soft_max(weights, samples)
        gradient    = (samples.T @ (y - targets)) / n
        weights     -= mu*(gradient + lambda_reg * weights)

        if i % 200 == 0:
            # loss = np.mean(np.sum((targets - y)*(targets - y), axis=1)) #mean squared error
            loss = cross_entropy_loss(y, targets)
            print(f"epoch {i:4d} | loss {loss:.4f}")

    return weights

# calculate cross-entropy loss for softmax classification
def cross_entropy_loss(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

# generate nice-looking confusion matrix
def plot_confusion_matrix(cm, title):
    plot.figure(figsize=(6, 5))
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Low', 'Medium', 'High'],
               yticklabels=['Low', 'Medium', 'High'])
    plot.xlabel("Predicted Label")
    plot.ylabel("True Label")
    plot.title(title)
    plot.tight_layout()
    plot.show()



theta       = regression_loop(theta, white_training, hot_white_training)

predicted   = np.argmax(white_testing @ theta, axis = 1)

white_testing_targets = white_targets[:1000]

cm          = confusion_matrix(white_testing_targets, predicted, labels=[0,1,2])

white_accuracy    = np.mean(predicted == white_testing_targets) 

print(f"White Wine Accuracy: {white_accuracy * 100:.2f}%")
print(cm)
plot_confusion_matrix(cm, "White Wine Confusion Matrix")



theta       = rng.random((12,NUM_CLASSES))

theta       = regression_loop(theta, red_training, hot_red_training) 

predicted   = np.argmax(red_testing @ theta, axis = 1)

red_testing_targets = red_targets[:500]

cm          = confusion_matrix(red_testing_targets, predicted, labels=[0,1,2])

red_accuracy    = np.mean(predicted == red_testing_targets) 

print(f"Red Wine Accuracy: {red_accuracy * 100:.2f}%")
print(cm)
plot_confusion_matrix(cm, "Red Wine Confusion Matrix")



# ----------------- POLYNOMIAL FEATURES -------------------------

# Polynomial feature transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_white_train_poly = poly.fit_transform(white_training[:, 1:])
X_white_test_poly = poly.transform(white_testing[:, 1:])

# Train logistic regression on expanded features
clf_white_poly = LogisticRegression(solver='lbfgs', max_iter=10000)
clf_white_poly.fit(X_white_train_poly, white_targets[1000:])
predicted_white_poly = clf_white_poly.predict(X_white_test_poly)

# Evaluation
cm_white_poly = confusion_matrix(white_targets[:1000], predicted_white_poly, labels=[0, 1, 2])
acc_white_poly = accuracy_score(white_targets[:1000], predicted_white_poly)

plot_confusion_matrix(cm_white_poly, "White Wine (Polynomial Features)")
print(f"White Wine Accuracy (Polynomial LogisticRegression): {acc_white_poly * 100:.2f}%")
#print(classification_report(white_targets[:1000], predicted, target_names=["Low", "Medium", "High"]))


# Polynomial feature transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_red_train_poly = poly.fit_transform(red_training[:, 1:])
X_red_test_poly = poly.transform(red_testing[:, 1:])

# Train logistic regression on expanded features
clf_red_poly = LogisticRegression(solver='lbfgs', max_iter=10000)
clf_red_poly.fit(X_red_train_poly, red_targets[500:])
predicted_red_poly = clf_red_poly.predict(X_red_test_poly)

# Evaluation
cm_red_poly = confusion_matrix(red_targets[:500], predicted_red_poly, labels=[0, 1, 2])
acc_red_poly = accuracy_score(red_targets[:500], predicted_red_poly)

print(f"Red Wine Accuracy (Polynomial LogisticRegression): {acc_red_poly * 100:.2f}%")
plot_confusion_matrix(cm_red_poly, "Red Wine (Polynomial Features)")
print(classification_report(red_targets[:500], predicted, target_names=["Low", "Medium", "High"]))



# ----------------- RANDOM FOREST -------------------------

rf_white = RandomForestClassifier(n_estimators=100, random_state=42)
rf_white.fit(white_training[:, 1:], white_targets[1000:])
rf_white_preds = rf_white.predict(white_testing[:, 1:])
rf_white_acc = accuracy_score(white_targets[:1000], rf_white_preds)
print(f"White Wine Accuracy (Random Forest): {rf_white_acc * 100:.2f}%")
cm_rf_white = confusion_matrix(white_targets[:1000], rf_white_preds, labels=[0,1,2])
plot_confusion_matrix(cm_rf_white, "White Wine (Random Forest)")


rf_red = RandomForestClassifier(n_estimators=100, random_state=42)
rf_red.fit(red_training[:, 1:], red_targets[500:])
rf_red_preds = rf_red.predict(red_testing[:, 1:])
rf_red_acc = accuracy_score(red_targets[:500], rf_red_preds)
print(f"Red Wine Accuracy (Random Forest): {rf_red_acc * 100:.2f}%")
cm_rf_red = confusion_matrix(red_targets[:500], rf_red_preds, labels=[0,1,2])
plot_confusion_matrix(cm_rf_red, "Red Wine (Random Forest)")



# Print all averages in one place
print("\n\nTotal Accuracies:")
print(f"White Wine Accuracy: {white_accuracy * 100:.2f}%")
print(f"White Wine Accuracy (Polynomial Logistic Regression): {acc_white_poly * 100:.2f}%")
print(f"White Wine Accuracy (Random Forest): {rf_white_acc * 100:.2f}%")
print(f"Red Wine Accuracy: {red_accuracy * 100:.2f}%")
print(f"Red Wine Accuracy (Polynomial Logistic Regression): {acc_red_poly * 100:.2f}%")
print(f"Red Wine Accuracy (Random Forest): {rf_red_acc * 100:.2f}%")

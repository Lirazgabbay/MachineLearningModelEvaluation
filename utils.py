import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from LogisticRegressionGD import LogisticRegressionGD
from NaiveBayesGaussian import NaiveBayesGaussian

def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
    plt.show()


def pearson_correlation( x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    numerator_sum = np.sum((x - mean_x) * (y - mean_y))

    denominator_sum = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))

    if denominator_sum == 0:
      return 0

    r = numerator_sum / denominator_sum  
  
    return r

def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    best_features = []

    X =  X.select_dtypes(include='number')
    lenloop= X.shape[1]
    pearson_correlation_list = []
    for col in range(lenloop):
        try:
            feature = X.iloc[:, col]
            feature_correlation = pearson_correlation(feature, y)
            pearson_correlation_list.append((col, abs(feature_correlation)))
        except ValueError:
            continue  
    pearson_correlation_list.sort(key=lambda x: x[1], reverse=True)
    best_features = [X.columns.values[index] for index, _ in pearson_correlation_list[:n_features]]
   
    return best_features

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    np.random.seed(random_state)

   
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]


    X_rows = len(X_shuffled)
    rows_in_fold = X_rows//folds

    accuracies = []
    for fold in range(folds):
      start_index_test = fold * rows_in_fold
      end_index_test = (fold + 1) * rows_in_fold if fold != folds - 1 else X_rows

      X_test = X_shuffled[start_index_test:end_index_test]
      y_test = y_shuffled[start_index_test:end_index_test]

      training_set_indice = np.concatenate((np.arange(0, start_index_test), np.arange(end_index_test, X_rows)))
      X_train = X_shuffled[training_set_indice]
      y_train = y_shuffled[training_set_indice]

      algo.fit(X_train, y_train)

      y_pred = algo.predict(X_test)

      accuracy = np.mean(y_pred == y_test)
      accuracies.append(accuracy)

    cv_accuracy = np.mean(accuracies)
    return cv_accuracy

    
def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None
    lor_reg = LogisticRegressionGD(eta = best_eta, eps = best_eps)
    lor_reg.fit(x_train, y_train)

    lor_train_acc = accuracy(lor_reg.predict(x_train), y_train)
    lor_test_acc = accuracy(lor_reg.predict(x_test), y_test)

    bayes_model = NaiveBayesGaussian(k=k)
    bayes_model.fit(x_train, y_train)

    bayes_train_acc = accuracy(bayes_model.predict(x_train), y_train)
    bayes_test_acc = accuracy(bayes_model.predict(x_test), y_test)

    plot_decision_regions(x_train, y_train, lor_reg,  title="Decision boundaries for All samples for Logistic Regression model" )
    plot_decision_regions(x_train, y_train, bayes_model, title="Decision boundaries for All samples for Naive Bayes Gaussian model" )

    plt.title("Cost vs Iteration number on All samples Logistic Regression")
    plt.plot(np.arange(len(lor_reg.Js)), lor_reg.Js)
    plt.xlabel('Iterations')
    plt.ylabel('cost')
    plt.show()

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def accuracy(X,y):
        length = len(y)
        count = 0
        for i in range(length):
            if y[i] == (X[i]):
                count += 1
                
        return count/length

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    
    # Generate dataset A
    dataset_a_means = [[-3, -3, -3], [0, 0, 0], [3, 3, 3], [9, 9, 9]]
    # covariance matrix is diagonal with zeros off the diagonal
    dataset_a_cov = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    dataset_a_labels = [0, 1, 0, 1]

    dataset_a_features, dataset_a_labels = generate_data(2500, dataset_a_means, dataset_a_cov, dataset_a_labels)

    # Generate dataset B
    dataset_b_means = [[0, 5, 0], [0, 7, 0]]
    #covariance matrix has non-zero values off the diagonal, indicating that there are strong correlations between features. 
    dataset_b_cov = np.array([[4, 4, 4], [4, 4, 4], [4, 4, 4]])
    dataset_b_labels = [0, 1]

    dataset_b_features, dataset_b_labels = generate_data(2500, dataset_b_means, dataset_b_cov, dataset_b_labels)

    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }

def generate_data(num_instances, means, cov, labels):
    dataset_features = np.empty((num_instances, 3))
    dataset_labels = np.empty((num_instances))
    gaussian_size = num_instances // len(means)

    for i, (mean, label) in enumerate(zip(means, labels)):
        points = np.random.multivariate_normal(mean, cov, gaussian_size)
        dataset_features[i * gaussian_size: (i + 1) * gaussian_size] = points
        dataset_labels[i * gaussian_size: (i + 1) * gaussian_size] = np.full(gaussian_size, label)
    
    return dataset_features, dataset_labels

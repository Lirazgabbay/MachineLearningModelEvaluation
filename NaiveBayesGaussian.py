import numpy as np
from EM import EM
from math_utils import gmm_pdf


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.gaussians = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        self.priors = {}
        self.gaussians = {}
        unique_labels = np.unique(y)  
        total_samples = len(y)  

        for class_label in unique_labels:
            class_samples = len(y[y == class_label])
            self.priors[class_label] = class_samples / total_samples

            self.gaussians[class_label] = {}
            for feature in range(X.shape[1]):
              self.gaussians[class_label][feature] = EM(k = self.k) # Initialize EM for the current feature

        for class_label in self.gaussians.keys():
            for feature in self.gaussians[class_label].keys():
                self.gaussians[class_label][feature].fit(X[y == class_label][:, feature].reshape(-1, 1))


    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        preds = []
        for instance in X:
            max_posterior = float('-inf') 
            best_class_label = None  
            
            for class_Label in self.priors.keys():
                posterior = self.priors[class_Label] * self.calc_likelihood(instance, class_Label)
                
                if posterior > max_posterior:
                    max_posterior = posterior 
                    best_class_label = class_Label  
            
            preds.append(best_class_label)
        preds = np.array(preds).reshape(-1, 1)
     
        return preds
    
    # Calculate the likelihood according to the formula.
    def calc_likelihood(self, instance, class_label):
        likelihood = 1
        for feature in range(instance.shape[0]):
            weights, mus, sigmas = self.gaussians[class_label][feature].get_dist_params()
            gmm = gmm_pdf(instance[feature], weights, mus, sigmas)
            likelihood *= gmm
        return likelihood
import numpy as np


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        self.theta = None


        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        np.random.seed(self.random_state)

        X = self._apply_bias_trick(X)
        m = len(y)
        self.theta = np.random.random(X.shape[1])
        counter = 0
        while(counter < self.n_iter):
          current_cost = self.compute_cost(X,y)
          self.Js.append(current_cost)
          self.theta = self.theta - self.eta * np.dot(X.T,(self.hypothesisFunction(X)-y)) / m
          self.thetas.append(self.theta.copy())
          if(len(self.Js) > 1 and np.abs(current_cost - self.Js[-2]) < self.eps):
            break
          counter += 1


    def _apply_bias_trick(self, X):
        """
        Applies the bias trick to the input data.

        Input:
        - X: Input data (m instances over n features).

        Returns:
        - X: Input data with an additional column of ones in the
            zeroth position (m instances over n+1 features).
        """
        return np.insert(X, 0, 1, axis=1)

    
    def compute_cost(self, X, y):
      """
      Compute the cost function for logistic regression
      Parameters
      ----------
      X: Input features (numpy array)
      y: Actual output labels (numpy array)

      """
      numOfInstances = len(y)
      hypothesis_func = self.hypothesisFunction(X)
      y_ln_hypothesis = np.dot(-y , np.log(hypothesis_func))
      one_Minus_y_ln_hypothesis = np.dot((1 - y) ,np.log(1-hypothesis_func))
      result = np.subtract(y_ln_hypothesis , one_Minus_y_ln_hypothesis)
      result = np.sum(result)
      cost = result/numOfInstances
      return cost

    def hypothesisFunction(self,X):
      """
      Compute the hypothesis function for logistic regression
      Parameters
      ----------
      X: Input features (numpy array)

      """
      return 1 / (1 + np.exp(-1 * np.dot(X, self.theta)))
    
    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        
        X = self._apply_bias_trick(X)
        predictions = self.hypothesisFunction(X)
        preds = np.where(predictions > 0.5, 1, 0)
      
        return preds
    
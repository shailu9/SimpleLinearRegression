import numpy as np

class SimpleLinearRegressionWithGradientDescent:
    """A simple linear regression model that uses gradient descent for optimization.
    This class implements a simple linear regression model that learns the relationship between a single feature and a
    target variable using gradient descent. The model has two parameters: the weight (w) and the bias (b), 
    which are updated iteratively to minimize the cost function."""
    def __init__(self, learning_rate=0.0001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = 0
        self.b =0

    def fit(self,X:np.ndarray,y:np.ndarray) -> None:
        """Fit the linear regression model to the training data using gradient descent.
        This is where we train our model by adjusting the weights (w) and bias (b) to minimize the cost function.
        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.
            Returns:
            None: The method updates the model parameters (weights and bias) in place."""
        J_history = [] # not sure where to use this yet
        for i in range(self.n_iterations):
            dj_dw, dj_db = self.__compute_gradients(X,y)
            self.w -= self.learning_rate * dj_dw
            self.b -= self.learning_rate * dj_db
            if i % 100 == 0:
                cost = self.__compute_cost(X,y)
                J_history.append(cost) 
                print(f"Cost after iteration {i}: {cost}")

    def __compute_cost(self,X:np.ndarray,y:np.ndarray) -> float:
        """Calculate the cost function(loss function or error function) for linear regression.
        Loss function is a measure of how well the model's predictions match the actual target values.
        Meaning what is the distance (delta or error) between the predicted (y^ of f_wb ) and actual values (y).
        This is a private method.
        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.
            Returns:
            total_cost (float): The computed cost for the given input and target values.
            """
        m = X.shape[0]
        cost = 0

        for i in range(m):
            f_wb = self.w * X[i] + self.b
            cost += (f_wb - y[i]) ** 2
        
        total_cost = cost / (2 * m)
        print(f"Cost after iteration {0}: {total_cost}")
        return total_cost

    def __compute_gradients(self,X:np.ndarray,y:np.ndarray) -> tuple[float,float]:
        """Calculate the gradients of the cost function with respect to the weights and bias.
        This is where we compute how much we need to adjust the weights and bias to minimize the cost function.
        Basically we calculate the partial derivatives of the cost function with respect to w and b, 
        which gives us the direction and magnitude of the adjustments needed to minimize the cost.
        This is a private method.
        Args:            
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.
        Returns:
            dj_dw (float): The gradient of the cost function with respect to the weights (w).
            dj_db (float): The gradient of the cost function with respect to the bias (b)."""
        m = X.shape[0]
        dj_dw = 0
        dj_db = 0

        for i in range(m):
            f_wb = self.w * X[i] + self.b
            dj_dw_i = (f_wb - y[i]) * X[i]
            dj_db_i = f_wb - y[i]

            dj_dw += dj_dw_i
            dj_db += dj_db_i
        
        dj_dw /= m
        dj_db /= m
        return dj_dw, dj_db

    def predict(self,X:np.ndarray)-> np.ndarray:
        """Predict the target values for given input features using the learned weights and bias.
        This is where we use the trained model to make predictions on new data.
        Args:
            X (np.ndarray): The input features for which we want to make predictions.
        Returns:
            predictions (np.ndarray): The predicted target values based on the input features and learned parameters."""
        return self.w * X + self.b
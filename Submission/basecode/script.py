import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    X = np.hstack((np.ones((n_data, 1)), train_data))  # Add bias column
    weights = initialWeights.reshape((n_features + 1, 1))  # Reshape to column vector

    # Compute predictions
    theta = sigmoid(np.dot(X, weights))

    # Compute cross-entropy error
    error = -np.sum(labeli * np.log(theta) + (1 - labeli) * np.log(1 - theta)) / n_data

    # Compute gradient
    error_grad = np.dot(X.T, (theta - labeli)) / n_data

    return error, error_grad.flatten()


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    n_data = data.shape[0]
    # Add bias term
    X = np.hstack((np.ones((n_data, 1)), data))

    # Compute probabilities for each class
    probabilities = sigmoid(np.dot(X, W))

    # Assign labels based on maximum probability
    label = np.argmax(probabilities, axis=1).reshape((n_data, 1))

    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        Y: the label vector of size N x 10 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    n_class = 10
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data


    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label

def evaluate_model(model, train_data, train_label, validation_data, validation_label, test_data, test_label):
    """
    Evaluate the given SVM model on training, validation, and testing datasets.

    Inputs:
    - model: Trained SVM model
    - train_data: Training data
    - train_label: Training labels
    - validation_data: Validation data
    - validation_label: Validation labels
    - test_data: Testing data
    - test_label: Testing labels
    """
    # Compute accuracies
    train_accuracy = model.score(train_data, train_label) * 100
    validation_accuracy = model.score(validation_data, validation_label) * 100
    test_accuracy = model.score(test_data, test_label) * 100

    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Accuracy: {validation_accuracy:.2f}%")
    print(f"Testing Accuracy: {test_accuracy:.2f}%")

    # Compute and print confusion matrix for testing data
    test_predictions = model.predict(test_data)
    cm = confusion_matrix(test_label, test_predictions)

    print("\nConfusion Matrix (Testing Data):")
    print(cm)

    # Display confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
    plt.title("Confusion Matrix (Testing Data)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    # Print classification report
    print("\nClassification Report (Testing Data):")
    print(classification_report(test_label, test_predictions))

# Visualize and analyze results

def visualize_and_analyze_with_heatmap(train_predictions, train_label, test_predictions, test_label, validation_predictions, validation_label, W):
    """
    Records and visualizes errors, computes accuracy, and generates heatmaps and classification reports.
    """

    # Classification report
    print("\nClassification Report (Training Data):")
    print(classification_report(train_label, train_predictions))

    print("\nClassification Report (Testing Data):")
    print(classification_report(test_label, test_predictions))

    print("\nClassification Report (Validation Data):")
    print(classification_report(validation_label, validation_predictions))


    # Confusion matrix
    train_cm = confusion_matrix(train_label, train_predictions)
    test_cm = confusion_matrix(test_label, test_predictions)
    validation_cm = confusion_matrix(validation_label, validation_predictions)

    # Heatmap for training confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(train_cm, annot=True, fmt="d", cmap="Blues", xticklabels=[str(i) for i in range(10)],
                yticklabels=[str(i) for i in range(10)])
    plt.title("Confusion Matrix (Training Data)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    # Heatmap for testing confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues", xticklabels=[str(i) for i in range(10)],
                yticklabels=[str(i) for i in range(10)])
    plt.title("Confusion Matrix (Testing Data)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()  

    # Heatmap for validation confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(validation_cm, annot=True, fmt="d", cmap="Blues", xticklabels=[str(i) for i in range(10)],
                yticklabels=[str(i) for i in range(10)])
    plt.title("Confusion Matrix (Validation Data)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()      

if __name__ == "__main__":

    """
    Script for Logistic Regression
    """
    # Store total errors during training
    total_errors_train = []
    total_errors_test = []

    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
    
    # number of classes
    n_class = 10
    
    # number of training samples
    n_train = train_data.shape[0]
    
    # number of features
    n_feature = train_data.shape[1]
    
    Y = np.zeros((n_train, n_class))
    for i in range(n_class):
        Y[:, i] = (train_label == i).astype(int).ravel()
    
    # Logistic Regression with Gradient Descent
    W = np.zeros((n_feature + 1, n_class))
    initialWeights = np.zeros(n_feature + 1)
    opts = {'maxiter': 100}
    for i in range(n_class):
        labeli = Y[:, i].reshape(n_train, 1)
        args = (train_data, labeli)
        nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        W[:, i] = nn_params.x.reshape((n_feature + 1,))
        # Compute and store the training error for this class
        train_error, _ = blrObjFunction(nn_params.x, train_data, labeli)
        total_errors_train.append(train_error * n_train)  # Multiply by n_train to get total error

        # Compute and store the test error for this class
        binary_test_labels = (test_label == i).astype(int)
        test_error, _ = blrObjFunction(nn_params.x, test_data, binary_test_labels)
        total_errors_test.append(test_error * test_data.shape[0])  # Multiply by n_test to get total error

        # Total errors are now stored in total_errors_train and total_errors_test
        
    categories = [str(i) for i in range(10)]
    plt.figure(figsize=(10, 6))
    plt.bar(categories, total_errors_train, alpha=0.7, label="Training Error")
    plt.bar(categories, total_errors_test, alpha=0.7, label="Testing Error")
    plt.xlabel("Category")
    plt.ylabel("Total Cross-Entropy Error")
    plt.title("Total Error by Category")
    plt.legend()
    plt.show()

    print("\nTotal Errors (Training Data):")
    for i, error in enumerate(total_errors_train):
        print(f"Category {i}: {error:.4f}")

    print("\nTotal Errors (Testing Data):")
    for i, error in enumerate(total_errors_test):
        print(f"Category {i}: {error:.4f}")

    # Find the accuracy on Training Dataset
    predicted_label_train = blrPredict(W, train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_train == train_label).astype(float))) + '%')
    
    # Find the accuracy on Validation Dataset
    predicted_label_validation = blrPredict(W, validation_data)
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_validation == validation_label).astype(float))) + '%')
    
    # Find the accuracy on Testing Dataset
    predicted_label_test = blrPredict(W, test_data)
    print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_test == test_label).astype(float))) + '%')

    visualize_and_analyze_with_heatmap(predicted_label_train, train_label, predicted_label_test, test_label, predicted_label_validation, validation_label, W)

    
    """
    Script for Support Vector Machine
    """
    
    # # Convert labels to 1D arrays for sklearn compatibility
    train_label = train_label.ravel()
    validation_label = validation_label.ravel()
    test_label = test_label.ravel()

    # Randomly sample 10,000 training samples (to address SVM scaling issues)
    random_indices = np.random.choice(train_data.shape[0], 10000, replace=False)
    train_data_sampled = train_data[random_indices]
    train_label_sampled = train_label[random_indices]

    # 1. Linear Kernel
    print("\nTraining SVM with Linear Kernel...")
    linear_svm = svm.SVC(kernel="linear")
    linear_svm.fit(train_data_sampled, train_label_sampled)

    # Evaluate on training, validation, and testing data
    print("\nResults for Linear Kernel:")
    evaluate_model(linear_svm, train_data, train_label, validation_data, validation_label, test_data, test_label)

    # 2. RBF Kernel with gamma=1
    print("\nTraining SVM with RBF Kernel (gamma=1)...")
    rbf_svm_gamma_1 = svm.SVC(kernel="rbf", gamma=1.0)
    rbf_svm_gamma_1.fit(train_data_sampled, train_label_sampled)

    # Evaluate on training, validation, and testing data
    print("\nResults for RBF Kernel (gamma=1):")
    evaluate_model(rbf_svm_gamma_1, train_data, train_label, validation_data, validation_label, test_data, test_label)

    # 3. RBF Kernel with default gamma
    print("\nTraining SVM with RBF Kernel (default gamma)...")
    rbf_svm_default_gamma = svm.SVC(kernel="rbf", gamma="scale")
    rbf_svm_default_gamma.fit(train_data_sampled, train_label_sampled)

    # Evaluate on training, validation, and testing data
    print("\nResults for RBF Kernel (default gamma):")
    evaluate_model(
        rbf_svm_default_gamma,
        train_data,
        train_label,
        validation_data,
        validation_label,
        test_data,
        test_label,
    )

    # 4. RBF Kernel with default gamma and varying C
    print("\nTraining SVM with RBF Kernel (default gamma) and varying C...")
    C_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    training_accuracies = []
    validation_accuracies = []
    testing_accuracies = []

    for C in C_values:
        print(f"\nTraining with C = {C}...")
        rbf_svm_varying_C = svm.SVC(kernel="rbf", gamma="scale", C=C)
        rbf_svm_varying_C.fit(train_data_sampled, train_label_sampled)

        # Evaluate on training, validation, and testing data
        train_acc = rbf_svm_varying_C.score(train_data, train_label) * 100
        val_acc = rbf_svm_varying_C.score(validation_data, validation_label) * 100
        test_acc = rbf_svm_varying_C.score(test_data, test_label) * 100

        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"Testing Accuracy: {test_acc:.2f}%")

        training_accuracies.append(train_acc)
        validation_accuracies.append(val_acc)
        testing_accuracies.append(test_acc)

    # Plot accuracy vs. C values
    plt.figure(figsize=(10, 6))
    plt.plot(C_values, training_accuracies, label="Training Accuracy")
    plt.plot(C_values, validation_accuracies, label="Validation Accuracy")
    plt.plot(C_values, testing_accuracies, label="Testing Accuracy")
    plt.xlabel("C Values")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs C (RBF Kernel with Default Gamma)")
    plt.legend()
    plt.grid()
    plt.show()

    # Running on complete set

    rbf_model_full = svm.SVC(kernel = 'rbf', gamma = 'scale', C = 10)
    rbf_model_full.fit(train_data, train_label.ravel())

    print('----------\n RBF with FULL training set with best C : \n------------')
    print('\n Training Accuracy:' + str(100 * rbf_model_full.score(train_data, train_label)) + '%')
    print('\n Validation Accuracy:' + str(100 * rbf_model_full.score(validation_data, validation_label)) + '%')
    print('\n Testing Accuracy:' + str(100 * rbf_model_full.score(test_data, test_label)) + '%')
    
    """
    Script for Extra Credit Part
    """
    # FOR EXTRA CREDIT ONLY
    W_b = np.zeros((n_feature + 1, n_class))
    initialWeights_b = np.zeros((n_feature + 1, n_class))
    opts_b = {'maxiter': 100}
    
    args_b = (train_data, Y)
    nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
    W_b = nn_params.x.reshape((n_feature + 1, n_class))
    
    # Find the accuracy on Training Dataset
    predicted_label_b = mlrPredict(W_b, train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')
    
    # Find the accuracy on Validation Dataset
    predicted_label_b = mlrPredict(W_b, validation_data)
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')
    
    # Find the accuracy on Testing Dataset
    predicted_label_b = mlrPredict(W_b, test_data)
    print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
    
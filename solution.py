import numpy as np
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, eta, C, niter, batch_size, verbose):
        self.eta = eta
        self.C = C
        self.niter = niter
        self.batch_size = batch_size
        self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        """
        y : numpy array of shape (n,)
        m : int (num_classes)
        returns : numpy array of shape (n, m)
        """
        array_1 = np.zeros(shape=(len(y), m))
        for label in range(m):
            array_1[:,label] = np.where(y==label,1,-1)
        
        return array_1

    def compute_loss(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : float
        """
        m = y.shape[1]
        loss = 0
        reg = np.sum(np.sum(self.w ** 2, axis=1),axis=0)
        for j in range(m):
            predict = np.dot(x,self.w[:,j])
            loss += np.maximum(0, 2 - predict * y[:,j]) ** 2
        return np.mean(loss) + (self.C/2.) * reg
    
    def compute_gradient(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : numpy array of shape (num_features, num_classes)
        """
        m = y.shape[1]
        grad = np.zeros(shape=(x.shape[1],m))
        for j in range(m):
            for i in range(x.shape[0]):
                predict = np.dot(x[i,:],self.w[:,j])
                loss = np.maximum(0, 2 - predict * y[i,j])
                grad[:,j] += loss * x[i,:] * y[i,j]
            
        return -2*grad / x.shape[0] + self.C * self.w

    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
        x : numpy array of shape (num_examples_to_infer, num_features)
        returns : numpy array of shape (num_examples_to_infer, num_classes)
        """
        m = self.w.shape[1]
        inferred = np.dot(x,self.w)
        inferred = inferred.argmax(axis=1)
        return self.make_one_versus_all_labels(inferred, m)
    
    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (num_examples, num_classes)
        y : numpy array of shape (num_examples, num_classes)
        returns : float
        """
        return np.mean([all(x) for x in y_inferred == y])

    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, num_features)
        y_train : numpy array of shape (number of training examples, num_classes)
        x_test : numpy array of shape (number of training examples, nujm_features)
        y_test : numpy array of shape (number of training examples, num_classes)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x, y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train, y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test, y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print(f"Iteration {iteration} | Train loss {train_loss:.04f} | Train acc {train_accuracy:.04f} |"
                      f" Test loss {test_loss:.04f} | Test acc {test_accuracy:.04f}")

            # Record losses, accs
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            test_losses.append(test_loss)
            test_accs.append(test_accuracy)

        return train_losses, train_accs, test_losses, test_accs


# DO NOT MODIFY THIS FUNCTION
def load_data():
    
    # Load the data files
    print("Loading data...")
    x_train = np.load("data/train_features_cifar100_reduced.npz")["train_data"]
    x_test = np.load("data/test_features_cifar100_reduced.npz")["test_data"]
    y_train = np.load("data/train_labels_cifar100_reduced.npz")["train_label"]
    y_test = np.load("data/test_labels_cifar100_reduced.npz")["test_label"]

    # normalize the data
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # add implicit bias column
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)
    
    print("data loaded successfully")
    return x_train, y_train, x_test, y_test

def train_and_test(x_train, y_train, x_test, y_test):
    
    c_val = [0.1,1,30]
    ylab = ['train loss','train accuracy','test loss','test accuracy']
    logs = []
    
    for val in c_val:
        print("Fitting the model...")
        svm = SVM(eta=0.0001, C=val, niter=50, batch_size=5000, verbose=True)
        logs.append(svm.fit(x_train, y_train, x_test, y_test))    
    
    for i in range(4):
        plt.figure()
        plt.plot(np.arange(50), logs[0][i])
        plt.plot(np.arange(50), logs[1][i])
        plt.plot(np.arange(50), logs[2][i])
        plt.xlabel('iteration')
        plt.ylabel(ylab[i])
        plt.legend('C = ' + str(c_val))

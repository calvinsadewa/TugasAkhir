"""network_direct_reccurent.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)

def sign(num):
    if (num > 0): 
        return 1
    elif (num < 0):
        return -1
    else:
        return 0

class WealthUtility(object):
    action = 0
    
    def get_ret(self,price_ret,action_before,action_now,cost):
        import math
        transaction_cost = cost * math.fabs(action_now - action_before)
        return price_ret*action_before - transaction_cost

    def add_data(self,price_ret,cost,output):
        self.price_ret = price_ret
        self.cost = cost
        self.prev_action = self.action
        self.action = self.output_to_action(output)
        self.ret = self.get_ret(self.price_ret,self.prev_action,self.action,cost)
        
    def output_to_action(self,output):
        return output * 2 - 1
        
    def coeff(self):
        first = self.deriv_util_to_return() * self.deriv_return_to_action() * self.deriv_action_to_output()
        second = self.deriv_util_to_return() * self.deriv_return_to_prev_action() * self.deriv_action_to_output()
        return (first,second)
    
    def deriv_action_to_output(self):
        return 2
        
    def deriv_util_to_return (self):
        return 1
        
    def deriv_return_to_action(self):
        return -self.cost*sign(self.action - self.prev_action)
        
    def deriv_return_to_prev_action(self):
        return self.price_ret + self.cost*sign(self.action - self.prev_action)
    
    def get_util(ret):
        return ret


#### Main Recurrent Network class
class Network(object):

    def __init__(self, sizes, utility = WealthUtility(), dropout = 0.5):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.sizes[0] += 1
        self.prev_result = 0
        self.default_weight_initializer()
        self.utility = utility
        self.prev_nabla_b = [np.zeros(b.shape) for b in self.biases]
        self.prev_nabla_w = [np.zeros(w.shape) for w in self.weights]
        self.dropout = dropout

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        a = np.reshape([self.prev_result] + a,(len(a)+1,1))
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def online_update(self, price_ret, cost, x, eta, lmbda):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        output = self.feedforward(x)
        self.utility.add_data(price_ret,cost,output[0][0])
        
        delta_nabla_b, delta_nabla_w = self.backpropthroughtime(x)
        first,second = self.utility.coeff()
        
        nabla_b = [db*first + pb*second for db, pb in zip(delta_nabla_b,self.prev_nabla_b)]
        nabla_w = [dw*first + pw*second for dw, pw in zip(delta_nabla_w,self.prev_nabla_w)]
        
        self.prev_nabla_b = delta_nabla_b
        self.prev_nabla_w = delta_nabla_w
        self.weights = [(1-eta*(lmbda))*w+(eta)*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b+(eta)*nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.prev_result = output[0][0]

    def backpropthroughtime(self, x):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = [self.prev_result] + x
        activation = np.reshape(activation,(len(activation),1))
        activations = [activation] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            
            mask = (np.random.rand(*z.shape) < self.dropout) / self.dropout # inverse dropout mask.
            z*= mask
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        self.prev_result = activation[0][0]
        delta = np.ones(activation.shape)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        delta = np.dot(self.weights[0].transpose(), delta)
        for l in xrange(0,self.num_layers-1):
            nabla_b[l] += delta[0][0] * self.prev_nabla_b[l]
            nabla_w[l] += delta[0][0] * self.prev_nabla_w[l]
        return (nabla_b, nabla_w)

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def generate_sin_price():
    import math
    return (math.sin(x) + 2 for x in xrange(0,10000))
    

# Using the generator pattern (an iterable)
class sin_price(object):
    def __init__(self,base_price = 100,n = 10000, a = 20, f = 0.1):
        self.base_price = base_price
        self.n = n
        self.cur_n = 0
        self.a = a
        self.f = f

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        import math
        if (self.cur_n < self.n):
            price = self.base_price + math.sin(self.cur_n*self.f)*self.a
            self.cur_n += 1
            return price
        else:
            raise StopIteration()

# Using the generator pattern (an iterable)
class random_prices(object):
    def __init__(self,base_price = 100,n = 10000):
        self.price = base_price
        self.n = n
        self.cur_n = 0

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        if (self.cur_n < self.n):
            price = self.price + random.random() - 0.5
            self.price = price
            self.cur_n += 1
            return price
        else:
            raise StopIteration()

def test_moody(prices):
    import copy
    cost = 0.01
    price_lookback = 5
    train_batch = 50
    prev_price = prices[:price_lookback]
    sizes = [price_lookback,10,7,5,3,1]
    network = Network(sizes)
    utility = network.utility
    eta = 0.01
    lmbda = 0.01
    cum_ret = 1
    train = 5
    stop = 10000
    t = 0
    trades = []
    returns = []
    train_list = []
    for price in prices[price_lookback:]:
        do_trade = t > train and t < stop
        
        price_ret = (price - prev_price[0])/ prev_price[0]
        
        train_list = [(price_ret,copy.copy(prev_price))] + train_list[0:train_batch]
        
        for p_ret,p_p in train_list:
            network.online_update(p_ret,cost,p_p,eta,lmbda)
        
        prev_price = [price] + prev_price[:-1]
        
        if do_trade:
            cum_ret *=  (1 + utility.ret)
            trades.append(utility.action)
            returns.append(1 + utility.ret)
        
        t += 1
        
    import matplotlib.pyplot as plt
    plt.plot(trades)
    plt.show()
    plt.plot(returns)
    plt.show()
    return cum_ret
    
def test(prices = random_prices()):
    import matplotlib.pyplot as plt
    prices = list(prices)
    plt.plot(prices)
    plt.show()
    rets = []
    for i in xrange(0,100):
        rets.append(test_moody(prices))
        print rets[-1]
    print max(rets)
    print min(rets)
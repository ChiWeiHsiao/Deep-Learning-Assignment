__author__ = 'ChiWeiHsiao'

import scipy.io as sio
import numpy as np
import random

class NeuralNetwork():
    def __init__(self, train_x, train_label, h_nodes1, h_nodes2, batch_size=20, learning_rate=0.05, learn_decay='False', n_hidden_layers=2, epoches=100, outfile='try.txt'):
        self.Xs, self.Labels =  train_x, train_label # Xs: a list of 45000 * np_arrays with shape=(1,784)
        # Parameters
        self.nh1 = h_nodes1
        self.nh2 = h_nodes2
        self.ni = self.Xs[0].shape[0] #number of input nodes, 784
        self.no = self.Labels[0].shape[0] #number of output nodes, 10
        self.hlayers = n_hidden_layers
        self.layers = n_hidden_layers+2 #hidden, plus input and output
        self.shape = self.get_shape()
        print('self.shape', self.shape)
        self.batch_size = batch_size
        self.init_learning_rate = learning_rate
        self.learn_decay = learn_decay
        self.learning_rate = learning_rate
        self.epoches = epoches
        # Random initialized weight
        self.weight = [] 
        self.bias = []
        self.initialize_weight()
        print('NN Initialize...finish!')
        # Statistics
        self.train_miss_statistics = []
        self.train_cost_statistics = []
        self.test_miss_statistics = []
        self.test_cost_statistics = []

        # Print Information
        self.filename = outfile
        print(outfile)
        print('number of hidden layers: '+str(n_hidden_layers))
        print('hidden nodes 1: '+str(h_nodes1))
        print('hidden nodes 2: '+str(h_nodes2))
        print('initial learning rate: '+str(learning_rate))
        print('learning rate decay?: '+str(learn_decay))
        print('batch size: '+str(batch_size))
        print('epoches: '+str(epoches))
        print('==================================')

        with open(self.filename, 'w') as fp:
            fp.write('number of hidden layers: '+str(n_hidden_layers)+'\n')
            fp.write('hidden nodes 1: '+str(h_nodes1)+'\n')
            fp.write('hidden nodes 2: '+str(h_nodes2)+'\n')
            fp.write('initial learning rate (step decay): '+str(learning_rate)+'\n')
            fp.write('batch size: '+str(batch_size)+'\n')
            fp.write('epoches: '+str(epoches)+'\n')

    def initialize_weight(self):
        # W(l) = (s_l, s_l+1)
        self.weight.append(self.draw_normal((self.ni, self.nh1)))
        self.weight.append( self.draw_normal( (self.nh1, self.nh2) ) )
        self.weight.append( self.draw_normal( (self.nh2, self.no) ) )
        self.bias.append(self.draw_normal(self.nh1))
        self.bias.append(self.draw_normal(self.nh2))
        self.bias.append(self.draw_normal(self.no))

    def forward_propagate(self, X):
        a = X #input vector (1, 748)
        A = []
        A.append(a)
        z = []
        for l in range(self.hlayers+1):
            z = np.dot(a, self.weight[l])
            z = np.add(z, self.bias[l])
            if l == self.hlayers:
                a = self.softmax(z)
            else:
                a = self.sigmoid(z) #(10,)
            A.append(a)
        return A #return the ouput value of each node in each layer 

    def cost_function(self, Target, Output):
        #of a single example
        #Output: probability vector
        return -(Target*np.log(Output))

    def train(self):
        #Parameters
        m = len(self.Xs)
        batch_size = self.batch_size
        epoches = self.epoches
        last_batch = m - batch_size + 1
        #Statistics
        total_error = [] #cost function
        miss_classify = []

        m = self.get_shape_of_weight()
        v = self.get_shape_of_weight()
        for epoche in range(0, epoches):
            if self.learn_decay:
                self.learning_rate = self.init_learning_rate / (1+epoche*0.2) #1/t decay
            print('Epoche_',epoche,':  ')
            with open(filename, 'a') as fp:
                fp.write('Epoche_'+str(epoche)+':\n')
            self.shuffle_data() #shuffle before each epoche
            # mini-batch
            for batch in range(0, last_batch, batch_size):
                start_pattern = batch
                end_pattern = start_pattern + batch_size
                Gradient = self.get_shape_of_weight()
                Gradient_bias = self.get_shape_of_bias()
                for p in range(start_pattern, end_pattern):
                    # forward propagation
                    A = self.forward_propagate(self.Xs[p])
                    error = self.cost_function(self.Labels[p], A[-1]) / batch_size
                    delta = self.get_shape_of_node()    # delta[l][i] = (10,) #array
                    delta[self.layers-1] = A[self.layers-1] - self.Labels[p] #softmax derivative = y-t
                    for l in range(self.layers-2, 0,-1):
                        # TODO: Vectorization this loop
                        for i in range(self.shape[l]):
                            weighted_sum = np.dot(self.weight[l][i], delta[l+1]) #sum over next layer, j
                            delta[l][i] = weighted_sum * A[l][i] * (1-A[l][i]) #f'(z) #delta[l][i], scalar
                    for l in range(0, self.layers-1):
                        Gradient[l] = Gradient[l] + np.dot(A[l][:, np.newaxis], delta[l+1][np.newaxis])
                        Gradient_bias[l] = Gradient_bias[l] + delta[l+1] 
                # update weight and bias once a batch
                for l in range(self.layers-1):
                    D = Gradient[l] / batch_size
                    D_bias = Gradient_bias[l]/ batch_size
                    # Adaptive Moment Estimation
                    #beta1, beta2, eps, self.learning_rate = 0.9, 0.999, 0.00000001, 0.001
                    beta1, beta2, eps = 0.9, 0.999, 0.00000001
                    m[l] = beta1*m[l] + (1-beta1)*D
                    v[l] = beta2*v[l] + (1-beta2)*(D**2)
                    #x += - learning_rate * m / (np.sqrt(v) + eps)
                    #self.weight[l] -= self.learning_rate * D
                    self.weight[l] -= self.learning_rate * m[l] / (np.sqrt(v[l]) + eps)
                    self.bias[l] -= self.learning_rate * D_bias
            # Learning Curve
            train_error_this_epoche = self.error_rate(train_x, train_label)
            test_error_this_epoche = self.error_rate(test_x, test_label)
            print('train miss in this epoche: ', train_error_this_epoche['miss'])
            print('train cost in this epoche: ', train_error_this_epoche['cost'])
            print('test miss in this epoche: ', test_error_this_epoche['miss'])
            print('test cost in this epoche: ', test_error_this_epoche['cost'])
            self.train_miss_statistics.append(train_error_this_epoche['miss'])
            self.train_cost_statistics.append(train_error_this_epoche['cost'])
            self.test_miss_statistics.append(test_error_this_epoche['miss'])
            self.test_cost_statistics.append(test_error_this_epoche['cost'])
            with open(filename, 'a') as fp:
                fp.write('train miss in this epoche: '+str(train_error_this_epoche['miss'])+'\n')
                fp.write('train cost in this epoche: '+str(train_error_this_epoche['cost'])+'\n')
                fp.write('test miss in this epoche: '+str(test_error_this_epoche['miss'])+'\n')
                fp.write('test cost in this epoche: '+str(test_error_this_epoche['cost'])+'\n')

    def predict(self, X):
        #forward_propagation
        a = X   #(1,748)
        z = []
        for l in range(self.hlayers+1):
            z = np.dot(a, self.weight[l])
            z = np.add(z, self.bias[l])
            if l == self.hlayers:
                a = self.softmax(z)
            else:
                a = self.sigmoid(z)
        #print('probability vector: ',a)
        #print('Forward, predict: ', predict)
        return a

    def error_rate(self, inputs, targets):
        m = len(inputs)
        miss, cost = 0, 0
        for i in range(m):
            output = self.predict(inputs[i])
            miss = miss + ( np.argmax(output) != np.argmax(targets[i]) )
            cost += np.sum(self.cost_function(targets[i], output))
        rate = {'miss': miss/m, 'cost': cost/m}
        return  rate

    def shuffle_data(self):
        # shuffle before each epoche
        pairs = list(zip(self.Xs, self.Labels))
        random.shuffle(pairs)
        self.Xs, self.Labels = zip(*pairs)

    def sigmoid(self, z):
        return np.reciprocal( 1 + np.exp(-z))

    def sigmoid_derivative(self, s):
        #S'(t)=S(t)(1-S(t))
        return np.multiply(s, 1-s)

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))

    def draw_normal(self, size):
        mu, sigma = 0, 1 #0.01?
        return np.random.normal(mu, sigma, size)

    def get_shape(self):
        shape = []
        shape.append(self.ni)  
        shape.append(self.nh1)
        shape.append(self.nh2)
        shape.append(self.no) 
        return shape

    def get_shape_of_weight(self):
        W = []
        for l in range(self.layers-1):
            W.append(np.zeros((self.shape[l], self.shape[l+1])))
        return W
    
    def get_shape_of_node(self):
        N = []
        for l in range(self.layers):
            N.append(np.zeros(self.shape[l]))
        return N

    def get_shape_of_bias(self):
        b = []
        for l in range(1, self.layers):
            b.append(np.zeros(self.shape[l])) #? ,1
        return b


if __name__ == "__main__":
    #read from file
    mat_contents = sio.loadmat('SVHN.mat', struct_as_record=False)
    train_x = mat_contents['train_x']   #(45000, 784)
    train_label = mat_contents['train_label']   #(45000, 10)
    test_x = mat_contents['test_x'] #(15000, 784)
    test_label = mat_contents['test_label']

    #training
    filename = 'adam-100epochs.txt'
    node1, node2, batch, rate, l, e, learn_decay = 200, 100, 20, 0.01, 2, 100, False
    nn = NeuralNetwork(train_x, train_label, h_nodes1=node1, h_nodes2=node2, batch_size=batch, learning_rate=rate, learn_decay=learn_decay,  n_hidden_layers=l, epoches=e, outfile=filename)
    nn.train()

    # print out Training learning curve
    print('===============Curve===================')
    print('Train Learning Curve on train data:')
    print('miss = ',nn.train_miss_statistics)
    print('avg cost = ', nn.train_cost_statistics)
    with open(filename, 'a') as fp:
        fp.write('=============Curve=====================\n')
        fp.write('Train Learning Curve on train data:\n')
        fp.write('miss = '+str(nn.train_miss_statistics)+'\n')
        fp.write('avg cost = '+str(nn.train_cost_statistics)+'\n\n')

    # print out Testing learning curve
    print('Test Learning Curve on train data:')
    print('miss = ',nn.test_miss_statistics)
    print('avg cost = ', nn.test_cost_statistics)
    with open(filename, 'a') as fp:
        fp.write('Test Learning Curve on train data:\n')
        fp.write('miss = '+str(nn.test_miss_statistics)+'\n')
        fp.write('avg cost = '+str(nn.test_cost_statistics)+'\n\n')

    #testing
    train_error = nn.error_rate(train_x, train_label)
    train_cost = train_error['cost']
    train_miss = train_error['miss']

    test_error = nn.error_rate(test_x, test_label)
    test_cost = test_error['cost']
    test_miss = test_error['miss']

    with open(filename, 'a') as fp:
        fp.write('==================================\n')
        fp.write('Training Data:\n')
        fp.write('Miss rate = '+ str(train_miss)+'\n')
        fp.write('average cost = '+ str(train_cost)+'\n\n')
        fp.write('Testing Data:\n')
        fp.write('miss rate = '+str(test_miss)+'\n')
        fp.write('average cost = '+ str(test_cost)+'\n\n')
    print('==================================')
    print('Training Data:')
    print('Miss rate = ',train_miss)
    print('average cost = ',train_cost)
    print('Testing Data:')
    print('miss rate = ',test_miss)
    print('average cost = ',test_cost)

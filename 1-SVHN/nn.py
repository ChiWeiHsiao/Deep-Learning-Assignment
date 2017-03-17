__author__ = 'ChiWeiHsiao'

import scipy.io as sio
import numpy as np
import random

class NeuralNetwork():
    def __init__(self, train_x, train_label, h_nodes1, h_nodes2, batch_size=20, learning_rate=0.05, learn_decay='False', n_hidden_layers=2, epoches=100,, outfile='try.txt'):
        self.Xs, self.Labels =  train_x, train_label # Xs: a list of 45000 * np_arrays with shape=(1,784)

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

        self.weight = [] 
        self.bias = []
        self.initialize_weight()
        print('NN Initialize...finish!')
        
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
        return np.negative(np.multiply(Target, np.log(Output)))

    def train(self):
        #Parameters
        m = len(self.Xs)
        batch_size = self.batch_size
        epoches = self.epoches
        last_batch = m - batch_size + 1
        #Statistics
        total_error = [] #cost function
        miss_classify = []

        for epoche in range(0, epoches):
            if self.learn_decay:
                self.learning_rate = self.init_learning_rate / (1+epoche*0.2) #1/t decay
            print('Epoche_',epoche,':  ')
            self.shuffle_data() #shuffle before each epoche
            #mini-batch
            for batch in range(0, last_batch, batch_size):
                start_pattern = batch
                end_pattern = start_pattern + batch_size
                Gradient = self.get_shape_of_weight()
                Gradient_bias = self.get_shape_of_bias()
                for p in range(start_pattern, end_pattern):
                    #forward propagation
                    A = self.forward_propagate(self.Xs[p])
                    #error = self.cost_function(self.Labels[p], A[-1]) / batch_size
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
                #update weight and bias once a batch
                for l in range(self.layers-1):
                    D = Gradient[l] / batch_size
                    D_bias = Gradient_bias[l]/ batch_size
                    self.weight[l] = self.weight[l] - self.learning_rate * D
                    self.bias[l] =  self.bias[l] - self.learning_rate * D_bias
            miss_this_epoche = self.miss_classify_rate(train_x, train_label)
            print('miss in this epoche: ', miss_this_epoche)
            with open(filename, 'a') as fp:
                fp.write('miss in this epoche: '+str(miss_this_epoche)+'\n')

    def train_regularization(self):
        labda = 0.01 
        #weight decay

    def train_ADAM(self):
        #Adaptive Moment Estimation
        1
    
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

    def miss_classify_rate(self, inputs, targets):
        error = 0
        m = len(inputs)
        for i in range(m):
            predict = np.argmax(self.predict(inputs[i]))
            error = error + (predict!=np.argmax(targets[i]))
            #print('miss classifications: ', error)
        return error / m

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
    filename = 'r1.txt'
    node1, node2, batch, rate, l, e, learn_decay = 200, 100, 20, 0.3, 2, 50, True
    
    nn = NeuralNetwork(train_x, train_label, h_nodes1=node1, h_nodes2=node2, batch_size=batch, learning_rate=rate, learn_decay=learn_decay,  n_hidden_layers=l, epoches=e, plot=False, outfile=filename)
    print('before train, cost function: ',nn.cost_function( train_label[0], nn.predict(train_x[0])))
    nn.train()
    print('after train, cost function: ',nn.cost_function( train_label[0], nn.predict(train_x[0])))

    #testing
    miss_rate_train = nn.miss_classify_rate(train_x, train_label)
    miss_rate_test = nn.miss_classify_rate(test_x, test_label)
    with open(filename, 'a') as fp:
        fp.write('Missclassification rate on Training Data = '+ str(miss_rate_train)+'\n')
        fp.write('Missclassification rate on Testing Data = '+str(miss_rate_test)+'\n')
    print('Missclassification rate on Training Data = ', miss_rate_train)
    print('Missclassification rate on Testing Data = ', miss_rate_test)

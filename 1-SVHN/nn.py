import scipy.io as sio
import numpy as np
import random
#parameter
LAYERS = 2
LEARN_RATE = 0.001
ITERATIONS = 1000
BATCH_SIZE = 100

class NeuralNetwork():
    def __init__(self, train_x, train_label, n_hidden_nodes):
        self.Xs, self.Labels =  train_x, train_label # Xs: a list of 45000 * np_arrays with shape=(1,784)

        self.nh = n_hidden_nodes
        self.ni = self.Xs[0].shape[1] #number of input nodes, 784
        self.no = self.Labels[0].shape[1] #number of output nodes, 10

        self.weight = []    #list of multiple-dimention nd_arrays
        self.bias = []  #(3,1)
        self.initialize_weight()
        self.shuffle_data()
        print('NN Initialize...finish!')

    def initialize_weight(self):
        self.weight.append(self.draw_normal((self.nh, self.ni))) # theta_1 = self.nh*784
        print('weight:',self.weight[0].shape)
        for i in range(LAYERS-1):
            self.weight.append( self.draw_normal( (self.nh, self.nh) ) )
            print('weight_',i+1,':',self.weight[i+1].shape)
        self.weight.append( self.draw_normal( (self.no, self.nh) ) ) #theta_L = 10*self.nh
        print('weight_out:',self.weight[-1].shape)
        self.bias = self.draw_normal((LAYERS+1, 1))
        print('bias:',self.bias.shape)
        print('==================================')

    def minibatch_GD(self):
        BATCH_SIZE

    def feed_forward(self, X):
        A = np.array(X).T # A(0) = input vector (784,1)
        print('initial A:',A.shape)
        Z = []
        for l in range(LAYERS+1):
            #print('==============')
            #print('l:',l)
            #print('weight:',self.weight[l].shape)
            #print('A:',A.shape)
            Z = np.dot(self.weight[l], A)  #10,784  784,784
            Z = np.add(Z, self.bias[l])
            A = self.sigmoid(Z)
            #print('after-A:',A.shape)
        print('Feed forward, output: ',A)
        return A

    def back_propagate(self):

        return -1

    def shuffle_data(self):
        # shuffle before minibatchGD 
        print('shuffle...')
        pairs = list(zip(self.Xs, self.Labels))
        random.shuffle(pairs)
        self.Xs, self.Labels = zip(*pairs)
        print('shuffle...Finished!')
        #pair = np.concatenate((self.Xs, self.Labels), axis=1)
        #np.random.shuffle(pair)
        #split = np.hsplit(pair, np.array([(-1)*self.Labels[0].shape[0], train_label.shape[0]])) #-10, 45000
        #self.Xs = split[0]
        #self.Labels = split[1]

    def cost_function(self, Target, Output):
        mul = np.multiply(Target, np.log(Output))  # BATCH_SIZE x 10
        return np.negative( np.divide(np.sum(mul), BATCH_SIZE))

    def sigmoid(self, T):
        return np.reciprocal( np.add(1, np.exp( np.negative(T))))

    def sigmoid_derivative(self, S):
        #S'(t)=S(t)(1-S(t))
        return np.multiply(S, np.subtract(1,S))

    def tanh(self, T):
        return np.tanh(T)

    def draw_normal(self, size):
        mu, sigma = 0, 0.01
        return np.random.normal(mu, sigma, size)

if __name__ == "__main__":
    #read from file
    mat_contents = sio.loadmat('SVHN.mat', struct_as_record=False)

    train_x = mat_contents['train_x']   #(45000, 784)
    train_x = np.array_split(train_x, 45000)
    print('train_x[0]:',train_x[0].shape)
    train_label = mat_contents['train_label']   #(45000, 10)
    train_label = np.array_split(train_label, 45000)
    print('train_label[0]:',train_label[0].shape)

    test_x = mat_contents['test_x'] #(15000, 784)
    test_x = np.array_split(test_x, 15000)
    test_label = mat_contents['test_label']
    test_label = np.array_split(test_label, 15000) #15000 x (1,10) nd_array  

    #train
    nn = NeuralNetwork(train_x, train_label, n_hidden_nodes = 15)
    for x in test_x:
        nn.feed_forward(x)

    #nn.test(test_x, test_label)

__author__ = 'ChiWeiHsiao'

import scipy.io as sio
import numpy as np
import random
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self, train_x, train_label, h_nodes1=500, h_nodes2=250, batch_size=100, learning_rate=0.005, n_hidden_layers=2, epoches=1000, plot=True, outfile='try.txt'):
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
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.plot = plot #plot the learning curve

        self.weight = [] 
        self.bias = []
        self.initialize_weight()
        print('NN Initialize...finish!')
        
        self.filename = outfile
        print('number of hidden layers: '+str(n_hidden_layers))
        print('hidden nodes 1: '+str(h_nodes1))
        print('hidden nodes 2: '+str(h_nodes2))
        print('initial learning rate (step decay): '+str(learning_rate))
        print('batch size: '+str(batch_size))
        print('epoches: '+str(epoches))

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
        print('==================================')

    def forward_propagate(self, X):
        a = X #input vector (1, 748)
        A = []
        A.append(a)
        z = []
        for l in range(self.hlayers+1):
            #print('=forward=============')
            #print('l:',l)
            #print('weight:',self.weight[l].shape)
            #print('a:',a.shape)
            z = np.dot(a, self.weight[l])
            #print('doted z:',z.shape)
            z = np.add(z, self.bias[l])
            #print('added z:',z.shape)
            if l == self.hlayers:
                a = self.softmax(z)
            else:
                a = self.sigmoid(z) #(10,)
            A.append(a)
        return A    #return the ouput value of each node in each layer 

    def cost_function(self, Target, Output):
        #of a single example
        #Target: one-hot vector
        #Output: probability vector
        mul = np.multiply(Target, np.log(Output))  # BATCH_SIZE x 10
        return np.negative( np.divide(np.sum(mul), self.batch_size))

    def train(self):
        #Parameters
        m = len(self.Xs)
        batch_size = self.batch_size
        epoches = self.epoches
        last_batch = m - batch_size + 1 #49500-100+1
        #Statistics
        total_error = [] #cost function
        miss_classify = []

        for epoche in range(1, epoches+1):
            self.learning_rate = self.init_learning_rate / (1+epoche) #1/t decay
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
                    error = self.cost_function(self.Labels[p], A[-1])
                    delta = self.get_shape_of_node()    # delta[l][i] = (10,) #array
                    delta[self.layers-1] = np.subtract(A[self.layers-1], self.Labels[p]) #softmax derivative = y-t
                    #delta[self.layers-1] = np.negative(np.multiply(self.Labels[p], np.subtract(1, A[self.layers-1]))) #-t(1-y)
                    for l in range(self.layers-2, 0,-1):
                        # TODO: Vectorization this loop
                        for i in range(self.shape[l]):
                            weighted_sum = np.dot(self.weight[l][i], delta[l+1]) #sum over next layer, j
                            delta[l][i] = weighted_sum * A[l][i] * (1-A[l][i]) #f'(z) #delta[l][i], scalar
                    for l in range(0, self.layers-1):
                        Gradient[l] = np.add(Gradient[l], np.dot(A[l][:, np.newaxis], delta[l+1][np.newaxis])) 
                        Gradient_bias[l] = np.add(Gradient_bias[l], delta[l+1]) 
                #update weight and bias once a batch
                for l in range(self.layers-1):
                    D = np.divide(Gradient[l], batch_size)
                    D_bias = np.divide(Gradient_bias[l], batch_size)
                    #print('D_',l,': ', D)
                    #print('D_bias',l,': ', D_bias)
                    self.weight[l] = np.subtract(self.weight[l], np.multiply(self.learning_rate,D)) 
                    self.bias[l] =  np.subtract(self.bias[l], np.multiply(self.learning_rate, D_bias))
            print('miss in this epoche: ', self.miss_classify_rate(train_x, train_label))

        #stop training when error < 10 %

        #print('Missclassification rate on Training Data = ', miss_classify/m )

        if(self.plot):
            print('plot 2 learning curve: error, miss classification rate')
            # x-axis is epoches
            #plt.
        return 'finish all epoches!'

    def train_regularization(self):
        labda = 0.01 
        #weight decay

    def train_ADAM(self):
        #Adaptive Moment Estimation
        1
    
    def predict(self, X):
        #forward_propagation
        a = X   #(1,748)
        #print('a', a.shape)
        z = []
        for l in range(self.hlayers+1):
            z = np.dot(a, self.weight[l])
            #print('dot z', z.shape)
            z = np.add(z, self.bias[l])
            #print('added z', z.shape)
            if l == self.hlayers:
                a = self.softmax(z)
            else:
                a = self.sigmoid(z)
        #predict = np.argmax(a)
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
        return np.reciprocal( np.add(1, np.exp( np.negative(z))))

    def sigmoid_derivative(self, s):
        #S'(t)=S(t)(1-S(t))
        return np.multiply(s, np.subtract(1,s))

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))

    def draw_normal(self, size):
        mu, sigma = 0, 0.01
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

    def show_weight_and_bias(self, filename = 'none'):
        if filename == 'none':
            print('Show weight and bias on screen!')
            for l in range(self.layers-1):
                print('------layer_',l,'------------')
                print('Weight[',l,']:')
                print(self.weight[l])
                print('Bias[',l,']:')
                print(self.bias[l])
        else:
            with open(filename, 'ab') as fp:
                for l in range(self.layers-1):
                    #fp.write('------layer_'+str(l)+'------------')
                    #fp.write('Weight['+str(l)+']:')
                    msg = '------layer_'+str(l)+'------------\n'
                    fp.write(msg.encode('utf-8'))
                    msg = 'Weight['+str(l)+']: \n'
                    fp.write(msg.encode('utf-8'))
                    for w in self.weight:
                        np.savetxt(fp, w, fmt='%-4.5f')
                    msg = 'Bias['+str(l)+']: \n'
                    fp.write(msg.encode('utf-8'))
                    for b in self.bias:
                        np.savetxt(fp, b, fmt='%-4.5f')


if __name__ == "__main__":
    #read from file
    mat_contents = sio.loadmat('SVHN.mat', struct_as_record=False)

    train_x = mat_contents['train_x']   #(45000, 784)
    #train_x = np.split(train_x, 45000)
    print('train_x[0]:',train_x[0].shape)
    print('train_x[0]:',train_x[0].shape[0])
    train_label = mat_contents['train_label']   #(45000, 10)
    #train_label = np.array_split(train_label, 45000)
    print('train_label[0]:',train_label[0].shape)

    test_x = mat_contents['test_x'] #(15000, 784)
    #test_x = np.array_split(test_x, 15000)
    test_label = mat_contents['test_label']
    #test_label = np.array_split(test_label, 15000) #15000 x (1,10) nd_array  

    #training
    node1, node2, batch, rate, l, e = 128, 32, 4500, 0.1, 2, 50
    filename = 'n1_'+str(node1)+'n2_'+str(node2)+'-rate_'+str(rate)+'-l_'+str(l)+'-e_'+str(e)+'-b_'+str(batch)+'.txt'
    
    nn = NeuralNetwork(train_x, train_label, h_nodes1=node1, h_nodes2=node2, batch_size=batch, learning_rate=rate, n_hidden_layers=l, epoches=e, plot=False, outfile=filename)
    print('before train, cost function: ',nn.cost_function( train_label[0], nn.predict(train_x[0])))
    nn.train()
    nn.show_weight_and_bias(filename)
    print('after train, cost function: ',nn.cost_function( train_label[0], nn.predict(train_x[0])))


    #testing
    miss_rate_train = nn.miss_classify_rate(train_x, train_label)
    miss_rate_test = nn.miss_classify_rate(test_x, test_label)
    with open(filename, 'a') as fp:
        fp.write('Missclassification rate on Training Data = '+ str(miss_rate_train)+'\n')
        fp.write('Missclassification rate on Testing Data = '+str(miss_rate_test)+'\n')
    print('Missclassification rate on Training Data = ', miss_rate_train)
    print('Missclassification rate on Testing Data = ', miss_rate_test)

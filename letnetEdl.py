#It associates a likelihood value with each categorical distribution.

#import necessary libraries
import os
import tensorflow as tf
import numpy as np
import scipy.ndimage as nd
import pylab as pl
from matplotlib import pyplot as plt
from IPython import display
from tensorflow.examples.tutorials.mnist import input_data

# Download MNIST dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#print(mnist.train.images.shape)

K= 10 # number of classes

#digit_one = mnist.train.images[4].copy()
#plt.imshow(digit_one.reshape(28,28)) 
#plt.show()

# define some utility functions
def var(name, shape, init=None):
    if init is None:
        #stddev: a python scalar or a scalar tensor. 
        #Standard deviation of the random values to generate
        init = tf.truncated_normal_initializer(stddev = (2 / shape[0]) ** 0.5)
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                          initializer=init)

#Must have strides[0] = strides[3] = 1. 
#For the most common case of the same horizontal and vertices strides, 
#strides = [1, stride, stride, 1]
def conv(Xin, f, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(Xin, f, strides, padding)

def max_pool(Xin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(Xin, ksize, strides, padding)

def rotate_img(x, deg):
    import scipy.ndimage as nd
    return nd.rotate(x.reshape(28,28),deg,reshape=False).ravel()

# This function to generate evidence is used for the first example
def relu_evidence(logits):
    #Computes rectified linear: max(features, 0)
    return tf.nn.relu(logits)

# This one usually works better and used for the second and third examples
# For general settings and different datasets, you may try this one first
def exp_evidence(logits): 
    #Computes exponential of x element-wise. \(y = e^x\)
    return tf.exp(tf.clip_by_value(logits,-10,10))
    #Any values less than clip_value_min are set to clip_value_min.
    #Any values greater than clip_value_max are set to clip_value_max

# This one is another alternative and 
# usually behaves better than the relu_evidence 
def softplus_evidence(logits):
    #Computes softplus: log(exp(features) + 1)
    return tf.nn.softplus(logits)

def KL(alpha):
    #tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]
    beta=tf.constant(np.ones((1,K)),dtype=tf.float32)
    #x = tf.constant([[1, 1, 1], [1, 1, 1]])
    #tf.reduce_sum(x, 1, keepdims=True)  # [[3], [3]]
    S_alpha = tf.reduce_sum(alpha,axis=1,keep_dims=True)
    S_beta = tf.reduce_sum(beta,axis=1,keep_dims=True)
    #Computes the log of the absolute value of Gamma(x) element-wise
    lnB = tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha),axis=1,keep_dims=True)
    lnB_uni = tf.reduce_sum(tf.lgamma(beta),axis=1,keep_dims=True) - tf.lgamma(S_beta)
    
    #Computes Psi, the derivative of Lgamma 
    #the log of the absolute value of Gamma(x)), element-wi
    dg0 = tf.digamma(S_alpha)
    dg1 = tf.digamma(alpha)
    
    kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keep_dims=True) + lnB + lnB_uni
    return kl

def mse_loss(p, alpha, global_step, annealing_step): 
    S = tf.reduce_sum(alpha, axis=1, keep_dims=True) 
    E = alpha - 1
    m = alpha / S
    
    # Using the equaltion (5)
    A = tf.reduce_sum((p-m)**2, axis=1, keep_dims=True) 
    B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keep_dims=True) 
    
    #Returns the min of x and y (i.e. x < y ? x : y) element-wise.
    #Casts a tensor to a new type
    #x = tf.constant([1.8, 2.2], dtype=tf.float32)
    #tf.cast(x, tf.int32)  # [1, 2], dtype=tf.int32
    annealing_coef = tf.minimum(1.0, tf.cast(global_step/annealing_step, tf.float32))
    
    alp = E*(1-p) + 1 
    C =  annealing_coef * KL(alp)
    return (A + B) + C

# train LeNet network with expected mean square error loss
def LeNet_EDL(logits2evidence=relu_evidence,loss_function=mse_loss, lmb=0.005):
    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(shape=[None,28*28], dtype=tf.float32)
        Y = tf.placeholder(shape=[None,10], dtype=tf.float32)
        keep_prob = tf.placeholder(dtype=tf.float32)

        #A variable maintains state in the graph across calls to run(). 
        #You add a variable to the graph by constructing an instance of the class Variable
        global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
        annealing_step = tf.placeholder(dtype=tf.int32) 
    
        # first hidden layer - conv
        W1 = var('W1', [5,5,1,20])
        b1 = var('b1', [20])
        out1 = max_pool(tf.nn.relu(conv(tf.reshape(X, [-1, 28,28, 1]), 
                                        W1, strides=[1, 1, 1, 1]) + b1))
        # second hidden layer - conv
        W2 = var('W2', [5,5,20,50])
        b2 = var('b2', [50])
        out2 = max_pool(tf.nn.relu(conv(out1, W2, strides=[1, 1, 1, 1]) + b2))
        
        # flatten the output
        Xflat = tf.contrib.layers.flatten(out2)
        
        # third hidden layer - fully connected
        W3 = var('W3', [Xflat.get_shape()[1].value, 500])
        b3 = var('b3', [500]) 
        out3 = tf.nn.relu(tf.matmul(Xflat, W3) + b3)
        out3 = tf.nn.dropout(out3, keep_prob=keep_prob)
        
        #output layer
        W4 = var('W4', [500,10])
        b4 = var('b4',[10])
        #Multiplies matrix a by matrix b, producing a * b
        logits = tf.matmul(out3, W4) + b4
        
        evidence = logits2evidence(logits)
        alpha = evidence + 1
        
        u = K / tf.reduce_sum(alpha, axis=1, keep_dims=True) #uncertainty
        prob = alpha / tf.reduce_sum(alpha, 1, keepdims=True) #probability

        #Computes the mean of elements across dimensions of a tensor.
        #x = tf.constant([[1., 1.], [2., 2.]])
        #tf.reduce_mean(x)  # 1.5
        print ('^^^^^Global_step: ', global_step)
        loss = tf.reduce_mean(loss_function(Y, alpha, global_step, annealing_step))
        
        #Computes half the L2 norm of a tensor without the sqrt
        #output = sum(t ** 2) / 2
        l2_loss = (tf.nn.l2_loss(W3)+tf.nn.l2_loss(W4)) * lmb
        
        #Optimizer that implements the Adam algorithm.
        step = tf.train.AdamOptimizer().minimize(loss + l2_loss, global_step=global_step)
        
        # Calculate accuracy
        pred = tf.argmax(logits, 1)
        truth = tf.argmax(Y, 1)
        match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
        acc = tf.reduce_mean(match)
        
        total_evidence = tf.reduce_sum(evidence, 1, keepdims=True) 
        mean_ev = tf.reduce_mean(total_evidence)
        mean_ev_succ = tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True) * match) / tf.reduce_sum(match + 1e-20)
        mean_ev_fail = tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True) * (1-match)) / (tf.reduce_sum(tf.abs(1-match))+1e-20) 
        
        return g, step, X, Y, annealing_step, keep_prob, prob, acc, loss, u, evidence, mean_ev, mean_ev_succ, mean_ev_fail

g2, step2, X2, Y2, annealing_step, keep_prob2, prob2, acc2, loss2, u, evidence, \
    mean_ev, mean_ev_succ, mean_ev_fail= LeNet_EDL()

sess2 = tf.Session(graph=g2)
with g2.as_default():
    sess2.run(tf.global_variables_initializer())

bsize = 1000 #batch size
n_batches = mnist.train.num_examples // bsize
L_train_acc1=[]
L_train_ev_s=[]
L_train_ev_f=[]

L_test_acc1=[]
L_test_ev_s=[]
L_test_ev_f=[]

checkpoint_path = "./training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=5)
# Display the model's architecture

for epoch in range(50):   
    for i in range(n_batches):
        data, label = mnist.train.next_batch(bsize)
        # print ('Label shape:', label.shape)
        # print ('Data shape:', data.shape)
        # print (label[2: 5, :])
        feed_dict={X2:data, Y2:label, keep_prob2:.5, annealing_step:10*n_batches}
        sess2.run(step2, feed_dict)
        print('epoch %d - %d%%) '% (epoch+1, (100*(i+1))//n_batches), end='\r' if i<n_batches-1 else '')
        
    train_acc, train_succ, train_fail = sess2.run([acc2, mean_ev_succ, mean_ev_fail], feed_dict={X2:mnist.train.images,Y2:mnist.train.labels,keep_prob2:1.})
    test_acc, test_succ, test_fail = sess2.run([acc2,mean_ev_succ,mean_ev_fail], feed_dict={X2:mnist.test.images,Y2:mnist.test.labels,keep_prob2:1.})
    
    L_train_acc1.append(train_acc)
    L_train_ev_s.append(train_succ)
    L_train_ev_f.append(train_fail)
    
    L_test_acc1.append(test_acc)
    L_test_ev_s.append(test_succ)
    L_test_ev_f.append(test_fail)
    
    print('training: %2.4f (%2.4f - %2.4f) \t testing: %2.4f (%2.4f - %2.4f)' % 
          (train_acc, train_succ, train_fail, test_acc, test_succ, test_fail))


def draw_EDL_results(train_acc1, train_ev_s, train_ev_f, test_acc1, test_ev_s, test_ev_f): 
    # calculate uncertainty for training and testing data for correctly and misclassified samples
    train_u_succ = K / (K+np.array(train_ev_s))
    train_u_fail = K / (K+np.array(train_ev_f))
    test_u_succ  = K / (K+np.array(test_ev_s))
    test_u_fail  = K / (K+np.array(test_ev_f))
    
    f, axs = pl.subplots(2, 2)
    f.set_size_inches([10,10])
    
    axs[0,0].plot(train_ev_s,c='r',marker='+')
    axs[0,0].plot(train_ev_f,c='k',marker='x')
    axs[0,0].set_title('Train Data')
    axs[0,0].set_xlabel('Epoch')
    axs[0,0].set_ylabel('Estimated total evidence for classification') 
    axs[0,0].legend(['Correct Clasifications','Misclasifications'])
    
    
    axs[0,1].plot(train_u_succ,c='r',marker='+')
    axs[0,1].plot(train_u_fail,c='k',marker='x')
    axs[0,1].plot(train_acc1,c='blue',marker='*')
    axs[0,1].set_title('Train Data')
    axs[0,1].set_xlabel('Epoch')
    axs[0,1].set_ylabel('Estimated uncertainty for classification')
    axs[0,1].legend(['Correct clasifications','Misclasifications', 'Accuracy'])
    
    axs[1,0].plot(test_ev_s,c='r',marker='+')
    axs[1,0].plot(test_ev_f,c='k',marker='x')
    axs[1,0].set_title('Test Data')
    axs[1,0].set_xlabel('Epoch')
    axs[1,0].set_ylabel('Estimated total evidence for classification') 
    axs[1,0].legend(['Correct Clasifications','Misclasifications'])
    
    
    axs[1,1].plot(test_u_succ,c='r',marker='+')
    axs[1,1].plot(test_u_fail,c='k',marker='x')
    axs[1,1].plot(test_acc1,c='blue',marker='*')
    axs[1,1].set_title('Test Data')
    axs[1,1].set_xlabel('Epoch')
    axs[1,1].set_ylabel('Estimated uncertainty for classification')
    axs[1,1].legend(['Correct clasifications','Misclasifications', 'Accuracy'])
    plt.show()

#draw_EDL_results(L_train_acc1, L_train_ev_s, L_train_ev_f, L_test_acc1, L_test_ev_s, L_test_ev_f)

# This method rotates an image counter-clockwise and classify it for different degress of rotation. 
# It plots the highest classification probability along with the class label for each rotation degree.
def rotating_image_classification(img, sess, prob, X, keep_prob, uncertainty=None, threshold=0.5):
    Mdeg = 180 
    Ndeg = int(Mdeg/10)+1
    ldeg = []
    lp = []
    lu=[]
    scores = np.zeros((1,K))
    rimgs = np.zeros((28,28*Ndeg))

    for i, deg in enumerate(np.linspace(0,Mdeg, Ndeg)):
        nimg = rotate_img(img,deg).reshape(28,28)
        nimg = np.clip(a=nimg,a_min=0,a_max=1)
        rimgs[:,i*28:(i+1)*28] = nimg
        feed_dict={X:nimg.reshape(1,-1), keep_prob:1.0}
        if uncertainty is None:
            p_pred_t = sess.run(prob, feed_dict=feed_dict)
        else:
            p_pred_t,u = sess.run([prob,uncertainty], feed_dict=feed_dict)
            lu.append(u.mean())
        scores += p_pred_t >= threshold
        ldeg.append(deg) 
        lp.append(p_pred_t[0])

    labels = np.arange(10)[scores[0].astype(bool)]
    lp = np.array(lp)[:,labels]
    c = ['black','blue','red','brown','purple','cyan']
    marker = ['s','^','o']*2
    labels = labels.tolist()
    
    for i in range(len(labels)):
        plt.plot(ldeg,lp[:,i],marker=marker[i],c=c[i])
    
    if uncertainty is not None:
        labels += ['uncertainty']
        plt.plot(ldeg,lu,marker='<',c='red')
        
    plt.legend(labels)
 
    plt.xlim([0,Mdeg])  
    plt.xlabel('Rotation Degree')
    plt.ylabel('Classification Probability')
    plt.show()

    plt.figure(figsize=[6.2,100])
    plt.imshow(1-rimgs,cmap='gray')
    plt.axis('off')
    plt.show()

rotating_image_classification(digit_one, sess2, prob2, X2, keep_prob2, u)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


ROW = 1166

def file2Array(filename):
    
    output = [0] * (ROW * ROW)
    
    print(filename)
    fIn = open(filename, 'r')
    lines = fIn.readlines()
    
    for line in lines:
        parts = line.split(',')
        row = int(parts[0])
        col = int(parts[1])
        val = float(parts[2])
        output[row*ROW + col] = val
        
    fIn.close()
    return output


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


    

if __name__ == '__main__':
    inputFilePath = './Neural_data/Control/'
    
    result = []
    
    for path, subdirs, files in os.walk(inputFilePath):
        for file in files:
#             print(file)
            result.append(file2Array(inputFilePath + file))
            
    X = tf.placeholder(tf.float32, shape=[None, 1359556])

    D_W1 = tf.Variable(xavier_init([1359556, 1166]))
    D_b1 = tf.Variable(tf.zeros(shape=[1166]))
    
    D_W2 = tf.Variable(xavier_init([1166, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))
    
    theta_D = [D_W1, D_W2, D_b1, D_b2]
    
    
    Z = tf.placeholder(tf.float32, shape=[None, 100])
    
    G_W1 = tf.Variable(xavier_init([100, 1166]))
    G_b1 = tf.Variable(tf.zeros(shape=[1166]))
    
    G_W2 = tf.Variable(xavier_init([1166, 1359556]))
    G_b2 = tf.Variable(tf.zeros(shape=[1359556]))
    
    theta_G = [G_W1, G_W2, G_b1, G_b2]
    
    
    
    G_sample = generator(Z)
    D_real, D_logit_real = discriminator(X)
    D_fake, D_logit_fake = discriminator(G_sample)
    
    # D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
    # G_loss = -tf.reduce_mean(tf.log(D_fake))
    
    # Alternative losses:
    # -------------------
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    
    mb_size = 1
    Z_dim = 100
        
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    if not os.path.exists('out_temp/'):
        os.makedirs('out_temp/')
        
        
    i = 0

    for it in range(20000):
#         if it % 1000 == 0:
#             samples = sess.run(G_sample, feed_dict={Z: sample_Z(1, Z_dim)})
#      
#             fig = plot(samples)
#             plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
#             i += 1
#             plt.close(fig)
    
#         X_mb, _ = mnist.train.next_batch(mb_size)
        iter = it % 76
        
        X_mb = result[iter:iter + mb_size]
        
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})
    
        print(it)
        
        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={Z: sample_Z(1, Z_dim)})
            fout = open('out_temp/{}.txt'.format(str(i).zfill(3)), 'w')
            fout.write(str(samples))
            fout.close()
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()
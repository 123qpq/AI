import numpy as np
import tensorflow as tf

X = np.array([1., 2., 3., 4.])
Y = np.array([1., 3., 5., 7.])

W = tf.Variable(tf.random_normal_initializer(-100., 100.)(shape=[1])) #-100~100 범위 정규분포값 생성

for step in range(300):
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    #Gradient descent
    alpha = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W,X)-Y, X))
    descent = W - tf.multiply(alpha, gradient) #W = W-alpha/m * sigma{(w*x-y)x}
    W.assign(descent)
    
    if step % 10 == 0:
        print('{:5} | {:10.4f} | {:10.6f}'.format(step, cost.numpy(), W.numpy()[0]))
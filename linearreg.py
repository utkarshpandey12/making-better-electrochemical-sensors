import numpy as np
import pandas as pd
import tensorflow as tf
sen = pd.read_csv('sensor.csv')
x1Data = np.array(sen["CO SENSOR COUNT"],np.float32)
x2Data = np.array(sen["H2S SENSOR COUNT"],np.float32)
yData = np.array(sen["OUTPUT CONCENTRATION"],np.float32)
W1 = tf.Variable(tf.zeros([1, 1]),dtype=tf.float32, name="W1")
W2 = tf.Variable(tf.zeros([1, 1]),dtype=tf.float32,name="W2")
b = tf.Variable(tf.zeros([1]),dtype=tf.float32, name="b")
X1 = tf.placeholder(tf.float32, [None, 1], name="X1")
X2 = tf.placeholder(tf.float32, [None, 1], name="X2")
W_1 = tf.matmul(X1,W1)
W_2 = tf.matmul(X2,W2)
y = W_1 + W_2 + b
y_ = tf.placeholder(tf.float32, [None, 1])
cost = tf.reduce_mean(tf.square(y_ - y))
train_step_ftrl = tf.train.FtrlOptimizer(learning_rate=0.1).minimize(cost)
x1_data = x1Data.reshape(-1, 1)
x2_data = x2Data.reshape(-1, 1)
y_data = yData.reshape(-1, 1)
dataset_size = len(x1_data)
init = tf.global_variables_initializer()
with tf.Session() as sess:
  
   sess.run(init)
   for i in range(50000):
    feed = { X1: x1_data, X2: x2_data, y_: y_data }
    sess.run(train_step_ftrl, feed_dict=feed)
   print("W1: %s" % sess.run(W1))
   print("W2: %s" % sess.run(W2))
   print("b: %f" % sess.run(b))
   print("cost: %f" % sess.run(cost, feed_dict=feed))
    

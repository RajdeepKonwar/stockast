##
# @author Premanand Kumar (p8kumar AT ucsd.edu)
#
# @section LICENSE
# Copyright (c) 2017-18, Regents of the University of California
# All rights reserved.
#
# @section REQUIREMENTS
#   Python 2 or 3
#   numpy
#   pandas
#   scikit-learn
#   TensorFlow - 1.0.0
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##

import  csv
import  matplotlib.pyplot     as plt
import  numpy                 as np
import  pandas                as pd
import  tensorflow            as tf
from    sklearn.preprocessing import StandardScaler, Imputer
from    sklearn.pipeline      import Pipeline

csv_file      = "Desktop/ML/train.csv"
csv_all       = pd.read_csv(csv_file)
csv_all.fillna(method='ffill', inplace=True)
csv_all       = csv_all.fillna(0)
csv_ret       = csv_all.loc[0,'Ret_2':'Ret_180']
features      = csv_all.loc[0,'Feature_1':'Feature_25']
features      = features.values.tolist()
num_pipeline  = Pipeline([('std_scaler', StandardScaler())])
features      = num_pipeline.fit_transform(csv_all.loc[0,'Feature_1':'Feature_25'])
data          = csv_all.loc[0,'Ret_2':'Ret_180']
spot          = 100
st_price      = []

for i in range(179):
    stock_price = np.exp(csv_ret[i-1])*spot
    spot = stock_price
    st_price.append(stock_price)
plt.plot(np.arange(1,180),st_price)
plt.show()
csv_ret = st_price

t_min, t_max = 1,180
resolution = 1

X_batch = np.transpose(csv_ret[0:90])
X_batch_append = features
for i in range(89):
    X_batch_append = np.append(X_batch_append,features)
X_batch = np.array(X_batch).reshape(9,10,1)
X_batch_append = np.array(X_batch_append).reshape(9,10,25)
X_batch = np.append(X_batch_append, X_batch,axis = 2)
X_batch = np.array(X_batch).reshape(9,10,26)
y_batch = np.transpose(csv_ret[1:91])
y_batch = np.array(y_batch).reshape(9,10,1)
print(type(X_batch))

n_steps   = 10
n_inputs  = 26
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
    output_size=n_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

learning_rate = 0.0001
loss = tf.reduce_mean(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

saver = tf.train.Saver()

n_iterations  = 10000
batch_size    = 10

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict = {X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
        saver.save(sess, "./stock_price")

with tf.Session() as sess:
    saver.restore(sess, "./stock_price")

    X_new = X_batch
    y_pred = sess.run(outputs, feed_dict={X: X_new})

print(y_pred)
y_pred.size

with tf.Session() as sess:
    saver.restore(sess, "./stock_price")
    sequence = X_batch[8][:][:]
    answer = []
    n_steps_update = n_steps
    iteration = 0;
    X_batch_feed = sequence[:][iteration:n_steps_update+iteration][:]
    X_batch_feed = np.array(X_batch_feed).reshape(1,10, 26)
    y_pred = sess.run(outputs, feed_dict={X: X_batch_feed})
    append = np.append(features,y_pred[0, -1, 0])
    sequence = np.append(sequence,append)
    n_steps_update = n_steps_update +1
    sequence = np.array(sequence).reshape(1,n_steps_update, 26)
    answer.append(y_pred[0, -1, 0])

    for iteration in range(1,90):
            X_batch_feed = sequence[0][iteration:1+n_steps_update+iteration][:]
            X_batch_feed = np.array(X_batch_feed).reshape(1,10, 26)
            y_pred = sess.run(outputs, feed_dict={X: X_batch_feed})
            append = np.append(features,y_pred[0, -1, 0])
            sequence = np.append(sequence,append)
            n_steps_update = n_steps_update +1
            sequence = np.array(sequence).reshape(1,n_steps_update, 26)
            answer.append(y_pred[0, -1, 0])

with open("Desktop/ML/ml_data.csv", 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(csv_ret[0:90] + answer)

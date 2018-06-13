##
# @file This file is part of stockast.
#
# @section LICENSE
# MIT License
# 
# Copyright (c) 2017-18 Premanand Kumar, Rajdeep Konwar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# @section DESCRIPTION
# Machine learning script.
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

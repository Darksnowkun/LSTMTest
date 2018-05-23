import numpy as np
from random import shuffle
import tensorflow as tf

import time
start_time = time.time()


trainingInput = ['{0:020b}'.format(i) for i in range(2**20)]
shuffle(trainingInput)
trainingInput = [map(int, i) for i in trainingInput]
ti = []
for i in trainingInput:
    tempList = []
    for j in i:
        tempList.append([j])
    ti.append(np.array(tempList))

trainingInput = ti

dehbug = 1

trainingOutput = []

for i in trainingInput:
    count = 0
    for j in i:
        if j[0] == 1:
            count += 1
    temp_list = ([0] * 21)
    temp_list[count] = 1
    trainingOutput.append(temp_list)

dehbug = 1

NUM_EXAMPLES = 10000
test_input = trainingInput[NUM_EXAMPLES:]
test_output = trainingOutput[NUM_EXAMPLES:]  # everything beyond 10,000

train_input = trainingInput[:NUM_EXAMPLES]
train_output = trainingOutput[:NUM_EXAMPLES]  # till 10,000

dehbug = 1

# ----------- TENSORFLOW START ---------------------------#

with tf.device('/GPU:0'):

    data = tf.placeholder(tf.float32, [None, 20, 1])
    target = tf.placeholder(tf.float32, [None, 21])

    num_hidden = 24
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)

    val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init_op)

batch_size = 500
no_of_batches = int(len(train_input)/batch_size)
epoch = 500
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr += batch_size
        sess.run(minimize,{data: inp, target: out})
    print("Epoch - ",str(i))
    incorrect = sess.run(error,{data: test_input, target: test_output})
    print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))\

sess.close()

#print(sess.run(model.prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]}))

print("--- %s seconds ---" % (time.time() - start_time))
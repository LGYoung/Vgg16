import numpy as np
import tensorflow as tf
from vgg16 import Vgg16
from data_utils import load_CIFAR10, fill_feed_dict, one_hot

np.random.seed(1)
BATCH_SIZE = 128


train_X, train_Y, test_X, test_Y = load_CIFAR10('E:\ml初探\Vgg3\cifar-10-batches-py')
train_Y = one_hot(train_Y, 10)
test_Y = one_hot(test_Y, 10)

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.int16, [None, 10])

vgg = Vgg16(X)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=vgg.fc8))
correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(vgg.probs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

# optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    with tf.device('/gpu:0'):
        init = tf.global_variables_initializer()
        sess.run(init)
        vgg.load_weights(['fc6', 'fc7', 'fc8'], 'vgg16.npy', sess)
        print('------------------------------train--------------------------------------------')
        for epoch in range(101):
            for x_batch, y_batch in fill_feed_dict(train_X, train_Y, BATCH_SIZE):
                fetches = [optimizer, loss, accuracy]
                _, cost, acc = sess.run(fetches, feed_dict={X: x_batch, Y: y_batch})
            if epoch % 10 == 0:
                print('epoch:', epoch, '       ', 'cost:', cost, '        ', 'accuracy:', acc)
        print('-------------------------------------------------------------------------------')
        saver.save(sess, save_path='E:\ml初探\Vgg3\parameters\weights')
        print('------------------------------test---------------------------------------------')
        n_steps = test_X.shape[0] // BATCH_SIZE
        n_samples = BATCH_SIZE * n_steps
        total_correct_pred = 0
        print('test samples:', n_samples)
        for x_batch, y_batch in fill_feed_dict(test_X, test_Y, BATCH_SIZE):
            batch_correct = sess.run(correct_pred, feed_dict={X: x_batch, Y: y_batch})
            total_correct_pred += np.sum(batch_correct)
        print('correct samples:', total_correct_pred)
        print('test accuracy:', total_correct_pred/n_samples*100, '%')
        print('-------------------------------------------------------------------------------')



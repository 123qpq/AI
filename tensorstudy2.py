# MNIST 데이터 가져옴
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
# 이미지 데이터 플레이스홀더
x = tf.placeholder("float", [None, 784])

# 웨이트와 바이어스 변수
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 모델 구현
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 정답 레이블용 플레이스 홀더
y_ = tf.placeholder("float", [None,10])

# Loss 함수
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 학습 오퍼레이션
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 모든 변수 초기화
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# 임의로 100개 샘플링
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print(i)
# 정답율
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
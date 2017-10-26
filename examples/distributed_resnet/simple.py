import json

import tensorflow as tf

cluster_spec = json.load(open('config.json', 'rt'))
cluster_spec = tf.train.ClusterSpec(cluster_spec)
server = tf.train.Server(cluster_spec, job_name='host', task_index=0)

with tf.device('/job:worker/task:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.device('/job:worker/task:1'):
    d = tf.matmul(a, b) + tf.log(100 + c)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=True)
with tf.Session(server.target, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    result = sess.run(d)
    print(result)
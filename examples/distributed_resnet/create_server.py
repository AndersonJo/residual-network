import tensorflow as tf
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=int, help='The task number')
parser.add_argument('--job', type=str, default='worker', help='job name ("worker" or "host")')
args = parser.parse_args()

cluster_spec = json.load(open('config.json', 'rt'))
cluster = tf.train.ClusterSpec(cluster_spec)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2, allow_growth=True)
server = tf.train.Server(cluster, job_name=args.job, task_index=args.task,
                         config=tf.ConfigProto(gpu_options=gpu_options))
server.start()
server.join()

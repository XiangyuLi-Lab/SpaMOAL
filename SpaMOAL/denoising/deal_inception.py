'''
Inference of inception-v3 model with pretrained parameters on ImageNet
'''
import os
import time
import sys
import tensorflow.compat.v1 as tf
import argparse
# To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()
import tensorflow_hub as hub
import numpy as np
import cv2
import pandas as pd


parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", type=str, default="E11_0")
parser.add_argument("--model", type=str, default="inception_v3")
parser.add_argument("--size", type=int, default=299)    
args = parser.parse_args()

if args.model == "inception_v3":
    size = 299
    module = hub.Module(f"./{args.model}")
elif args.model == "inception_resnet_v2":
    size = 299
    
    module = hub.load(f"./{args.model}")
elif args.model == "resnet50":
    size = 224
    module = hub.load("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4")
    #module = hub.load(f"./{args.model}") 
else:
    size = 299
# Load saved inception-v3 model
# module = hub.Module(f"./{args.model}")

# images should be resized to 299x299
input_imgs = tf.placeholder(tf.float32, shape=[None, size, size, 3])
features = module(input_imgs)

# Provide the file indices
# This can be changed to image indices in strings or other formats
# spot_info = pd.read_csv(f'./spot_info_{args.name}.csv', header=0, index_col=None)
spot_info = pd.read_csv(f'../Data/{args.name}/{args.name}_position.csv', header = None, index_col=None)
spot_info =spot_info[1:]
print(spot_info)
image_no = spot_info.shape[0]

start = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 分批次处理相关设置
    batch_size = 32  # 可根据实际情况调整批次大小
    num_batches = int(np.ceil(image_no / batch_size))

    fea_all = None  # 用于存储所有批次的特征

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, image_no)

        img_batch = np.zeros([end_idx - start_idx, size, size, 3])

        # 加载当前批次的图像
        for i in range(start_idx, end_idx):
            temp = np.load(f'../Data/{args.name}/img_{args.name}_{args.size}' + '/' + spot_info.iloc[i, 0] + '.npy')
            temp = cv2.resize(temp, (size, size))
            temp2 = temp.astype(np.float32) / 255.0
            img_batch[i - start_idx, :, :, :] = temp2

        # 使用当前批次图像进行特征提取
        fea_batch = sess.run(features, feed_dict={input_imgs: img_batch})

        if fea_all is None:
            fea_all = fea_batch
        else:
            fea_all = np.vstack((fea_all, fea_batch))

    print("step1 sucessful!!")

    df2 = pd.DataFrame(fea_all)
    spot_info = pd.read_csv(f'../Data/{args.name}/{args.name}_position.csv', index_col=None)
    new_index = spot_info.iloc[:, 0]
    df2.set_index(new_index, inplace=True)
    df2.to_csv(f"../Data/{args.name}/{args.name}_image_{args.model}.csv", index=True)

    # 保存推断出的图像特征
    np.save(f'../Data/{args.name}/emb_{args.name}_{args.model}_{args.size}.npy', fea_all)
    print("step2 sucessful!")

    end = time.time()
    print("run time: " + str(end - start) + " seconds")
    # npy-->txt
    # read_npy = np.load(f'./emb_{args.name}_{args.model}.npy')
    # np.savetxt(f'./emb_{args.name}_{args.model}.csv',read_npy,delimiter=',')
    # print("run sucessful!!")
    
 

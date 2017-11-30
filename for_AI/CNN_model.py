#encoding=utf-8
#coding=utf-8
import tensorflow as tf
import handle_picture as hp
from PIL import Image,ImageOps
import numpy as np
import scipy.misc
import handle_picture as hp
import tradition_jepg
from scipy.misc import imread, imresize
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#实现将图片大小变成原来的1/8*1/8,维度不变
def compose(re_image,in_size):
    l1_depth = 32
    l2_depth = 64
    l3_depth = 128
    l4_depth = 32
    l5_depth = 3
    keep_prob = 0.5
    x_image = tf.reshape(re_image,[-1,in_size,in_size,3])
    x_image = tf.cast(x_image,tf.float32)

    def weight_variable(shape):
        initial = tf.truncated_normal(shape,stddev = 0.1)
        return tf.Variable(initial)

    def bia_variable(shape):
        initial = tf.constant(0.1,shape = shape)
        return tf.Variable(initial)

    def conv2d(inputs,W):
        return tf.nn.conv2d(inputs,W,strides = [1,1,1,1],padding = 'SAME')

    def pool(inputs):
        return tf.nn.max_pool(inputs,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')


    #conv layer1
    W_conv1 = weight_variable([5,5,3,l1_depth])
    b_conv1 = bia_variable([l1_depth])
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
    #conv layer2
    W_conv2 = weight_variable([5,5,l1_depth,l2_depth])
    b_conv2 = bia_variable([l2_depth])
    h_conv2 = tf.nn.relu(conv2d(h_conv1,W_conv2) + b_conv2)
    h_pool2 = pool(h_conv2)
    #conv layer3
    W_conv3 = weight_variable([5,5,l2_depth,l3_depth])
    b_conv3 = bia_variable([l3_depth])
    h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3) + b_conv3)
    #conv layer4
    W_conv4 = weight_variable([5,5,l3_depth,l4_depth])
    b_conv4 = bia_variable([l4_depth])
    h_conv4 = tf.nn.relu(conv2d(h_conv3,W_conv4) + b_conv4)
    h_pool4 = pool(h_conv4)
    #conv layer5
    W_conv5 = weight_variable([5,5,l4_depth,l5_depth])
    b_conv5 = bia_variable([l5_depth])
    h_conv5 = tf.nn.relu(conv2d(h_pool4,W_conv5) + b_conv5)
    h_pool5 = pool(h_conv5)
    compact_picture = h_pool5
    
    
    return compact_picture

def decompose(decoded_image,in_size):
    de_l1_depth = 32
    de_l2_depth = 64
    de_l3_depth = 128
    de_l4_depth = 256
    de_l5_depth = 512
    de_l6_depth = 768
    keep_prob1 = 0.5
    temp = int(in_size/64)
    de_f1_length = temp*temp*de_l3_depth 
    def weight_variable(shape):
        initial = tf.truncated_normal(shape,stddev = 0.1)
        return tf.Variable(initial)

    def bia_variable(shape):
        initial = tf.constant(0.1,shape = shape)
        return tf.Variable(initial)

    def conv2d(inputs,W):
        return tf.nn.conv2d(inputs,W,strides = [1,1,1,1],padding = 'SAME')

    def pool(inputs):
        return tf.nn.max_pool(inputs,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')


    #conv layer1
    de_W_conv1 = weight_variable([5,5,3,de_l1_depth])
    de_b_conv1 = bia_variable([de_l1_depth])
    de_h_conv1 = tf.nn.relu(conv2d(decoded_image,de_W_conv1) + de_b_conv1)
    
    #conv layer2
    de_W_conv2 = weight_variable([5,5,de_l1_depth,de_l2_depth])
    de_b_conv2 = bia_variable([de_l2_depth])
    de_h_conv2 = tf.nn.relu(conv2d(de_h_conv1,de_W_conv2) + de_b_conv2)

    #conv layer3
    de_W_conv3 = weight_variable([5,5,de_l2_depth,de_l3_depth])
    de_b_conv3 = bia_variable([de_l3_depth])
    de_h_conv3 = tf.nn.relu(conv2d(de_h_conv2,de_W_conv3) + de_b_conv3)
    
    #conv layer4
    de_W_conv4 = weight_variable([5,5,de_l3_depth,de_l4_depth])
    de_b_conv4 = bia_variable([de_l4_depth])
    de_h_conv4 = tf.nn.relu(conv2d(de_h_conv3,de_W_conv4) + de_b_conv4)
    
    #conv layer5
    de_W_conv5 = weight_variable([5,5,de_l4_depth,de_l5_depth])
    de_b_conv5 = bia_variable([de_l5_depth])
    de_h_conv5 = tf.nn.relu(conv2d(de_h_conv4,de_W_conv5) + de_b_conv5)
    de_h_pool5 = pool(de_h_conv5)
    
    #conv layer6
    de_W_conv6 = weight_variable([5,5,de_l5_depth,de_l6_depth])
    de_b_conv6 = bia_variable([de_l6_depth])
    de_h_conv6 = tf.nn.relu(conv2d(de_h_pool5,de_W_conv6) + de_b_conv6)

    de_W_func1 = weight_variable([de_f1_length,de_f1_length])
    de_b_func1 = bia_variable([de_f1_length])
    de_h_temp = tf.reshape(de_h_conv6,[-1,de_f1_length])
    de_h_func1 = tf.nn.relu(tf.matmul(de_h_temp,de_W_func1) + de_b_func1)


 

    #compact Representation
    re_picture = tf.reshape(de_h_func1,[-1,in_size,in_size,3])

    return re_picture

def main():
    in_size = 256
    temp_size = int(in_size/8)
    image_list,number_img = hp.get_images(in_size)
    image_dic = hp.load_images()
    with tf.Session() as sess:
        for step in range(number_img):
            #create the porper_size and train image which can change
            image_for_train = image_dic[image_list[step]]
            image_for_train = np.asarray(image_for_train)
            image_for_train = image_for_train.astype(np.float32)

            #cnn for compose
            re_image = compose(image_for_train ,in_size)

            #得到传统压缩和解压的输入
            sess.run(tf.global_variables_initializer())
            temp_image = sess.run(re_image)

            #经过传统的压缩和解压处理并且得到porper size for cnn decompose
            temp_image = np.asarray(temp_image[0])
            temp_image = np.clip(temp_image,0,255).astype('uint8')


            jepg_image = tradition_jepg.jepg(temp_image)
            jepg_image = [imresize(jepg_image, (temp_size, temp_size))]

            jepg_image = np.asarray(jepg_image)
            jepg_image = jepg_image.astype(np.float32)

            #cnn decompose
            decomposed_image = decompose(jepg_image,in_size)
            sess.run(tf.global_variables_initializer())
            loss = tf.reduce_mean(tf.square(decomposed_image - image_for_train))
            optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
            sess.run(tf.global_variables_initializer())
            j = 0
            for step in range(1,2000):
                sess.run(optimizer)
                print(sess.run(loss))
                j = j + 1
                if step % 1999 == 0:
                    show_image = sess.run(decomposed_image)
                    img_data = show_image[0]
                    img_data = np.array(img_data).astype('uint8')
                    image = Image.fromarray(img_data)
                    filename = '/home/xzhou/for_AI/outputs/%d.jpg' %(j)
                    scipy.misc.imsave(filename,image)

main()

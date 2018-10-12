import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

%matplotlib inline

model = load_vgg_model("imagenet-vgg-verydeep-19.mat")
print(model)

content_image = scipy.misc.imread("images/louvre.jpg")
imshow(content_image)

def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
  
    a_C_unrolled = tf.transpose(tf.reshape(a_C, [-1]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [-1]))
    
    J_content = tf.reduce_sum((a_C_unrolled - a_G_unrolled)**2) / (4 * n_H * n_W * n_C)
  
    
    return J_content

'''style_image = scipy.misc.imread("images/monet_800600.jpg")
imshow(style_image)'''

def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))
   
    return GA


def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.reshape(a_S, [n_H*n_W, n_C])
    a_G = tf.reshape(a_G, [n_H*n_W, n_C])

    GS = gram_matrix(tf.transpose(a_S)) 
    GG = gram_matrix(tf.transpose(a_G))

    J_style_layer = tf.reduce_sum((GS - GG)**2) / (4 * n_C**2 * (n_W * n_H)**2)
 
    return J_style_layer

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def compute_style_cost(model, STYLE_LAYERS):
  
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        
        a_S = sess.run(out)
        
        a_G = out
       
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
   
    J = alpha * J_content + beta * J_style
    
    return J



tf.reset_default_graph()

sess = tf.InteractiveSession()


img = Image.open('images/Eiffel.jpg')
size = 1000
baseheight = size
wpercent = (baseheight/float(img.size[0]))
wsize = int((float(img.size[0])*float(wpercent)))
img = img.resize((baseheight,wsize), Image.ANTIALIAS)

new_width, new_height = size, size
height, width = img.size

left = (width - new_width)/2
top = (height - new_height)/2
right = (width + new_width)/2
bottom = (height + new_height)/2

img = img.crop((left, top, right, bottom))
img.save('images/content.jpg')
content_image = scipy.misc.imread("images/content.jpg")
imshow(content_image)
content_image = reshape_and_normalize_image(content_image)

img = Image.open('images/Fish.jpg')

baseheight = size
wpercent = (baseheight/float(img.size[0]))
wsize = int((float(img.size[0])*float(wpercent)))
img = img.resize((baseheight,wsize), Image.ANTIALIAS)

new_width, new_height = size, size
height, width = img.size

left = (width - new_width)/2
top = (height - new_height)/2
right = (width + new_width)/2
bottom = (height + new_height)/2

img = img.crop((left, top, right, bottom))
img.save('images/style.jpg')
imshow(img)

style_image = scipy.misc.imread("images/style.jpg")
style_image = reshape_and_normalize_image(style_image)
print(CONFIG.IMAGE_HEIGHT)



generated_image = generate_noise_image(content_image)

model = load_vgg_model("imagenet-vgg-verydeep-19.mat")

sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = compute_content_cost(a_C, a_G)

sess.run(model['input'].assign(style_image))
J_style = compute_style_cost(model, STYLE_LAYERS)

J = total_cost(J_content, J_style)

optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(J)


def model_nn(sess, input_image, num_iterations = 1000):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):

        sess.run(train_step)
       
        generated_image = sess.run(model['input'])
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
        
            save_image("output/" + str(i) + ".png", generated_image)
            imshow(scipy.misc.imread("output/" + str(i) + ".png"))
            
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image

model_nn(sess, generated_image)
print("hehe")

import tensorflow as tf

def sqaure(x):
    return x*x

def double(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = square(x)
                
    grad = tape.gradient(y,x)
    return grad.numpy()

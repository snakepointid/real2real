import tensorflow as tf

def layout_variables():
                for var in tf.global_variables():
                        print(var.name)   

def layout_trainable_variables():
        for v in tf.trainable_variables():
                print(v.name) 
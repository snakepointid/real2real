import tensorflow as tf

def layout_variables(info_detail=False):
	for var in tf.global_variables():
		if info_detail:
                        print(v)
                else:
                        print(v.name)
def layout_trainable_variables(info_detail=True):
        for v in tf.trainable_variables():
                if info_detail:
                        print(v)
                else:
                        print(v.name) 

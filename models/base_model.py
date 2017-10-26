from __future__ import print_function
import tensorflow as tf
import six
import abc
from real2real.app.params import *
from real2real.utils.shape_ops import label_smoothing

@six.add_metaclass(abc.ABCMeta)
class baseModel(object):
        def __init__(self, is_training=True):
                self.graph = tf.Graph()
                self.is_training = is_training
                with self.graph.as_default():
                	self.is_dropout = tf.placeholder_with_default(False, shape=(), name='is_dropout')
                        self._build_()
                        self._metrics_()
                        if self.is_training:
                                self._cost_()
                                self._optimize_() 
                        self._save_()
                     
        @abc.abstractmethod
        def _build_(self):
                raise NotImplementedError

        @abc.abstractmethod
        def _metrics_(self):
                raise NotImplementedError

        @abc.abstractmethod
        def _cost_(self):
                raise NotImplementedError

       	def _optimize_(self):
		# Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=baseModelParams.learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                self.init_op  = tf.global_variables_initializer()
                # Summary 
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all() 

        def _save_(self):
                self.global_saver = tf.train.Saver()

                for variable in tf.trainable_variables():
                        if variable.name=="chinese/token/lookup_table:0":
                                with tf.variable_scope("",reuse=True):
                                        zh_embed = tf.get_variable("chinese/token/lookup_table")   
                                self.zh_embed_saver = tf.train.Saver([zh_embed])

                if variable.name=="english/token/lookup_table:0":
                                with tf.variable_scope("",reuse=True):
                                        en_embed = tf.get_variable("english/token/lookup_table")
                                self.en_embed_saver = tf.train.Saver([en_embed])


                        
@six.add_metaclass(abc.ABCMeta)
class multiClsModel(baseModel):
        @abc.abstractmethod
        def _build_(self):
                raise NotImplementedError
                
        def _metrics_(self):
                self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
                if multiClsModelParams.loss_softmax:
                        self.probs = tf.nn.softmax(self.logits)
                else:
                        self.probs = tf.nn.sigmoid(self.logits)
                self.acc = tf.reduce_mean(tf.to_float(tf.equal(self.preds, self.target)))
                # save log
                tf.summary.scalar('acc', self.acc)

        def _cost_(self):
                if multiClsModelParams.loss_softmax:
                        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.target,[-1,1]))
                else:
                        logits = tf.reshape(self.logits,[-1,1])
                        if self.target.dtype.is_integer:
                                labels = tf.one_hot(indices=tf.reshape(self.target,[-1,1]),depth=tf.shape(self.logits)[1])
                                labels = tf.reshape(labels,[-1,1])
                        else:
                                labels = self.target
                        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
                self.mean_loss = tf.reduce_mean(self.loss)

@six.add_metaclass(abc.ABCMeta)
class regressModel(baseModel):
        @abc.abstractmethod
        def _build_(self):
                raise NotImplementedError

        def _metrics_(self):
                self.mae = tf.reduce_mean(tf.abs(self.logits-self.target))
                # save log
                tf.summary.scalar('mae', self.mae)

        def _cost_(self):
                self.loss = tf.square(tf.subtract(self.logits,self.target))
                self.mean_loss = tf.reduce_mean(self.loss)
                if regressModelParams.loss_rmse:
                        self.mean_loss = tf.sqrt(self.mean_loss)

@six.add_metaclass(abc.ABCMeta)
class embedModel(multiClsModel):
        @abc.abstractmethod
        def _build_(self):
                raise NotImplementedError

        def _metrics_(self):
                self.mae = tf.reduce_mean(tf.abs(self.logits-self.target))
                # save log
                tf.summary.scalar('mae', self.mae)

        def _cost_(self):
                self.loss = tf.square(tf.subtract(self.logits,self.target))
                self.mean_loss = tf.reduce_mean(self.loss)
                if regressModelParams.loss_rmse:
                        self.mean_loss = tf.sqrt(self.mean_loss)
         

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
                self.token_embed_savers ={}
                for variable in tf.global_variables():
                        for language in ['chinese','english']:
                                if variable.name=="%s/token/lookup_table:0"%language:
                                        with tf.variable_scope(language,reuse=True):
                                                token_embed = tf.get_variable("token/lookup_table")   
                                        self.token_embed_savers[language] = tf.train.Saver([token_embed])
                        
@six.add_metaclass(abc.ABCMeta)
class multiClsModel(baseModel):
        @abc.abstractmethod
        def _build_(self):
                raise NotImplementedError
                
        def _metrics_(self):
                self.preds = tf.to_int64(tf.arg_max(self.logits, dimension=-1))
                if multiClsModelParams.loss_softmax:
                        self.probs = tf.nn.softmax(self.logits)
                else:
                        self.probs = tf.nn.sigmoid(self.logits)
                self.acc = tf.reduce_mean(tf.to_float(tf.equal(self.preds, self.target)))
                # save log
                tf.summary.scalar('acc', self.acc)

        def _cost_(self):
                if multiClsModelParams.loss_softmax:
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.target,[-1,1]))
                else:
                        logits = tf.reshape(self.logits,[-1,1])
                        if self.target.dtype.is_integer:
                                labels = tf.one_hot(indices=tf.reshape(self.target,[-1,1]),depth=tf.shape(self.logits)[1])
                                labels = tf.reshape(labels,[-1,1])
                        else:
                                labels = self.target
                        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
                self.mean_loss = tf.reduce_mean(loss)

@six.add_metaclass(abc.ABCMeta)
class binomialModel(baseModel):
        @abc.abstractmethod
        def _build_(self):
                raise NotImplementedError
                
        def _metrics_(self):
                self.loss = self.mean_loss
                # save log
                tf.summary.scalar('loss', self.loss)

        def _cost_(self):           
                loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.target)
                self.mean_loss = tf.reduce_mean(loss)                

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
                loss = tf.square(tf.subtract(self.logits,self.target))
                self.mean_loss = tf.reduce_mean(loss)
                if regressModelParams.loss_rmse:
                        self.mean_loss = tf.sqrt(self.mean_loss)

 

 
 
         

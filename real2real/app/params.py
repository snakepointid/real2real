class baseModelParams:
		model_mode = 'train'
		test_rate  = 0.02
		batch_size = 100
		dropout_rate  = 0.5
		learning_rate = 0.0001
		num_epochs = 20
class nlpModelParams(baseModelParams):
		flag_sinusoid = True     
		flag_position_embed = True

class multiClsModelParams(baseModelParams):
   		flag_label_smooth = True
		target_vocab_size = 100000

class regressModelParams(baseModelParams):
        	loss_rmse = True        
class pairEmbedModelParms(multiClsModelParams):
			vocab_size = 100000
			embedding_dim = 128
			activation_fn  = "tensorflow.nn.relu"
			mlp_layers  = 2
			hidden_units  = 128
			window_span = 5
class nmtParams(nlpModelParams,multiClsModelParams):
		source_maxlen = 30
		target_maxlen = 30 
		dropout_rate = 0.1
		source_vocab_size = 10000

class convLayerParams:
		kernel_size    = 5
		filter_nums    = 512
		dropout_rate   = 0.5
		conv_layer_num = 3
		strip_step     = 2
		activation_fn  = "tensorflow.nn.relu"

class convRankParams(nlpModelParams,regressModelParams,convLayerParams):
		source_vocab_size = 10000
		tag_size = 130000
		source_maxlen = 40
<<<<<<< HEAD
		embedding_dim = 128
		hidden_units  = 128
=======
		embedding_dim = 512
		hidden_units  = 512
>>>>>>> a6e4f631e69500c13fce160239425a687eb104f3
		mlp_layers  = 2


class directLayerParams:
		dropout_rate = 0.5

class attentionLayerParams:
		num_blocks   = 6
		num_heads    = 8
		dropout_rate = 0.5
		hidden_units = 512

class transformerParams(nmtParams,attentionLayerParams):
		pass


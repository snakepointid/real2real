class baseModelParams:
		test_rate  = 0.005
		batch_size = 100
		dropout_rate  = 0.1
		learning_rate = 0.0001
		num_epochs = 2000

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
		source_vocab_size = 10000

class convLayerParams(baseModelParams):
		filter_nums    = 128
		activation_fn  = "tensorflow.nn.relu"

class convRankParams(nlpModelParams,regressModelParams,convLayerParams):
		source_vocab_size = 10000
		tag_size = 130000
		source_maxlen = 40
		embedding_dim = 128
		hidden_units  = 128
		mlp_layers  = 2

class convClsParams(nlpModelParams,multiClsModelParams,convLayerParams):
		source_vocab_size = 10000
		target_vocab_size = 35
		title_maxlen = 30
		content_maxlen = 3000
		embedding_dim = 128
		hidden_units  = 128
		mlp_layers  = 2


class directLayerParams(baseModelParams):
		pass

class attentionLayerParams(baseModelParams):
		num_blocks   = 6
		num_heads    = 8
		hidden_units = 128

class transformerParams(nmtParams,attentionLayerParams):
		pass


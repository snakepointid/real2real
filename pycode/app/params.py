
class baseModelParams:
		model_mode = 'train'
		test_rate  = 0.02
		batch_size = 20000
		dropout_rate  = 0.5
		learning_rate = 0.0001
		num_epochs = 20
		train_model = 'convRank'
class nlpModelParams(baseModelParams):
		flag_sinusoid = True     
		flag_position_embed = True

class multiClsModelParams(baseModelParams):
        flag_label_smooth  = True

class regressModelParams(baseModelParams):
        loss_rmse = True        

class nmtParams(nlpModelParams,multiClsModelParams):
		source_maxlen = 30
		target_maxlen = 30 

		target_vocab_size = 10000
		source_vocab_size = 30000

class convLayerParams:
		kernel_size    = 3
		filter_nums    = 128
		dropout_rate   = 0.5
		conv_layer_num = 3
		strip_step     = 2
		activation_fn  = "tensorflow.nn.relu"

class convRankParams(nlpModelParams,regressModelParams,convLayerParams):
		source_vocab_size = 10000
		tag_size = 130000
		source_maxlen = 30
		embedding_dim = 128
		hidden_units  = 128
		mlp_layers  = 2


class directLayerParams:
		dropout_rate = 0.5

class attentionLayerParams:
		num_blocks   = 3
		num_heads    = 8
		dropout_rate = 0.5
		hidden_units = 128

class transformerParams(nmtParams,attentionLayerParams):
		pass


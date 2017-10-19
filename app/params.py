class baseLayerParams:
		dropout_rate  = 0.5
		activation_fn ="tensorflow.nn.relu"

class baseModelParams(baseLayerParams):
		test_rate  = 0.005
		batch_size = 100
		learning_rate = 0.0001
		num_epochs = 2000

class embedLayerParams(baseModelParams):
		flag_sinusoid = True     
		flag_position_embed = True

class multiClsModelParams(baseModelParams):
   		flag_label_smooth = True
		target_vocab_size = 100000

class regressModelParams(baseModelParams):
        loss_rmse = True     
   
class convLayerParams(baseLayerParams):
		filter_nums    = 128
	
class ctrRankModelParams(embedLayerParams,regressModelParams,convLayerParams):
		source_vocab_size = 10000
		tag_size = 130000
		source_maxlen = 40
		embedding_dim = 128
		hidden_units  = 128
		mlp_layers  = 2

class newsClsModelParams(embedLayerParams,multiClsModelParams,convLayerParams):
		source_vocab_size = 10000
		target_vocab_size = 35
		title_maxlen = 30
		content_maxlen = 3000
		embedding_dim = 128
		hidden_units  = 128
		mlp_layers  = 2



 


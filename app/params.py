class baseLayerParams:
        dropout_rate  = 0.5
        activation_fn ="tensorflow.nn.relu"
        zero_pad=True
        scale=True
        direct_cont=True
        norm=True
		
class convLayerParams(baseLayerParams):
        filter_nums  = 128
        token_cnn_params = [3,1,1]
        sentence_cnn_params = [1,1,1]

class fullLayerParams(baseLayerParams):
        hidden_units  = 128
        mlp_layers = 0

class attentionLayerParams(baseLayerParams):
        pass

class embedLayerParams(baseLayerParams):
        flag_sinusoid = True     
        flag_position_embed = True
        embedding_dim = 128

class baseModelParams:
        test_rate  = 0.005
        batch_size = 100
        learning_rate = 0.0001
        num_epochs = 2000

class multiClsModelParams(baseModelParams):
        flag_label_smooth = True
        target_vocab_size = 35
        loss_softmax=True
        
class regressModelParams(baseModelParams):
        loss_rmse = True     
   
class ctrRankModelParams(embedLayerParams,fullLayerParams,convLayerParams,regressModelParams):
        source_vocab_size = 10000
        tag_size = 130000
        source_maxlen = 40

class newsClsModelParams(embedLayerParams,fullLayerParams,convLayerParams,attentionLayerParams,multiClsModelParams):
        source_vocab_size = 10000
        target_vocab_size = 35
        title_maxlen = 30
        content_maxlen = 3000
        mode = 'title'

class appParams(baseModelParams):
	newsClsModel = 1


 


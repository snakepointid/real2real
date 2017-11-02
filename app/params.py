class baseLayerParams:
        dropout_rate  = 0.5
        activation_fn ="tensorflow.nn.relu"
        zero_pad=True
        scale=True
        direct_cont=True
        norm=True
        hidden_units  = 128
		
class convLayerParams(baseLayerParams):
        filter_nums  = 128

class fullLayerParams(baseLayerParams):
        mlp_layers = 1

class attentionLayerParams(baseLayerParams):
        pass
class rnnLayerParams(baseLayerParams):
        pass

class embedLayerParams(baseLayerParams):      
        embedding_dim = 128
        source_vocab_size = 10000

class textModuleParams(baseLayerParams):
        pass

class entityEmbedModuleParams(embedLayerParams):
        sinusoid = True
        position_embed = False
        
class fullConnectModuleParams(fullLayerParams):
        input_reshape=True

class baseModelParams:
        test_rate  = 0.005
        batch_size = 100
        learning_rate = 0.0001
        num_epochs = 2000

class multiClsModelParams(baseModelParams):
        target_label_num = 35
        loss_softmax=False
        
class regressModelParams(baseModelParams):
        loss_rmse = True     
   
class ctrRankModelParams(embedLayerParams,fullLayerParams,convLayerParams,regressModelParams):
        tag_size = 130000
        title_maxlen = 30
        title_cnn_params = [5,2,1]

class newsClsModelParams(embedLayerParams,fullLayerParams,convLayerParams,attentionLayerParams,multiClsModelParams):
        title_maxlen = 30
        content_maxlen = 3000
        title_cnn_params = [4,2,1] #kernel,stride,layers
        content_cnn_params = [5,2,3]
        final_layer = "title"
	text_encode_mode = 'CRA'

class nmtModelParams(embedLayerParams,multiClsModelParams):
        language="chinese"
        target_label_num = 100000
        source_cnn_params = [4,2,1] #kernel,stride,layers
        source_maxlen = 30
        target_maxlen = 30
class tokenEmbedModelParams(embedLayerParams,multiClsModelParams):
        source_vocab_size=60000
        language='english'
        target_label_num = 8
class appParams(baseModelParams):
	pass

 


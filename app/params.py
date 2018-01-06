class baseLayerParams:
        dropout_rate  = 0.5
        activation_fn ="tensorflow.nn.tanh"
        zero_pad=True
        scale=True
        direct_cont=False
        norm=True
        hidden_units  = 256
        
class convLayerParams(baseLayerParams):
        filter_nums  = 256

class fullLayerParams(baseLayerParams):
        mlp_layers = 0

class attentionLayerParams(baseLayerParams):
        pass
class rnnLayerParams(baseLayerParams):
        pass
class embedLayerParams(baseLayerParams):      
        embedding_dim = 20
        source_vocab_size = 10000

class textModuleParams(baseLayerParams):
        layers="CA"

class keywordModuleParams(baseLayerParams):
        types=1

class entityEmbedModuleParams(embedLayerParams):
        sinusoid = True
        position_embed = False

class fullConnectModuleParams(fullLayerParams):
        input_reshape=True

class baseModelParams:
        test_rate  = 0.01
        batch_size = 10000
        learning_rate = 0.001
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

class keywordModelParams(embedLayerParams,fullLayerParams,convLayerParams,regressModelParams):
        target_label_num = 35
        title_maxlen = 30
        source_vocab_size = 20*10000
	batch_size = 10000
        label_context=1

class newsClsModelParams(embedLayerParams,fullLayerParams,convLayerParams,attentionLayerParams,multiClsModelParams):
        title_maxlen = 30
        content_maxlen = 3000
        title_cnn_params = [4,2,1] #kernel,stride,layers
        content_cnn_params = [5,2,3]
        final_layer = "title"
	text_encode_mode = 'CRA'

class tokenEmbedModelParams(embedLayerParams,multiClsModelParams):
        test_rate = 0.001
        source_vocab_size=20*10000
        language='chinese'
        target_label_num = 7
        topk = 10
        batch_size = 10000
class LDAModelParams(embedLayerParams,multiClsModelParams):
        user_num=50000
        tag_num=10000

class appParams(baseModelParams):
	pass


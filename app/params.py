class baseLayerParams:
        dropout_rate  = 0.5
        activation_fn ="tensorflow.nn.relu"
        zero_pad=True
        scale=True
        direct_cont=True
        norm=True
		
class convLayerParams(baseLayerParams):
        filter_nums  = 128

class fullLayerParams(baseLayerParams):
        hidden_units  = 128
        mlp_layers = 0

class attentionLayerParams(baseLayerParams):
        pass

class embedLayerParams(baseLayerParams):      
        embedding_dim = 128
        source_vocab_size = 10000

class textModuleParams:
        stride_cnn = True
        target_atten = True

class entityEmbedModuleParams(embedLayerParams):
        flag_sinusoid = True     
        flag_position_embed = True

class baseModelParams:
        test_rate  = 0.005
        batch_size = 100
        learning_rate = 0.0001
        num_epochs = 2000

class multiClsModelParams(baseModelParams):
        flag_label_smooth = True
        target_label_num = 35
        loss_softmax=True
        
class regressModelParams(baseModelParams):
        loss_rmse = True     
   
class ctrRankModelParams(embedLayerParams,fullLayerParams,convLayerParams,regressModelParams):
        tag_size = 130000
        title_maxlen = 30
        test_rate  = 0.05
        title_cnn_params = [5,2,1]


class newsClsModelParams(embedLayerParams,fullLayerParams,convLayerParams,attentionLayerParams,multiClsModelParams):
        title_maxlen = 30
        content_maxlen = 3000
        title_cnn_params = [3,1,1] #kernel,stride,layers
        content_cnn_params = [5,2,3]
        final_layer = "title"
 


 


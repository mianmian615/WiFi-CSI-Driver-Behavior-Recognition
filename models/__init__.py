
#from .ResNet1D import *
from .ResNet1D_group import *
#from .ResNet2D import *
from .ResNet2Dnew import *
from .backbones import *
from .Sensefi_model_old import *
from .InceptionTime import *
from .resnet1d_rfid import *
from .resnet1d import *
from .resnet2d import *

def get_backbones_class(dataset, backbones_name):
    """
    返回一个可调用的工厂函数，调用时传入input_shape, hparams即可返回实例。
    支持RNN/GRU/LSTM/BiLSTM等特殊工厂。
    """
    # if backbones_name in ['RNN', 'GRU', 'LSTM', 'BiLSTM']:
    #     def factory(input_shape, hparams):
    #         return RNN_Factory(input_shape, hparams, rnn_type=backbones_name)
        # return factory
    if backbones_name in ['RNN2', 'GRU2', 'LSTM2', 'BiLSTM2', 'ABLSTM']:
        def factory2(input_shape, hparams):
            return RNN_Factory2(input_shape, hparams, rnn_type=backbones_name)
        return factory2
    if backbones_name in globals():
        return globals()[backbones_name]
    raise NotImplementedError(f"Backbones not found: {backbones_name}")
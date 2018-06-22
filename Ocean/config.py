import warnings
import os

class DefaultConfig(object):
    env = 'Ocean_1'  # visdom 环境
    model = 'DNN'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = os.path.join('data', 'beautiful.mat')
    test_data_root = os.path.join('data', 'beautiful.mat')
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 45  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 5  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 1000
    lr = 0.05  # initial learning rate
    lr_decay = 0.98  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0.0003  # 损失函数


def parse(self, kwargs):
    """
    根据字典kwargs 更新 config参数
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()

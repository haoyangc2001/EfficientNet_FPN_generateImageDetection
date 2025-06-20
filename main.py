import os
import json
import argparse
import sys
import torch 
from tensorboardX import SummaryWriter
import time
from utils.config import Config
from utils.common import load_config, get_train_paths, setup_seed
from utils.logger import create_logger
from data.dataset import BaseData
from train.trainer_progressive import PGTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--config_name', type=str, help='model configuration file', default='progressive')
    parser.add_argument('--data', type=str, default='homework')
    parser.add_argument('--mode', type=str, default='POSE')
    parser.add_argument('--device', default='cuda:0', type=str, help='cuda:n or cpu')
    parser.add_argument('--log_name', default=None, type=str, help='log file name')
    parser.add_argument('--debug', action='store_true', help="debug", default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # load configs
    opt = parse_args()
    config = load_config('configs.{}'.format(opt.config_name))
    config_attr = dir(config)           # 获取 config 对象的所有属性和方法
    config_params = {config_attr[i]: getattr(config, config_attr[i]) for i in range(len(config_attr)) if config_attr[i][:2] != '__'}

    # setup random seed
    setup_seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # setup run_dir
    time_str = time.strftime('%Y-%m-%d-%H-%M')    
    month_day = time_str.split('-')[1]+time_str.split('-')[2]
    run_dir = '{}/{}{}'.format(month_day, opt.mode, '_seed{}'.format(config.seed))

    # setup data path
    script_dir = os.path.dirname(os.path.abspath(__file__))                  # 获取当前脚本所在目录
    config_path = os.path.join(script_dir, 'configs', 'data_list.yaml')      # 构建指向配置文件的绝对路径
    data_list = Config(config_filepath=config_path)[opt.data]                # 读取配置文件，并获取数据列表
    model_dir, train_data_path, val_data_path = get_train_paths(script_dir, data_list, opt.config_name, run_dir)
    test_data_path = os.path.join(script_dir, data_list['test_data_path']).replace('./','')
    out_data_path = os.path.join(script_dir, data_list['out_data_path']).replace('./','')
    config.known_classes = data_list['known_classes']
    config.unknown_classes1, config.unknown_classes2 = data_list['unknown_classes1'], data_list['unknown_classes2']
    config.unknown_classes = config.unknown_classes1 + config.unknown_classes2 
    config.class_num = len(config.known_classes)  
    print('config.class_num', config.class_num)   

    # setup logs 
    os.makedirs(model_dir,exist_ok=True)
    writer = SummaryWriter(logdir=model_dir)
    logger = create_logger(model_dir, log_name=opt.log_name)    
    logger.info('model dir: %s' % model_dir)

    # save configs
    options_file = os.path.join(model_dir, 'options.json')
    with open(options_file, 'w') as fp:
        json.dump(vars(opt), fp, indent=4)
    config_file = os.path.join(model_dir, 'configs.json')
    with open(config_file, 'w') as fp:
        json.dump(config_params, fp, indent=4)
    logger.info('options: %s',opt)
    logger.info('config_params: %s',config_params)
    
    # setup data
    Data = BaseData(script_dir, train_data_path, val_data_path, 
            test_data_path, out_data_path, 
            opt, config)
    train_loader, test_loader, out_loader = Data.train_loader, Data.test_loader, Data.out_loader
    # train_loader, val_loader, test_loader, out_loader = Data.train_loader, Data.val_loader, Data.test_loader, Data.out_loader
    # out_loader1, out_loader2, out_loader3 = Data.out_loader1, Data.out_loader2, Data.out_loader3

    # setup trainer
    Trainer = PGTrainer(Data, device, config, opt, writer, logger, model_dir)   

    # begin to train
    start_epoch = 0
    logger.info("begin to train!")
    augnets = []
    for epoch in range(config.max_epochs):
        if opt.mode == 'baseline':
            Trainer.train_epoch_baseline(epoch)
        elif opt.mode == 'POSE':
            augnet = Trainer.train_epoch_POSE(augnets, epoch)
            augnets.append(augnet)
        else:
            logger.info('not defined mode')

        # # val-set evaluation
        # val_perf = Trainer.predict_set(val_loader, run_type='val')[-1]
        # logger.info('epoch %d -> metric %s, val: %.4f ' % (epoch, config.metric, val_perf))

        # closed-set and open-set evaluation
        if (epoch+1) % config.test_interval == 0: 
            logger.info('----------------------------  testing begin ----------------------------  ') 
            if len(augnets) > 0:
                Trainer.tsne_augnet(epoch, augnets, Data.tsne_loader, run_type='tsne_augnet')
            
            feature_known, _labels_k, _pred_k, test_perf = Trainer.predict_set(test_loader, run_type='closed-set')
            out_perf, oscr_perf = Trainer.test_out(epoch, feature_known, _labels_k, _pred_k, out_loader, config.unknown_classes, 'out')
            # out_perf1, oscr_perf1 = Trainer.test_out(epoch, feature_known, _labels_k, _pred_k, out_loader1, config.unknown_classes1, 'out_seed')
            # out_perf2, oscr_perf2 = Trainer.test_out(epoch, feature_known, _labels_k, _pred_k, out_loader2, config.unknown_classes2, 'out_arch')
            # out_perf3, oscr_perf3 = Trainer.test_out(epoch, feature_known, _labels_k, _pred_k, out_loader3, config.unknown_classes3, 'out_data')
            # logger.info('epoch %d -> metric %s, closed-set: %.2f, unseen seed: %.2f, %.2f, unseen arch: %.2f, %.2f, unseeen dataset: %.2f, %.2f, unseen all: %.2f, %.2f' % 
            #                 (epoch, config.metric, test_perf, out_perf1, oscr_perf1, out_perf2, oscr_perf2, out_perf3, oscr_perf3, out_perf, oscr_perf))
            logger.info('----------------------------  testing end ----------------------------  ') 

            if (epoch+1) % config.save_interval == 0: 
                save_suffix = 'model_{}_test{}_{}_AUC_{}_OSCR_{}.pth'.format(
                                epoch,
                                test_perf,
                                config.metric,
                                out_perf,
                                oscr_perf)
                Trainer.save_model(epoch, save_suffix)




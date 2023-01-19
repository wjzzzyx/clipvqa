# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         main
# Description:  the entrance of procedure
#-------------------------------------------------------------------------------

import argparse
import os
import torch
from torch.utils.data import DataLoader

from config import cfg, update_config
from dataset import dataset_RAD, dataset_SLAKE
from language.classify_question import classify_model
from model import ClipTransEncoder, ClipBAN, ClipTransEncoderV2
from utils.create_dictionary import Dictionary
from train import train
from test import test


def parse_args():
    parser = argparse.ArgumentParser(description="Med VQA")
    # cfg
    parser.add_argument(
            "--cfg",
            help="decide which cfg to use",
            required=False,
            default="/home/test.yaml",
            type=str,
            )
    # GPU config
    parser.add_argument('--gpu', type=int, default=0,
                        help='use gpu device. default:0')
    parser.add_argument('--test', type=bool, default=False,
                        help='Test or train.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.cuda.empty_cache()

    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    update_config(cfg, args)
    data_dir = cfg.DATASET.DATA_DIR
    args.data_dir = data_dir
    # Fixed random seed
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)
    d = Dictionary.load_from_file(data_dir + '/dictionary.pkl')
    if cfg.DATASET.DATASET == "RAD":
        train_dataset = dataset_RAD.VQARADFeatureDataset('train', cfg,d,dataroot=data_dir)
        val_dataset = dataset_RAD.VQARADFeatureDataset('test', cfg,d,dataroot=data_dir)
    elif cfg.DATASET.DATASET == "SLAKE":
        train_dataset = dataset_SLAKE.VQASLAKEFeatureDataset('train', cfg,d,dataroot=data_dir)
        val_dataset = dataset_SLAKE.VQASLAKEFeatureDataset('test', cfg,d,dataroot=data_dir)
    else:
        raise ValueError(f"Dataset {cfg.DATASET.DATASET} is not supported!")
    
    train_loader = DataLoader(train_dataset, cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False,pin_memory=True)

    # load the model
    glove_weights_path = os.path.join(data_dir, "glove6b_init_300d.npy")
    question_classify = classify_model(d.ntoken, glove_weights_path)
    if cfg.DATASET.DATASET == "SLAKE":
        ckpt = './saved_models/type_classifier_slake.pth'
        pretrained_model = torch.load(ckpt, map_location='cuda:0')['model_state']
    else:
        ckpt = './saved_models/type_classifier.pth'
        qtype_ckpt = './saved_models/qtype_classifier.pth'
        pretrained_model = torch.load(ckpt, map_location='cuda:0')
    question_classify.load_state_dict(pretrained_model)

    # training phase
    # create VQA model and question classify model
    if args.test:
        if cfg.MODEL == 'cliptransencoder':
            model = ClipTransEncoder(val_dataset, cfg)
        elif cfg.MODEL == 'cliptransencoderv2':
            model = ClipTransEncoderV2(val_dataset, cfg)
        elif cfg.MODEL == 'clipban':
            model = ClipBAN(val_dataset, cfg)
        model_data = torch.load(cfg.TEST.MODEL_FILE)
        model.load_state_dict(model_data.get('model_state', model_data), strict=False)
        test(cfg, model, question_classify, val_loader, train_dataset.num_close_candidates, args.device)
    else:
        if cfg.MODEL == 'cliptransencoder':
            model = ClipTransEncoder(train_dataset, cfg)
        elif cfg.MODEL == 'cliptransencoderv2':
            model = ClipTransEncoderV2(train_dataset, cfg)
        elif cfg.MODEL == 'clipban':
            model = ClipBAN(train_dataset, cfg)
        train(cfg, model, question_classify, train_loader, val_loader, train_dataset.num_close_candidates, args.device)

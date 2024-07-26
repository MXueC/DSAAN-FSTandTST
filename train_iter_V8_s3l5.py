#!/usr/bin/python3
#coding=utf-8

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from Dataset_V2_method4 import DatasetTest, DatasetTrain
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from mynet_V8 import  mynet
from argparse import ArgumentParser

import torch
import random
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def setup_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True





def train(model_name,batch_size=8, num_workers=2, EPOCH=2, short_len=3,long_len=5, RESUME=True, path_checkpoint="./checkpoint/ckpt_best_0.pth",save_epoch = 1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # print("using {} device.".format(device))
    # train data
    transform_train = transforms.Compose([transforms.Resize([112, 112]),  # 原始vgg使用224*224的图片，现在缩小了7倍
                                          transforms.ToTensor()  # 将图片进行归一化，并把数据转换成Tensor类型
                                          ])
    # validate test
    transform_test = transforms.Compose([transforms.Resize([112, 112]),
                                         transforms.ToTensor()  # 将图片进行归一化，并把数据转换成Tensor类型
                                         ])
    # model

    model = mynet(short_len=short_len,long_len=long_len)
    model.to(device)
    loss_function_gray = nn.CrossEntropyLoss() # BCEFocalLoss() #
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    if RESUME:
        pass
    else:
        print("load dataset")
        train_dataset = DatasetTrain("./train_label_compare.txt", seq_len=long_len, transform=transform_train)
        # train_dataset = DatasetTest("./test_2250.txt", seq_len=long_len, transform=transform_test)
        # DatasetTrain("./train_label.txt", seq_len=long_len, transform=transform_train)
        train_num = len(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)

        val_dataset = DatasetTest("./test_label_compare.txt", seq_len=long_len, transform=transform_test)
        val_num = len(val_dataset)
        validate_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        print("using {} images for training, {} images for validation.".format(train_num, val_num))

        # 训练epoch
        results = pd.DataFrame()
        best_acc = 0.0

        val_acc_lt = []
        train_acc_lt = []
        best_acc_lt = []
        loss_lt = []
        precision_lt = []
        recall_lt = []
        f1_lt = []

        train_steps = len(train_loader)
        for epoch in range(EPOCH):
            # train
            model.train()
            train_bar = tqdm(train_loader)
            running_loss_gray = 0.0  #########################

            for step, data in enumerate(train_bar):
                # for step,(images,labels) in enumerate(train_loader):
                grays, labels = data
                optimizer.zero_grad()
                outputs_gray,out1,out2 = model(grays.to(device))
                labels = labels.to(device)
                loss_gray = 0.6*loss_function_gray(outputs_gray,labels)+ 0.2*loss_function_gray(out1, labels) +0.2*loss_function_gray(out2, labels)
                loss_gray.backward()
                optimizer.step()
                # print statistics
                running_loss_gray += loss_gray.item()
                train_bar.desc = "train epoch[{}/{}] loss_total:{:.3f}  ".format(epoch + 1, EPOCH, running_loss_gray)
                # vis.image(running_loss_gray, f'loss (epoch: {epoch}, step: {step})')

            # train acc
            model.eval()
            acc = 0.0  # accumulate accurate number / epoch
            train_bar = tqdm(train_loader)
            with torch.no_grad():
                for iter, data in enumerate(train_bar):
                    val_grays, val_labels = data
                    outputs_gray,out1,out2 = model(val_grays.to(device))
                    predict_y_gray = torch.max(outputs_gray, dim=1)[1]
                    # print(predict_y_gray)
                    acc += torch.eq(predict_y_gray, val_labels.to(device)).sum().item()

            train_accurate = acc / train_num
            train_acc_lt.append(train_accurate)
            loss_lt.append(running_loss_gray)
            # train_acc_lt.append(train_accurate)
            print('[epoch %d] train_loss: %.3f  train_accuracy: %.4f  iter:%d/%d' %
                  (epoch + 1, (running_loss_gray) / train_steps, train_accurate, iter,
                   len(train_dataset) // batch_size))  ################
            print("train:{} ，acc:{}".format(len(train_dataset),acc))

            # 验证
            model.eval()
            acc = 0.0  # accumulate accurate number / epoch
            TP,FP,TN,FN = 0,0,0,0
            with torch.no_grad():
                for iter, (val_grays, val_labels) in enumerate(validate_loader):
                    # val_images, val_labels = val_data
                    outputs_gray,out1,out2 = model(val_grays.to(device))
                    predict_y_gray = torch.max(outputs_gray, dim=1)[1]
                    # print(predict_y_gray)
                    acc += torch.eq(predict_y_gray, val_labels.to(device)).sum().item()
                    TN  += ((val_labels.to(device) == 0) & (predict_y_gray == 0)).sum().item()
                    TP  += ((val_labels.to(device) == 1) & (predict_y_gray == 1)).sum().item()
                    FP  += ((val_labels.to(device) == 0) & (predict_y_gray == 1)).sum().item()
                    FN  += ((val_labels.to(device) == 1) & (predict_y_gray == 0)).sum().item()

            val_accurate = acc / val_num
            try:
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * precision * recall / (precision + recall)
            except:
                precision = TP / (TP + FP+1)
                recall = TP / (TP + FN+1)
                f1 = 2 * precision * recall / (precision + recall+1)

            precision_lt.append(precision)
            recall_lt.append(recall)
            f1_lt.append(f1)
            val_acc_lt.append(val_accurate)
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.4f  iter:%d/%d' %
                  (epoch + 1, (running_loss_gray) / train_steps, val_accurate, iter,
                   len(val_dataset) // batch_size))  ################
            print("TP:{},TN:{},FP:{},FN:{}".format(TP,TN,FP,FN))
            print("validate:{}, acc:{}".format(len(val_dataset), acc))

            if val_accurate>=best_acc:
                best_acc = val_accurate
                checkpoint = {
                    "net": model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch
                    }
                if not os.path.isdir("./checkpoint"):
                    os.mkdir("./checkpoint")
                if not os.path.isdir("./results"):
                    os.mkdir("./results")
                torch.save(checkpoint, './checkpoint/ckpt_%s_%s_V7.pth' % (str(epoch),str(best_acc)))
            best_acc_lt.append(best_acc)
        results["train_acc"] = train_acc_lt
        results["val_acc"] = val_acc_lt
        results["loss"] = loss_lt
        results["best_Acc"] = best_acc_lt
        results["precision"] = precision_lt
        results["recall"] = recall_lt
        results["f1_score"] = f1_lt
        results.to_csv("./results/{}.csv".format(model_name))
        print("model:{},best acc:{}".format(model_name,best_acc))
        print("Finished training")
        return best_acc


if __name__ == "__main__":
    setup_seed(2023)
    parser = ArgumentParser(description=" Process some parameters")
    parser.add_argument("--batch_size",default=8,type=int,help="this is the batch size of training samples")
    parser.add_argument("--EPOCH", default=3, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--long_len", default=5, type=int)
    parser.add_argument("--short_len", default=3, type=int)
    parser.add_argument("--RESUME", default=False, type=bool)
    parser.add_argument("--path_checkpoint", default="./checkpoint/", type=str)
    parser.add_argument("--save_epoch", default=1, type=int)
    parser.add_argument("--model_name",default="mynet")

    args = parser.parse_args()

    best_acc = train(model_name="mynetV8_s3l5",
                     batch_size=8,
                     num_workers=2,
                     EPOCH=20,
                     short_len=3,
                     long_len=5,
                     RESUME=False,
                     path_checkpoint="./checkpoint/ckpt_best_0.pth", save_epoch=args.save_epoch,
                     )
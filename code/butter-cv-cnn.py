import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os

from sklearn.metrics import roc_curve, roc_auc_score

from utils import *
from models import *

np.set_printoptions(suppress=True)  # not use scientific counting

torch.set_printoptions(
    linewidth=150,  # Maximum number of characters per line, default is 80
    sci_mode=False  # Display data using scientific counting, default is True
)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("GPU is available：", torch.cuda.is_available())
print("number of GPU：", torch.cuda.device_count())

EPOCH = 100
k_fold = 10
LR = 0.0001  # learning rate
validation_per_epoch = 10  # Verify every number of epochs
BATCH_SIZE = 32 * torch.cuda.device_count()
imgdir = "./img/cnn-butter0.2-cv20-volun2/"  # path of saving image of result
# volun[x], the x means the number of volunteers for training

if not os.path.exists(os.path.dirname(imgdir)): os.makedirs(os.path.dirname(imgdir))

# read csi data
data_xx = np.load("../temp/data_xx_1x3_butter0.2_cv20.npy")
data_yys = np.load("../temp/data_yys.npy")  # [volunteer ID, activity ID, repeat ID, the start time of the sliced item]
print("the data have been read, data_xx:", data_xx.shape, "data_yys:", data_yys.shape)

# select data of training and test set, only use the data while data_yy>=0
print("data select...")
data_yy = []
for i in range(0, len(data_yys)):
    if data_yys[i, 1] in [0, 1, 2]:
        y = 1
    elif data_yys[i, 1] in [3, 4, 5, 6, 10, 11, 12, 13, 14, 15] and data_yys[i, 3] in list(range(0, 120, 1)):
        y = 0
    else:
        y = -1
    data_yy.append(y)
data_yy = np.array(data_yy)
print(data_yy.shape)
print("y=0:", len(np.where(data_yy == 0)[0]))
print("y=1:", len(np.where(data_yy == 1)[0]))
print("y=-1:", len(np.where(data_yy == -1)[0]))

# Split the dataset by volunteer
volunteers = np.arange(1, 21)
np.random.seed(0)
np.random.shuffle(volunteers)

for k in range(0, k_fold):
    print("k=", k)

    # less data for test
    # test_volunteers=volunteers[0:int(len(volunteers)*(1/k_fold))]
    # train_volunteers=volunteers[int(len(volunteers)*(1/k_fold)):]
    # volunteers = np.concatenate((train_volunteers, test_volunteers))  # roll

    # less data for training
    train_volunteers = volunteers[0:int(len(volunteers) * (1 / k_fold))]
    test_volunteers = volunteers[int(len(volunteers) * (1 / k_fold)):]
    volunteers = np.concatenate((test_volunteers, train_volunteers))  # roll

    train_index = np.where(np.isin(data_yys[:, 0], train_volunteers) & (data_yy >= 0))[0]
    test_index = np.where(np.isin(data_yys[:, 0], test_volunteers) & (data_yy >= 0))[0]

    print("the volunteer of training set:", len(train_index), train_volunteers)
    print("the volunteer of test set:", len(test_index), test_volunteers)

    # training set
    data_x_train = torch.FloatTensor(data_xx[train_index]).reshape(len(train_index), 1, -1, data_xx.shape[-1])
    # data_x_train=data_x_train.reshape(data_x_train.size(0),data_x_train.size(2),3,data_x_train.size(-1)//3).transpose(2,1)
    data_y_train = torch.LongTensor(data_yy[train_index])
    data_ys_train = torch.LongTensor(data_yys[train_index])
    print("training set:", data_x_train.shape, data_y_train.shape)

    # test set
    data_x_test = torch.FloatTensor(data_xx[test_index]).reshape(len(test_index), 1, -1, data_xx.shape[-1])
    # data_x_test=data_x_test.reshape(data_x_test.size(0),data_x_test.size(2),3,data_x_test.size(-1)//3).transpose(2,1)
    data_y_test = torch.LongTensor(data_yy[test_index])
    data_ys_test = torch.LongTensor(data_yys[test_index])
    print("test set:", data_x_test.shape, data_y_test.shape)

    # TCS dataset (only the volunteers in test set)
    external_index = np.where((data_yys[:, 3] > 0) & (np.isin(data_yys[:, 1], np.arange(10, 16))) & (
        np.isin(data_yys[:, 0], test_volunteers)))[0]
    data_x_external = torch.FloatTensor(data_xx[external_index]).reshape(len(external_index), 1, -1, data_xx.shape[-1])
    # data_x_external=data_x_external.reshape(data_x_external.size(0),data_x_external.size(2),3,data_x_external.size(-1)//3).transpose(2,1)
    data_ys_external = torch.LongTensor(data_yys[external_index])
    print("TCS dataset:", data_x_external.shape, data_ys_external.shape)

    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(data_x_train, data_y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(data_x_test, data_y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )
    external_loader = Data.DataLoader(
        dataset=Data.TensorDataset(data_x_external, data_ys_external),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )

    # the data of process of training
    history = {
        "loss": [],
        "acc": [],
        "prec": [],
        "sens": [],
        "spec": [],
        "f1": [],
        "vali_loss": [],
        "vali_acc": [],
        "vali_prec": [],
        "vali_sens": [],
        "vali_spec": [],
        "vali_f1": [],
    }

    print("Initializing the model...")
    model = CNN()  # Initializing the model
    model.linear = nn.Linear(1280, 2)  # Modify the fully connected layer
    model = nn.DataParallel(model)  # Set up multiple Gpus
    model = model.cuda()
    # Adding optimization methods
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # optimizer=torch.optim.RMSprop(model.parameters(),lr=LR)
    # Specify the loss function using cross-information entropy
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.cuda()

    # Start training
    print("Start training...")
    for epoch in range(len(history["loss"]), EPOCH):
        model.train()  # Set train mode
        losses = []
        accuracy = []
        precision = []
        sensitivity = []
        specificity = []
        f1s = []
        for step, data in enumerate(train_loader):
            x, y = data
            x = x.cuda()
            y = y.cuda()

            output = model(x)  # predict
            loss = loss_fn(output, y)  # Calculating the loss
            optimizer.zero_grad()  # Before each iteration, reset the gradient to zero
            loss.backward()  # Back propagation
            optimizer.step()  # Gradient descent

            # Showing the training process
            y_pred = torch.max(output, 1)[1].data
            acc, prec, sens, spec, f1 = analyse2(y_pred.cpu(), y.cpu())
            accuracy.append(acc)
            sensitivity.append(sens)
            specificity.append(spec)
            precision.append(prec)
            f1s.append(f1)
            losses.append(loss.item())
            print("\repoch:%d,step:%d,loss:%.4f,acc:%.4f,prec:%.4f,sens:%.4f,spec:%.4f,f1:%.4f," % (
                epoch, step, loss.item(), acc, prec, sens, spec, f1), end="\t")
            torch.cuda.empty_cache()  # Clear cache of GPU to avoid cache overflow

        # recorded the historical data of each epoch
        loss = sum(losses) / len(losses)
        acc = sum(accuracy) / len(accuracy)
        prec = sum(precision) / len(precision)
        sens = sum(sensitivity) / len(sensitivity)
        spec = sum(specificity) / len(specificity)
        f1 = sum(f1s) / len(f1s)
        history["loss"].append(loss)
        history["acc"].append(acc.cpu())
        history["prec"].append(prec)
        history["sens"].append(sens)
        history["spec"].append(spec)
        history["f1"].append(f1)

        # Evaluation model
        if epoch % validation_per_epoch == 0:
            model.eval()  # set eval mode
            y_pred, y_real, vali_loss = get_all_preds(model, test_loader, loss_fn)
            # vali_acc=sum(y_pred==y_real).item()/y_real.size(0)
            vali_acc, vali_prec, vali_sens, vali_spec, vali_f1 = analyse2(y_pred, y_real)
            print("\repoch:%d,step:%d,loss:%.4f,acc:%.4f,prec:%.4f,sens:%.4f,spec:%.4f,f1:%.4f|v_loss:%.4f,"
                  "v_acc:%.4f,v_prec:%.4f,v_sens:%.4f,v_spec:%.4f,v_f1:%.4f" %
                  (epoch, step, loss, acc, prec, sens, spec, f1, vali_loss,
                   vali_acc, vali_prec, vali_sens, vali_spec, vali_f1))
            # Record historical data of validation
            history["vali_loss"].append(vali_loss)
            history["vali_acc"].append(vali_acc)
            history["vali_prec"].append(vali_prec)
            history["vali_sens"].append(vali_sens)
            history["vali_spec"].append(vali_spec)
            history["vali_f1"].append(vali_f1)

    # Plot the training curve
    show_history_plot(history, 0, validation_per_epoch, savedir=imgdir + str(k) + "-history.jpg")

    model.eval()

    y_pred_proba, y_real = get_all_preds(model, test_loader, softmax=True)
    test_auc = roc_auc_score(y_real, y_pred_proba[:, 1])
    print("auc", test_auc)
    y_pred = np.array([1 if p > 0.5 else 0 for p in y_pred_proba[:, 1]])

    test_accuracy, test_precision, test_sensitivity, test_specificity, test_f1 = analyse2(y_pred, y_real)

    # plot the ROC curve, calculate the AUC and save to file
    fpr, tpr, thresholds_roc = roc_curve(y_real, y_pred_proba[:, 1], pos_label=1)
    fig = plt.figure()
    plt.plot(fpr, tpr, marker='.')
    plt.title("ROC")
    plt.savefig(imgdir + str(k) + "-roc.jpg")
    # plt.show()

    df = pd.read_excel("../actual fall time.xlsx", index_col="ID")  # load actual time of fall from excel records
    for i, item in df.loc[1:, 10:].iterrows():
        for j in item.index:
            item[j] = item[j].split(",")
            for k in range(0, len(item[j])):
                item[j][k] = [int(x) for x in item[j][k].split("/")]
            # print(i,j,item[j])
        df.loc[i, 10:] = item

    y_pred, y_reals = get_all_preds(model, external_loader)  # predict the TCS dataset

    # error warning list (warning while not falling), no warning list (not warning while falling)
    error_list = [[], []]

    error_range = 3  # allow error range of time (The recorded actual fall time may be deviation)
    res = dict()
    for (i, item) in enumerate(y_reals):
        # item contains: [volunteer ID, activity ID, repeat ID、the start time of the sliced item]

        key = (item[0], item[1], item[2])
        if key not in res:
            res[key] = [
                [0, 0],  # the number of predict negative and positive
                df.loc[item[0], item[1]][item[2]][0],  # actual time of fall from excel records
                [],  # the time of sample which is predicted falling list
            ]

        if y_pred[i] == 0:
            res[key][0][0] += 1
        else:
            res[key][0][1] += 1
            # res[key][2].append("%d~%d"%(item[3],item[3]+8))  # the time range of sample which is predicted falling
            res[key][2].append(item[3] + 8)  # the time of sample which is predicted falling

            # recorded as error warning if predicted falling before the actually falling start or after actually falling complete
            if item[3] + 8 < res[key][1] - error_range or item[3] > res[key][1] + error_range:
                error_list[0].append([key, item[3] + 8])
    for key in res:
        correct_num = 0  # The number of correct prediction
        for t in res[key][2]:
            # recorded as correct warning if predicted falling when the fall is actually happening
            if res[key][1] - error_range <= t <= res[key][1] + error_range + 8:
                correct_num += 1
        if correct_num == 0:
            error_list[1].append(key)

    out = pd.read_csv("result.csv")
    result = pd.DataFrame([{
        'time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'filter': "butterworth0.2",
        'preprocess': "cv20",
        'model': "CNN",
        'epoch': EPOCH,
        'test_volunteers': str(test_volunteers.tolist()),
        'train_volunteers': str(train_volunteers.tolist()),
        'auc': "%.5f" % test_auc,
        'accuracy': "%.5f" % test_accuracy,
        'precision': "%.5f" % test_precision,
        'sensitivity': "%.5f" % test_sensitivity,
        'specificity': "%.5f" % test_specificity,
        'f1': "%.5f" % test_f1,
        'group': len(res),  # num of group in test set
        'sample': len(y_pred),  # num of sample in test set
        'missing_group': len(error_list[1]),  # num of no warning TCS sample
        'misinformation': len(error_list[0]),  # num of error warning sliced item
        'misinformation_group': len(set([x[0] for x in error_list[0]])),  # num of error warning TCS sample
    }])

    out = pd.concat((out, result), axis=0)
    out.to_csv("result.csv", index=False)

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


# predict (input data on GPU, output data on CPU）
@torch.no_grad()
def get_all_preds(model, loader,loss_fn=None,softmax=False):
    y_pred=torch.Tensor([]).int()#.cuda()
    y_real=torch.Tensor([]).int()#.cuda()
    losses=[]
    for step,data in enumerate(loader):
        x,y=data
        x=x.cuda()
        y=y.cuda()
        # print(x,y)
        output=model(x)
        if softmax:
            y_pred_step = torch.nn.functional.softmax(output,dim=1)
        else:
            y_pred_step = torch.max(output,1)[1].data
        y_pred=torch.cat((y_pred,y_pred_step.cpu()) ,dim = 0)
        y_real=torch.cat( (y_real,y.cpu()) ,dim = 0)
        if loss_fn!=None:
            loss=loss_fn(output,y)
            losses.append(loss.item())
    if loss_fn==None:
        return y_pred.numpy(),y_real.numpy()
    else:
        return y_pred.numpy(),y_real.numpy(),sum(losses)/len(losses)

def show_history_plot(history, start=0, validation_per_epoch=10,savedir=None):
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("loss")
    # plt.ylim(0,1)
    plt.plot(range(start, len(history["loss"])), history["loss"][start:], label="loss")
    plt.plot(range(start, len(history["loss"]), validation_per_epoch),
             history["vali_loss"][start // validation_per_epoch:], label="vali_loss")
    plt.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("accuracy")
    # plt.ylim(0,1)
    plt.plot(range(start, len(history["loss"])), history["acc"][start:], label="acc")
    plt.plot(range(start, len(history["loss"]), validation_per_epoch),
             history["vali_acc"][start // validation_per_epoch:], label="vali_acc")
    plt.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("sensitivity")
    # plt.ylim(0,1)
    plt.plot(range(start, len(history["loss"])), history["sens"][start:], label="sens")
    plt.plot(range(start, len(history["loss"]), validation_per_epoch),
             history["vali_sens"][start // validation_per_epoch:], label="vali_sens")
    plt.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("specificity")
    # plt.ylim(0,1)
    plt.plot(range(start, len(history["loss"])), history["spec"][start:], label="spec")
    plt.plot(range(start, len(history["loss"]), validation_per_epoch),
             history["vali_spec"][start // validation_per_epoch:], label="vali_spec")
    plt.legend()
    if savedir:
        plt.savefig(savedir)
    else:
        plt.show()

def analyse(y_pred,y_real):
    accuracy=sum(y_pred==y_real)/len(y_real)
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_real)):
        if y_real[i] == 1 and y_pred[i] == 1:
           TP += 1
        if y_real[i] == 0 and y_pred[i] == 1:
           FP += 1
        if y_real[i] == 0 and y_pred[i] == 0:
           TN += 1
        if y_real[i] == 1 and y_pred[i] == 0:
           FN += 1
    # Sensitivity, hit rate, recall, or true positive rate
    sensitivity = TP/(TP+FN) if TP+FN>0 else 0
    # Specificity or true negative rate
    specificity = TN/(TN+FP) if TN+FP>0 else 0
    return accuracy,sensitivity,specificity


def analyse2(y_pred, y_real):
    accuracy = sum(y_pred == y_real) / len(y_real)
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_real)):
        if y_real[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_real[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_real[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_real[i] == 1 and y_pred[i] == 0:
            FN += 1

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    sensitivity = TP / (TP + FN) if TP + FN > 0 else 0  # Sensitivity, hit rate, recall, or true positive rate(TPR)
    recall = sensitivity
    # Specificity or true negative rate
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return accuracy, precision, sensitivity, specificity, f1

def createHeatmap(y_pred,y_real,n):
    stacked = np.stack((y_pred,y_real),axis=1)#.astype(np.int64)
    cmt = np.zeros((n,n), dtype=np.int64)
    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1
    print("Prediction matrix:")
    print(cmt)
    cmt=cmt/cmt.sum(axis=0)*100

    import seaborn as sns
    fig = plt.figure(figsize=(0.75*n, 0.75*n))
    sns.heatmap(cmt,xticklabels=range(0,n), yticklabels=range(0,n), annot=True, cmap="YlGnBu",fmt='.3g')
    plt.show()


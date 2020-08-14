import torch
from torch.nn.functional import (softmax)
from sklearn.metrics import (roc_auc_score, confusion_matrix, f1_score)
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np


def post_process_output(output):
    # implementation based on problem statement
    THRESHOLD = 0.8
    # intermediate = torch.sigmoid(output) > THRESHOLD
    intermediate = output > THRESHOLD
    
    return intermediate.float()

# def post_process_output(output):
#     # implementation based on problem statement
#     return softmax(output, dim=1)


def onehot_converter(V, classes):
    # Create zero vector of desired shape
    OHV = torch.zeros((V.size()[0], classes))

    # Convert from ndarray to array
    V_a = V.view(-1)

    # Fill ones where the label as index
    OHV[range(V.size()[0]), V_a.long()] = 1

    return OHV


# def acc(output_list, target_list):
#     output_list = post_process_output(output_list)
#     I = torch.logical_and(target_list, output_list)
#     U = torch.logical_or(target_list, output_list)
#     batch_iou = torch.sum(I, dim=1, dtype=torch.float)/torch.sum(U, dim=1)
#     acc = torch.sum(batch_iou) / len(batch_iou)
#     return acc.item()

def acc(output_list, target_list):
    acc = torch.argmax(target_list, dim=1).eq(torch.argmax(post_process_output(output_list), dim=1))
    return 1.0 * torch.sum(acc.int()).item() / output_list.size()[0]


def f1(output_list, target_list, threshold=0.5):
    # output_list = torch.sigmoid(output_list) > threshold
    output_list = output_list > threshold

    output_list = output_list.float().numpy()
    target_list = target_list.float().numpy()
    return f1_score(target_list, output_list, average="samples")

def f1best(output_list,target_list,threshold=0.5):
    score=0
    for th in np.linspace(0.5,0.9,20):
        out = torch.sigmoid(output_list) > threshold
        out = out.float().numpy()
        target_list = target_list.float().numpy()
        score=max(score,f1_score(target_list, out, average="samples"))
    return score

def aroc(output_list, target_list):
    output_list = post_process_output(output_list)
    target_list = onehot_converter(target_list, output_list.shape[1])
    return roc_auc_score(
        target_list.numpy(),
        output_list.numpy(),
        average="macro"
    )


def confusion_matrix_generator(output_list, target_list, experiment_name):
    output_list = post_process_output(output_list)
    matrix = confusion_matrix(
        torch.argmax(target_list, dim=1).numpy(),
        torch.argmax(output_list, dim=1).numpy()
    )

    labels = ['H', 'MD', 'R', 'S']
    plt.figure()
    figure = sn.heatmap(matrix, xticklabels=labels,
                        yticklabels=labels, annot=True)
    plt.savefig('./results/' + experiment_name + '/confusion_matrix.jpg')

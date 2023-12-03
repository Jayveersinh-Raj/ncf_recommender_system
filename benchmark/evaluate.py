import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from movie_dataset import MovieLens
from MLP import MLP

def hit(gt_item, pred_items):
    """
    Function to calculate hit rate.
    
    Parameters:
    -----------
    gt_item: The ground truth item (movie watched by the user).
    pred_items: The predicted/recommended items by the model.
    
    Returns:
    -----------
    1 if the recommended item was watched, 0 otherwise.
    """
    if gt_item in pred_items:
        return 1
    return 0

def ndcg(gt_item, pred_items):
    """
    Normalized Discounted Cumulative Gain (NDCG) metric.
    
    Parameters:
    ------------
    gt_item: The ground truth item (movie watched by the user).
    pred_items: The predicted/recommended items by the model.
    
    Returns:
    ------------
    The calculated NDCG value based on the formula if the movie was watched, else 0.
    """
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0

def metrics(model, test_loader, top_k, device):
    """
    Function to calculate metrics - Hit Rate (HR) and Normalized Discounted Cumulative Gain (NDCG) for each user.
    
    Parameters:
    ------------
    model: The model checkpoints.
    test_loader: The test dataloader.
    top_k: The number of movies to recommend.
    device: CPU or GPU device.
    
    Returns:
    ------------
    Tuple: Mean HR and mean NDCG.
    """
    HR, NDCG = [], []

    for user, item, label in test_loader:
        user = user.to(device)
        item = item.to(device)

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)

        recommends = torch.take(item, indices).cpu().numpy().tolist()

        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)

if __name__ == '__main__':
    test_dataframe = pd.read_csv("data/evaluation.csv")
    total_dataframe = pd.read_csv("../data/interim/entire_dataset.csv")
    
    # Getting it in tensor inference ready dataset
    test_set = MovieLens(df=test_dataframe, total_df=total_dataframe, ng_ratio=99)
    
    # dataloader for test_dataset
    dataloader_test = DataLoader(dataset=test_set,
                                 batch_size=100,
                                 shuffle=False,
                                 num_workers=0,
                                 drop_last=True
                                 )
    
    # Loading the model
    model = torch.load("../models/MLP.pth")
    model.eval()
    
    HR, NDCG = metrics(model=model, test_loader=dataloader_test, top_k=10, device="cuda")
    print("HR: {:.3f}\nNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
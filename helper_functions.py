import torch
from torch.nn import Module
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

def display_data(data, m, b, index : int = None):
    # Creates a graph with all input data and the line defined
    # by your model
    # input: (x, y) matrix of data
    # m: slope component of regression
    # b: vertical translation component of regression
    if torch.is_tensor(m):
        m = m.item()
    if torch.is_tensor(b):
        b = b.item()

    fig = plt.figure()

    weights = data[:, 0]
    costs = data[:, 1]

    plt.scatter(weights, costs)

    plt.xlabel('Cow Weight (lbs)', fontsize=12)
    plt.ylabel('Cow Cost (CAD)', fontsize=12)

    x = np.linspace(800, 1300, 1000)
    
    if m is not None and b is not None:
        plt.plot(x, m * x + b, color = "red")

    plt.xlim([800, 1300])

    if not os.path.isdir("Graphs"):
        os.mkdir("Graphs")

    if index is None:
        plt.title("My Model's Prediction of Cow Cost on Cow Weight")
        plt.savefig("Graphs/Cow_Prediction.png")
    else:
        plt.title(f"My Model's Prediction of Cow Cost on Cow Weight at Index {index}")
        plt.savefig(f"Graphs/Cow_Prediction_at_Index_{index}.png")

def read_data(path):
    # Reads cow path csv file and returns numpy 2d array of
    # x and y pairs
    # path: (string) path to csv file "cow_cost.csv"
    data = list()
    with open(path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row == ["Weight", "Cost"]:
                continue
            row = [int(row[0]), int(row[1])]
            data.append(row)
    return np.array(data)
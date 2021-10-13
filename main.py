import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

from helper_functions import read_data, display_data

# Machine Learning Model:
# This is the model that we are defining.
# It will contain parameters that we want to train and
# will define how the equation that the data 
class CowPredictor(torch.nn.Module):
    def __init__(self):
        # The __init__(self) function is where we define parameters for the
        # model. super(Model, self).__init__() helps set up this initialization.
        #
        # Example:
        # self.x = torch.nn.Parameter(torch.tensor(1.0, requires_grad = True))
        # this defines x to be value 1.0
        #
        # This declares a variable "x" that can now be used by the model.

        super(CowPredictor, self).__init__()

        # Your code here:
        self.m = torch.nn.Parameter(torch.tensor(2.0, requires_grad = True))
        self.b = torch.nn.Parameter(torch.tensor(0.0, requires_grad = True))

    def forward(self, cow_weight):
        # __forward__(self, cow_weight) defines the equation you want your
        # data to run through.
        # Here, cow_weight is the input for your model.
        #
        # Example:
        # result = cow_weight * self.x - 10
        # return result
        #
        # This example takes an input, multiplies it by variable "x" defined
        # in __init__(), subtracts 10 and returns the result.
        
        # Your code here:
        cow_cost = self.m * cow_weight + (self.b * 1000)

        return cow_cost

data = read_data("cow_cost.csv")

#defining the model
model = CowPredictor()
model.train()

#defining the optimizer
optim = torch.optim.SGD(model.parameters(), lr=1e-7)

#defining the loss function
def MSE_loss(true, predicted):
    return (true - predicted) ** 2

display_data(data, model.m, model.b)

for epoch_number in range(10):
    for cow_number, pair in enumerate(data):
        optim.zero_grad()

        weight = pair[0]
        true_cost = pair[1]

        predicted_cost = model(weight)

        loss = MSE_loss(true_cost, predicted_cost)
        
        #Handing the optimizer the loss
        loss.backward()
        optim.step()

    display_data(data, model.m, model.b, index = epoch_number)



    

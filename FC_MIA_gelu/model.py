import torch.nn as nn
import torch

class DNN(nn.Module):

    def __init__(self, hidden_size):
        super(DNN, self).__init__()
        self.image_size = 32
        num_classes = 2
        self.total_image_size = self.image_size * self.image_size * 3
        self.hidden_size = hidden_size
        self.GELU = torch.nn.GELU()

        self.L_1 = torch.nn.Linear(self.total_image_size, self.hidden_size, bias = False)
        self.L_2 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias = False)
        self.L_3 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias = False)
        self.L_4 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias = False)
        self.L_5 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias = False)
        self.L_6 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias = False)
        self.L_7 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias = False)
        self.L_8 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias = False)
        self.L_9 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias = False)
        self.L_10 = torch.nn.Linear(self.hidden_size, num_classes, bias = False)

    def forward(self, z):
        h = z.view(-1, self.total_image_size)
        out = self.GELU(self.L_1(h))
        out = self.GELU(self.L_2(out))
        out = self.GELU(self.L_3(out))
        out = self.GELU(self.L_4(out))
        out = self.GELU(self.L_5(out))
        out = self.GELU(self.L_6(out))
        out = self.GELU(self.L_7(out))
        out = self.GELU(self.L_8(out))
        out = self.GELU(self.L_9(out))
        out = self.L_10(out)
        return out
import csv
from pathlib import Path

import click
import torch
import torch.nn as nn


def csv_to_weight_dict(csvpath, product):
    model_ids = {
        "LAI": 1,
        "fAPAR": 2,
        "fCOVER": 3,
        "CCC": 4,
        "CWC": 5,
        "Albedo": 6,
        "DASF": 7,
    }

    with open(csvpath, "r") as f:
        reader = csv.reader(f)
        w = [row for row in reader]
        w = w[model_ids[product]]
        w = [float(i) for i in w]

    # Parse according to format
    # See https://code.earthengine.google.com/28faa7482764331cf427924ed36d1529?accept_repo=users%2Frfernand387%2FLEAFToolboxModules
    # for original EE implementation

    # Extract the number of input channels

    # Start at tabledata6
    w_dict = {}
    pointer = 5  # zero indexed
    # Number of input weights
    n = int(w[pointer])
    n_ch_input = n
    pointer += 1
    # Read input weights
    w_dict["input"] = {}
    w_dict["input"]["weights"] = w[pointer:pointer + n]
    pointer += n
    # Number of input biases
    n = int(w[pointer])
    pointer += 1
    # Read input biases
    w_dict["input"]["biases"] = w[pointer:pointer + n]
    pointer += n
    # Number of weights in first hidden layer
    n = int(w[pointer])
    pointer += 1
    # Read hidden1 weights
    w_dict["h1"] = {}
    w_dict["h1"]["weights"] = w[pointer:pointer + n]
    pointer += n
    # Number of biases in first hidden layer
    n = int(w[pointer])
    pointer += 1
    # Read hidden1 biases
    w_dict["h1"]["biases"] = w[pointer:pointer + n]
    pointer += n
    # Number of weights in second hidden layer
    n = int(w[pointer])
    pointer += 1
    # Read hidden2 weights
    w_dict["h2"] = {}
    w_dict["h2"]["weights"] = w[pointer:pointer + n]
    pointer += n
    # Number of biases in second hidden layer
    n = int(w[pointer])
    pointer += 1
    # Read hidden2 biases
    w_dict["h2"]["biases"] = w[pointer:pointer + n]
    pointer += n
    # Number of weights inoutput layer
    n = int(w[pointer])
    pointer += 1
    # Read output weights
    w_dict["output"] = {}
    w_dict["output"]["weights"] = w[pointer:pointer + n]
    pointer += n
    # Number of biases in output layer
    n = int(w[pointer])
    pointer += 1
    # Read output biases
    w_dict["output"]["biases"] = w[pointer:pointer + n]

    return w_dict, n_ch_input


class Scale2d(nn.Module):
    def __init__(self, n_ch):
        super(Scale2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, n_ch, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(1, n_ch, 1, 1))

    def forward(self, x):
        return x * self.weight + self.bias


class UnScale2d(nn.Module):
    def __init__(self, n_ch):
        super(UnScale2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, n_ch, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(1, n_ch, 1, 1))

    def forward(self, x):
        return (x - self.bias) / self.weight


class LAI_CNN(nn.Module):
    def __init__(self, in_ch, h1_dim, out_ch):
        super(LAI_CNN, self).__init__()
        self.input = Scale2d(in_ch)
        self.h1 = nn.Conv2d(in_ch, h1_dim, 1, 1, 0, bias=True)
        self.h2 = nn.Conv2d(h1_dim, out_ch, 1, 1, 0, bias=True)
        self.output = UnScale2d(out_ch)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.input(x)
        x = self.h1(x)
        x = self.tanh(x)
        x = self.h2(x)
        x = self.output(x)
        return x


def weight_dict_to_torch(w_dict, n_ch_in, outpath):
    model = LAI_CNN(n_ch_in, 5, 1)
    with torch.no_grad():
        model.input.weight = nn.Parameter(
            torch.tensor(w_dict["input"]["weights"]).reshape(1, n_ch_in, 1, 1)
        )
        model.input.bias = nn.Parameter(
            torch.tensor(w_dict["input"]["biases"]).reshape(1, n_ch_in, 1, 1)
        )
        model.h1.weight = nn.Parameter(
            torch.tensor(w_dict["h1"]["weights"]).reshape(5, n_ch_in, 1, 1)
        )
        model.h1.bias = nn.Parameter(torch.tensor(w_dict["h1"]["biases"]).reshape(5))
        model.h2.weight = nn.Parameter(
            torch.tensor(w_dict["h2"]["weights"]).reshape(1, 5, 1, 1)
        )
        model.h2.bias = nn.Parameter(torch.tensor(w_dict["h2"]["biases"]).reshape(1))
        model.output.weight = nn.Parameter(
            torch.tensor(w_dict["output"]["weights"]).reshape(1, 1, 1, 1)
        )
        model.output.bias = nn.Parameter(
            torch.tensor(w_dict["output"]["biases"]).reshape(1, 1, 1, 1)
        )

    torch.save(model.state_dict(), outpath)


@click.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.argument(
    "model",
    type=click.Choice(["LAI", "fAPAR", "fCOVER", "CCC", "CWC", "Albedo", "DASF"]),
)
def main(csv_path, model):
    """
    Convert a CSV file of model weights to a PyTorch model file.
    csv_path: Path to the CSV file containing the model weights.
    model: The model type to convert. One of ['LAI', 'fAPAR', 'fCOVER', 'CCC', 'CWC', 'Albedo', 'DASF']
    """

    # Convert CSV to dictionary
    w_dict, n_ch_in = csv_to_weight_dict(csv_path, model)

    weights_filename = Path(csv_path).stem + "_LAI" + ".pth"
    weight_dict_to_torch(w_dict, n_ch_in, weights_filename)

    print(f"Model weights saved to {weights_filename}")


if __name__ == "__main__":
    main()

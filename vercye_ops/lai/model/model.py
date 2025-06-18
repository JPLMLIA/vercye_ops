import torch.nn as nn
import torch

# Specification of required input bands/channels and actual weights
# All input bands are expected to be in the same format as S2_SR_Harmonized in GEE
# This means for baseline >= 4, the offset of 1000 has to be subtracted.
default_model_weights = {
    "S2": {
        10: {
            "weights_path": "../trained_models/s2_sl2p_weiss_or_prosail_10m_NNT1_Single_0_1_LAI.pth",
            "channels": ["cosVZA", "cosSZA", "cosRAA", "B2", "B3", "B4", "B8"],
        },
        20: {
            "weights_path": "../trained_models/s2_sl2p_weiss_or_prosail_NNT3_Single_0_1_LAI.pth",
            "channels": [
                "cosVZA",
                "cosSZA",
                "cosRAA",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8A",
                "B11",
                "B12",
            ],
        },
    }
}

def load_model_from_weights(model_weights, in_channels):
    num_in_ch = len(in_channels)
    print(f"Loading model weights from {model_weights} with {num_in_ch} input channels")
    model = LAI_CNN(num_in_ch, 5, 1)
    model.load_state_dict(torch.load(model_weights))
    return model

def load_model(sateillite:str, resolution: int):
    sateillite = "S2"  # Currently only S2 is supported
    model_resolution = resolution
    if resolution not in default_model_weights[sateillite]:
        print(
            "Warning: No model weights found for this resolution. Falling back to model trained at a resolution of 20m."
        )
        model_resolution = 20

    model_options = default_model_weights[sateillite][model_resolution]
    model_weights = model_options["weights_path"]
    channels = model_options["channels"]

    # Load the pytorch model
    model = load_model_from_weights(model_weights, channels)

    return model


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
        self.num_in_channels = in_ch
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

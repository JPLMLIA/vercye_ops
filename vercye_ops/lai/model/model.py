import os

import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def rel_path(*path_parts):
    return os.path.join(BASE_DIR, *path_parts)


# Specification of required input bands/channels and actual weights
# See https://github.com/rfernand387/LEAF-Toolbox/blob/master/Source-Python/production/dictionariesSL2P.py for details
default_model_weights = {
    "S2": {
        # All input bands are expected to be in the same format as S2_SR_Harmonized in GEE
        # This means for baseline >= 4, the offset of 1000 has to be subtracted.
        "default_resolution": 20,
        "configs": {
            # Disabling 10m model for now, as it' not well validated yet
            # 10: {
            #     "weights_path": rel_path("../trained_models/s2_sl2p_weiss_or_prosail_10m_NNT1_Single_0_1_LAI.pth"),
            #     "channels": ["cosVZA", "cosSZA", "cosRAA", "B2", "B3", "B4", "B8"],
            #     "input_scaling": [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
            # },
            20: {
                "weights_path": rel_path("../trained_models/s2_sl2p_weiss_or_prosail_NNT3_Single_0_1_LAI.pth"),
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
                "input_scaling": [
                    0.0001,
                    0.0001,
                    0.0001,
                    0.0001,
                    0.0001,
                    0.0001,
                    0.0001,
                    0.0001,
                    0.0001,
                    0.0001,
                    0.0001,
                ],
            },
        },
    },
    # HLS is not yet properly working and is WIP that was accidentally merged on main
    # "HLS_S30": {
    #     "default_resolution": 30,
    #     "configs": {
    #         30: {
    #             "weights_path": rel_path(
    #                 "../trained_models/L8_sl2p_weiss_or_prosail_NNT3_Single_0_1_fromGEEFC_LAI.pth"
    #             ),
    #             "channels": [
    #                 "cosVZA",
    #                 "cosSZA",
    #                 "cosRAA",
    #                 "B3",
    #                 "B4",
    #                 "B8A",
    #                 "B11",
    #                 "B12",
    #             ],
    #             "input_scaling": [0.0001, 0.0001, 0.0001, 1, 1, 1, 1, 1],
    #         },
    #     },
    # },
    # "HLS_L30": {
    #     "default_resolution": 30,
    #     "configs": {
    #         30: {
    #             "weights_path": rel_path(
    #                 "../trained_models/L8_sl2p_weiss_or_prosail_NNT3_Single_0_1_fromGEEFC_LAI.pth"
    #             ),
    #             "channels": [
    #                 "cosVZA",
    #                 "cosSZA",
    #                 "cosRAA",
    #                 "B3",
    #                 "B4",
    #                 "B5",
    #                 "B6",
    #                 "B7",
    #             ],
    #             "input_scaling": [0.0001, 0.0001, 0.0001, 1, 1, 1, 1, 1],
    #         },
    #     },
    # },
}


def load_model_from_weights(model_weights, in_channels, input_scaling):
    num_in_ch = len(in_channels)
    print(f"Loading model weights from {model_weights} with {num_in_ch} input channels")
    model = LAI_CNN(num_in_ch, 5, 1, input_scaling)
    model.load_state_dict(torch.load(model_weights))
    return model


def load_model(satellite: str, resolution: int):
    model_resolution = resolution

    # We dont have an explicit model for each resolution, use default resolution for those that are not defined
    if resolution not in default_model_weights[satellite]["configs"]:
        default_resolution = default_model_weights[satellite]["default_resolution"]
        model_resolution = default_resolution
        print(
            f"Warning: No model weights found for this resolution. Falling back to default resolution {default_resolution}m."
        )

    model_options = default_model_weights[satellite]["configs"][model_resolution]
    model_weights = model_options["weights_path"]
    channels = model_options["channels"]
    input_scaling = model_options["input_scaling"]

    # Load the pytorch model
    model = load_model_from_weights(model_weights, channels, input_scaling)

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
    def __init__(self, in_ch, h1_dim, out_ch, input_scaling):
        super(LAI_CNN, self).__init__()
        self.num_in_ch = in_ch
        self.input = Scale2d(in_ch)
        self.h1 = nn.Conv2d(in_ch, h1_dim, 1, 1, 0, bias=True)
        self.h2 = nn.Conv2d(h1_dim, out_ch, 1, 1, 0, bias=True)
        self.output = UnScale2d(out_ch)
        self.tanh = nn.Tanh()

        self.scale = torch.tensor(input_scaling).view(1, -1, 1, 1)

    def forward(self, x):
        x = x * self.scale
        x = self.input(x)
        x = self.h1(x)
        x = self.tanh(x)
        x = self.h2(x)
        x = self.output(x)
        return x

import torch
import numpy as np
from torch.nn.functional import binary_cross_entropy_with_logits


def test_gen_bins():
    n_bins = 4
    bins = torch.linspace(0, 1, n_bins + 1)
    print(bins)
    bins2 = np.arange(0, 1, 0.25)
    print(bins2)

def quantize(weights, n_bits=8):
    assert n_bits < 16
    n_bins = 2 ** n_bits
    bins = torch.linspace(weights.min(), weights.max(), n_bins + 1)
    first_bin = bins[0]
    width = bins[1] - bins[0]
    bin_indexes = ((weights.view(-1, 1) > bins).to(torch.int).argmin(dim=1) * 1)
    return bin_indexes, width, first_bin

def dequantize(bin_indexes, width, first_bin):
    return bin_indexes * width + first_bin

def test_quantize_function():
    weights = torch.tensor([0.3, 0.2184, 0.0593, 0.973])
    bin_indexes, width, first_bin = quantize(weights, 8)
    print(bin_indexes, width, first_bin)
    dequantized = dequantize(bin_indexes, width, first_bin)
    print(dequantized)

def half_precision_quantize():
    weights = torch.tensor([0.3, 0.2184, 0.059338745, 0.973])
    fp16_weights = weights.to(torch.float16)
    print(fp16_weights)

    weights2 = torch.tensor([0.3, 0.2184, 0.059338745, 0.973]) * 1e-8
    print(weights2)
    fp16_weights = weights2.to(torch.float16)       # FP16 version doesn't work when the weights are too small to too large
    print(fp16_weights)

half_precision_quantize()
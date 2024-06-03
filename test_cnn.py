import unittest
import torch
from torch import nn
from CNN.conv2D import corr2d, Conv2D 

class TestConv2D(unittest.TestCase):
    def test_corr2d_basic(self):
        X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        expected_output = torch.tensor([[19.0, 25.0], [37.0, 43.0]])
        output = corr2d(X, K)
        self.assertTrue(torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}")

    def test_conv2d_layer(self):
        X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        conv_layer = Conv2D((2, 2))
        conv_layer.weights = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        conv_layer.bias = nn.Parameter(torch.tensor([1.0]))
        expected_output = torch.tensor([[38.0, 48.0], [68.0, 78.0]])
        output = conv_layer(X)
        self.assertTrue(torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}")

    def test_different_kernel_sizes(self):
        X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        K = torch.tensor([[1.0]])
        expected_output = X  # Since kernel is 1x1, output should be the same as input
        output = corr2d(X, K)
        self.assertTrue(torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}")

        K = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
        expected_output = torch.tensor([[-4.0, -4.0], [-4.0, -4.0]])
        output = corr2d(X, K)
        self.assertTrue(torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}")

    def test_gradient_check(self):
        X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True)
        conv_layer = Conv2D((2, 2))
        output = conv_layer(X)
        output.sum().backward()
        self.assertIsNotNone(X.grad, "Gradients not computed for input X")

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

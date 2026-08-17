import pytest
import torch

from library.qwen_image_autoencoder_kl import ChunkedConv2d


@pytest.mark.parametrize(
    ("kernel_size", "stride", "padding", "height", "width", "chunk_size"),
    [
        pytest.param(3, 2, 0, 18, 22, 8, id="stride-two-even-regression"),
        pytest.param(3, 2, 0, 65, 33, 16, id="vae-manually-padded-tall"),
        pytest.param(3, 1, 1, 97, 29, 16, id="padded-tall"),
        pytest.param(1, 1, 0, 97, 29, 16, id="pointwise-tall"),
    ],
)
def test_chunked_conv2d_matches_conv2d(kernel_size, stride, padding, height, width, chunk_size):
    """Chunking must preserve ordinary Conv2d shape and values."""
    torch.manual_seed(0)
    reference = torch.nn.Conv2d(3, 5, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    chunked = ChunkedConv2d(
        3,
        5,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
        spatial_chunk_size=chunk_size,
    )
    chunked.load_state_dict(reference.state_dict())
    sample = torch.randn(2, 3, height, width)

    expected = reference(sample)
    actual = chunked(sample)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)

from spandrel.architectures.RCAN import RCAN, RCANArch

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        RCANArch(),
        lambda: RCAN(),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV4/2x-AnimeSharpV4_RCAN.safetensors",
        name="2x-AnimeSharpV4_RCAN.safetensors",
    )
    assert_size_requirements(file.load_model())


def test_2x_AnimeSharpV4_RCAN(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV4/2x-AnimeSharpV4_RCAN.safetensors",
        name="2x-AnimeSharpV4_RCAN.safetensors",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, RCAN)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )

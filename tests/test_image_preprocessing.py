import numpy as np

from foreact.dataloader.gr00t_lerobot.image_preprocessing import (
    preprocess_observation_image,
)


def test_bottom_crop_and_resize_is_idempotent():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[124:, :, 1] = 255
    output = preprocess_observation_image(
        image,
        crop_xyxy=[0, 124, 640, 480],
        size_hw=[320, 576],
    )
    assert output.size == (576, 320)
    assert np.asarray(output)[..., 1].min() == 255

    repeated = preprocess_observation_image(
        output,
        crop_xyxy=[0, 124, 640, 480],
        size_hw=[320, 576],
    )
    assert repeated.size == output.size
    assert np.array_equal(np.asarray(repeated), np.asarray(output))


def test_invalid_crop_fails_loudly():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        preprocess_observation_image(
            image,
            crop_xyxy=[0, 0, 101, 100],
            size_hw=[50, 50],
        )
    except ValueError as error:
        assert "outside input image size" in str(error)
    else:
        raise AssertionError("invalid crop was silently accepted")

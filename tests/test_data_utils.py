from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import torch

import data_utils


COMUNE = "Brisighella"


def test_check_similarity_accepts_two_dimensional_images():
    image = np.indices((16, 16)).sum(axis=0).astype(np.float32) / 30
    assert data_utils.check_similarity(image, image, np.ones((16, 16), dtype=bool))


def test_check_similarity_rejects_anticorrelated_images_without_asserting():
    checkerboard = (np.indices((16, 16)).sum(axis=0) % 2).astype(np.float32)[None]
    assert not data_utils.check_similarity(
        checkerboard, 1 - checkerboard, np.ones((16, 16), dtype=bool)
    )


def _write_raster(
    path: Path,
    data: np.ndarray,
    *,
    nodata: float | None = None,
    transform=None,
) -> None:
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        transform=transform or from_origin(0, data.shape[1], 1, 1),
        nodata=nodata,
    ) as dst:
        dst.write(data)


def _make_comune(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    main_dir = tmp_path / "Comuni"
    directory = main_dir / COMUNE
    directory.mkdir(parents=True)
    monkeypatch.setattr(data_utils, "MAIN_DIR", str(main_dir))
    return directory


def test_generate_dataset_mask_excludes_nodata_and_nan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _make_comune(tmp_path, monkeypatch)
    data = np.ones((4, 2, 3), dtype=np.float32)
    data[3, 0, 1] = -9999
    data[3, 1, 2] = np.nan
    _write_raster(directory / "Cgr_2023_2m.tif", data, nodata=-9999)

    mask = data_utils.generate_dataset_mask(COMUNE)

    np.testing.assert_array_equal(
        mask,
        np.array([[True, False, True], [True, True, False]]),
    )


def test_get_data_classifies_raster_by_basename_not_parent_directory(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "ndvi-change-slope"
    directory.mkdir()
    path = directory / "Cgr_2023_2m.tif"
    _write_raster(path, np.full((4, 2, 2), 127.5, dtype=np.float32))

    with rasterio.open(path) as src:
        data = data_utils.get_data(src, to_norm=True)

    np.testing.assert_allclose(data, 0.5)


def test_segmentation_stack_has_semantic_order_and_ignores_non_rasters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _make_comune(tmp_path, monkeypatch)
    shape = (3, 4)
    _write_raster(
        directory / "Cgr_2023_2m.tif",
        np.stack([np.full(shape, value, np.float32) for value in (40, 80, 120, 160)]),
    )
    _write_raster(
        directory / "Agea_2014_2m.tif",
        np.stack([np.full(shape, value, np.float32) for value in (10, 20, 30, 40)]),
    )
    _write_raster(directory / "Slope_2m.tif", np.full(shape, 45, np.float32))
    _write_raster(directory / "Ndvi_2m.tif", np.full(shape, 0.25, np.float32))
    landslides = np.zeros(shape, dtype=np.float32)
    landslides[1, 2] = 1
    _write_raster(directory / "Frane_2m.tif", landslides)
    (directory / "metadata.json").write_text("{}")
    (directory / "cache").mkdir()

    original_listdir = data_utils.os.listdir
    monkeypatch.setattr(
        data_utils.os,
        "listdir",
        lambda path: list(reversed(original_listdir(path))),
    )

    inputs, target = data_utils.get_segmentation_stack(COMUNE, include_slope_ndvi=True)

    expected_values = np.array(
        [10, 20, 30, 40, 40, 80, 120, 160], dtype=np.float32
    ) / 255
    expected_values = np.concatenate([expected_values, [0.5, 0.625]])
    np.testing.assert_allclose(inputs[:, 0, 0], expected_values)
    assert target.shape == (1, *shape)
    assert target[0, 1, 2]


def test_super_resolution_stack_uses_event_tokens_and_stable_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_dir = tmp_path / "Comuni"
    directory = main_dir / "Predappio"
    directory.mkdir(parents=True)
    monkeypatch.setattr(data_utils, "MAIN_DIR", str(main_dir))
    shape = (3, 4)
    aerial = np.ones((4, *shape), dtype=np.float32)
    _write_raster(directory / "Cgr_2023_2m.tif", aerial * 200)
    _write_raster(directory / "Agea_2014_2m.tif", aerial * 100)
    _write_raster(
        directory / "Sentinel2_Predappio_post_2m.tif", aerial * 8000
    )
    _write_raster(directory / "Sentinel2_Predappio_pre_2m.tif", aerial * 2000)
    (directory / "notes.txt").write_text("not a raster")

    (sentinel_pre, agea), (sentinel_post, cgr) = (
        data_utils.get_super_resolution_stack("Predappio")
    )

    np.testing.assert_allclose(sentinel_pre, 0.2)
    np.testing.assert_allclose(sentinel_post, 0.8)
    np.testing.assert_allclose(agea, 100 / 255)
    np.testing.assert_allclose(cgr, 200 / 255)


def test_super_resolution_stack_excludes_non_sr_post_and_pre_rasters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _make_comune(tmp_path, monkeypatch)
    shape = (3, 4)
    four_band = np.ones((4, *shape), dtype=np.float32)
    _write_raster(directory / "Cgr_2023_2m.tif", four_band * 200)
    _write_raster(directory / "Agea_2014_2m.tif", four_band * 100)
    _write_raster(directory / "Sentinel2_pre_2m.tif", four_band * 2000)
    _write_raster(directory / "Sentinel2_post_2m.tif", four_band * 8000)
    _write_raster(directory / "ndvi_post.tif", np.ones(shape, np.float32))
    _write_raster(directory / "slope_pre.tif", np.ones(shape, np.float32))
    _write_raster(directory / "frane_post.tif", np.ones(shape, np.float32))
    _write_raster(directory / "change_pre.tif", np.ones(shape, np.float32))

    (sentinel_pre, _), (sentinel_post, _) = (
        data_utils.get_super_resolution_stack(COMUNE)
    )

    assert sentinel_pre.shape == (4, *shape)
    assert sentinel_post.shape == (4, *shape)


def test_super_resolution_stack_rejects_duplicate_role_rasters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _make_comune(tmp_path, monkeypatch)
    data = np.ones((4, 3, 4), dtype=np.float32)
    _write_raster(directory / "Cgr_2023_2m.tif", data * 200)
    _write_raster(directory / "Agea_2014_2m.tif", data * 100)
    _write_raster(directory / "Sentinel2_pre_2m.tif", data * 2000)
    _write_raster(directory / "Sentinel2_pre_backup_2m.tif", data * 3000)
    _write_raster(directory / "Sentinel2_post_2m.tif", data * 8000)

    with pytest.raises(
        ValueError, match=r"sentinel_pre.*exactly 4 channels.*found 8"
    ):
        data_utils.get_super_resolution_stack(COMUNE)


def test_stack_dimension_error_names_the_misaligned_raster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _make_comune(tmp_path, monkeypatch)
    _write_raster(directory / "Cgr_2023_2m.tif", np.ones((4, 4, 4), np.float32))
    _write_raster(directory / "Agea_2014_2m.tif", np.ones((4, 3, 4), np.float32))
    _write_raster(directory / "Frane_2m.tif", np.zeros((4, 4), np.float32))

    with pytest.raises(ValueError, match=r"Agea_2014_2m\.tif.*3, 4.*4, 4"):
        data_utils.get_segmentation_stack(COMUNE)


def test_stack_rejects_same_size_raster_on_a_different_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _make_comune(tmp_path, monkeypatch)
    data = np.ones((4, 4, 4), np.float32)
    _write_raster(directory / "Cgr_2023_2m.tif", data)
    _write_raster(
        directory / "Agea_2014_2m.tif",
        data,
        transform=from_origin(10, 4, 1, 1),
    )
    _write_raster(directory / "Frane_2m.tif", np.zeros((4, 4), np.float32))

    with pytest.raises(ValueError, match=r"Agea_2014_2m\.tif.*not aligned"):
        data_utils.get_segmentation_stack(COMUNE)


def test_segmentation_stack_requires_each_requested_semantic_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _make_comune(tmp_path, monkeypatch)
    data = np.ones((4, 4, 4), np.float32)
    _write_raster(directory / "Cgr_2023_2m.tif", data)
    _write_raster(directory / "Agea_2014_2m.tif", data)
    _write_raster(directory / "Slope_2m.tif", np.ones((4, 4), np.float32))
    _write_raster(directory / "Frane_2m.tif", np.zeros((4, 4), np.float32))

    with pytest.raises(ValueError, match="No ndvi data found"):
        data_utils.get_segmentation_stack(COMUNE, include_slope_ndvi=True)


def test_random_patch_fails_after_bounded_attempts_for_sparse_mask() -> None:
    image = np.ones((1, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=bool)

    with pytest.raises(ValueError, match="valid patch.*3 attempts"):
        data_utils.get_random_patch(
            image,
            image,
            patch_size=4,
            mask=mask,
            max_attempts=3,
            rng=np.random.default_rng(7),
        )


def test_random_patch_rejects_non_finite_data() -> None:
    first = np.full((1, 4, 4), np.nan, dtype=np.float32)
    second = np.ones_like(first)

    with pytest.raises(ValueError, match="finite valid patch"):
        data_utils.get_random_patch(
            first,
            second,
            patch_size=4,
            mask=np.ones((4, 4), dtype=bool),
            max_attempts=1,
            rng=np.random.default_rng(0),
        )


def test_random_patch_masks_and_zero_fills_isolated_non_finite_pixel() -> None:
    first = np.ones((1, 4, 4), dtype=np.float32)
    first[0, 1, 2] = np.nan
    second = np.ones_like(first)

    first_patch, _, patch_mask = data_utils.get_random_patch(
        first,
        second,
        patch_size=4,
        mask=np.ones((4, 4), dtype=bool),
        max_attempts=1,
        rng=np.random.default_rng(0),
    )

    assert not patch_mask[1, 2]
    assert first_patch[0, 1, 2] == 0
    assert np.isfinite(first_patch).all()


def test_augmentation_transforms_mask_before_brightness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random_values = iter([0.0, 1.0, 1.0, 0.0, 0.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    monkeypatch.setattr(np.random, "uniform", lambda _low, _high: 0.2)
    in_data = torch.zeros((1, 2, 3), dtype=torch.float32)
    out_data = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    mask = torch.tensor([[True, False, False], [False, False, False]])

    augmented_input, augmented_output = data_utils.augment_data(
        in_data, out_data, mask, prob=0.5
    )

    expected_input = torch.zeros_like(in_data)
    expected_input[0, 0, 2] = 0.2
    torch.testing.assert_close(augmented_input, expected_input)
    torch.testing.assert_close(augmented_output, torch.flip(out_data, dims=[2]))


def test_seeded_evaluation_is_deterministic_per_index_without_global_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = np.broadcast_to(
        np.arange(64, dtype=np.float32).reshape(1, 8, 8) / 64,
        (4, 8, 8),
    ).copy()
    mask = np.ones((8, 8), dtype=bool)
    monkeypatch.setattr(data_utils, "generate_dataset_mask", lambda _comune: mask)
    monkeypatch.setattr(
        data_utils,
        "get_super_resolution_stack",
        lambda _comune: ((grid, grid), (grid, grid)),
    )
    dataset = data_utils.SuperResolutionDataset(
        COMUNE,
        scale=1,
        patch_size=3,
        num_patches=10,
        to_augment=False,
        evaluation_seed=123,
    )
    np.random.seed(99)
    state_before = np.random.get_state()

    first = dataset[4]
    np.random.random(20)
    second = dataset[4]

    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    state_after = np.random.get_state()
    np.random.set_state(state_before)
    np.random.random(20)
    expected_state = np.random.get_state()
    assert state_after[0] == expected_state[0]
    np.testing.assert_array_equal(state_after[1], expected_state[1])
    assert state_after[2:] == expected_state[2:]


def test_seeded_evaluation_rejects_stochastic_augmentation() -> None:
    with pytest.raises(ValueError, match="evaluation_seed.*to_augment=False"):
        data_utils.SuperResolutionDataset(
            COMUNE,
            scale=1,
            evaluation_seed=123,
            to_augment=True,
        )

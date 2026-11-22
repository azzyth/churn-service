"""Test feature pipeline: bentuk output, tidak ada NaN, kategori tak dikenal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_preprocessor,
    make_xy,
)


@pytest.fixture(scope="module")
def train_frame(project_config):
    from src.config import project_path

    return pd.read_csv(project_path(project_config, "splits_dir") / "train.csv").head(500)


@pytest.fixture(scope="module")
def project_config():
    from src.config import load_config

    return load_config()


def test_feature_lists_are_disjoint_and_complete():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    assert set(FEATURE_COLUMNS) == set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    assert "customerID" not in FEATURE_COLUMNS
    assert "Churn" not in FEATURE_COLUMNS


def test_preprocessor_output_has_no_nan(train_frame):
    preprocessor = build_preprocessor()
    matrix = preprocessor.fit_transform(train_frame[FEATURE_COLUMNS])
    assert matrix.shape[0] == len(train_frame)
    assert matrix.shape[1] > len(FEATURE_COLUMNS)  # one-hot menambah kolom
    assert not np.isnan(matrix).any()


def test_make_xy_separates_target(train_frame, project_config):
    X, y = make_xy(train_frame, project_config)
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y)


def test_unknown_category_does_not_crash(train_frame):
    preprocessor = build_preprocessor()
    preprocessor.fit(train_frame[FEATURE_COLUMNS])

    unseen = train_frame[FEATURE_COLUMNS].head(1).copy()
    unseen["Contract"] = "Three year"  # nilai yang tidak ada di data training
    transformed = preprocessor.transform(unseen)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_tree_pipeline_without_scaling_keeps_shape(train_frame):
    """Varian tanpa scaling (untuk model pohon) harus punya dimensi sama."""
    from src.features.build import build_preprocessor as build

    scaled = build(scale_numeric=True).fit_transform(train_frame[FEATURE_COLUMNS])
    unscaled = build(scale_numeric=False).fit_transform(train_frame[FEATURE_COLUMNS])
    assert scaled.shape == unscaled.shape

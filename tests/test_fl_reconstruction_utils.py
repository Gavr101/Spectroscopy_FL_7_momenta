"""Tests for FL reconstruction utilities."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from fl_reconstruction_utils import (
    BASIS_TYPES,
    DEFAULT_RECONSTRUCTION_ORDERS,
    FOURIER_RECONSTRUCTION_ORDERS,
    RECONSTRUCTION_BASIS_TYPES,
    coefficient_count,
    cosine_fidelity,
    get_reconstruction_orders,
    get_transformer,
    load_fidelity_summary,
    load_fl_map,
    plot_fidelity_summary,
    reconstruct_map,
)


def test_load_fl_map_shape_and_nonnegative_values() -> None:
    """Check that loading cleans negative intensities and preserves shape."""

    map_data = load_fl_map(1)

    assert map_data.shape == (27, 201)
    assert np.isfinite(map_data).all()
    assert np.min(map_data) >= 0


def test_cosine_fidelity_identical_nonzero_arrays() -> None:
    """Check cosine fidelity for identical nonzero arrays."""

    values = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert np.isclose(cosine_fidelity(values, values), 1.0)


def test_coefficient_count_matches_transformer_components() -> None:
    """Check that coefficient counts match transformer metadata."""

    order = 2

    for basis_type in BASIS_TYPES:
        assert coefficient_count(basis_type, order) == get_transformer(
            basis_type,
            order,
        ).n_components


def test_reconstruct_map_preserves_shape_for_all_bases() -> None:
    """Check reconstruction output shape for every basis."""

    map_data = load_fl_map(1)

    for basis_type in RECONSTRUCTION_BASIS_TYPES:
        reconstructed = reconstruct_map(map_data, basis_type, order=1)
        assert reconstructed.shape == map_data.shape
        assert np.isfinite(reconstructed).all()


def test_reconstruction_basis_types_exclude_momentum() -> None:
    """Check that reconstruction experiments exclude polynomial moments."""

    assert "momentum" not in RECONSTRUCTION_BASIS_TYPES


def test_reconstruction_orders_by_basis() -> None:
    """Check Fourier and non-Fourier reconstruction order ranges."""

    assert get_reconstruction_orders("fourier") == FOURIER_RECONSTRUCTION_ORDERS
    assert FOURIER_RECONSTRUCTION_ORDERS == tuple(range(0, 8))
    assert get_reconstruction_orders("legendre") == DEFAULT_RECONSTRUCTION_ORDERS
    assert get_reconstruction_orders("zernike") == DEFAULT_RECONSTRUCTION_ORDERS
    assert DEFAULT_RECONSTRUCTION_ORDERS == tuple(range(0, 18))


def test_plot_fidelity_summary_uses_linear_fidelity_axis() -> None:
    """Check that fidelity summary uses a linear axis capped at one."""

    summary = {
        "legendre": [
            {"order": 0, "n_coefficients": 1, "mean": 0.9, "std": 0.01},
            {"order": 1, "n_coefficients": 3, "mean": 0.99, "std": 0.002},
        ],
    }
    labels = {
        "title": "Test",
        "x": "Number of coefficients",
        "y": "Cosine fidelity",
    }
    basis_labels = {"legendre": "Legendre"}

    fig = plot_fidelity_summary(summary, labels=labels, basis_labels=basis_labels)

    assert fig.axes[0].get_yscale() == "linear"
    assert np.isclose(fig.axes[0].get_ylim()[1], 1.0)
    plotted_y = fig.axes[0].lines[0].get_ydata()
    assert np.allclose(plotted_y, [0.9, 0.99])
    plt.close(fig)


def test_plot_fidelity_summary_limits_non_fourier_orders() -> None:
    """Check that plotting limits non-Fourier bases to order 13."""

    summary = {
        "legendre": [
            {"order": 13, "n_coefficients": 105, "mean": 0.95, "std": 0.01},
            {"order": 14, "n_coefficients": 120, "mean": 0.96, "std": 0.01},
        ],
        "fourier": [
            {"order": 8, "n_coefficients": 145, "mean": 0.97, "std": 0.01},
        ],
    }
    labels = {
        "title": "Test",
        "x": "Number of coefficients",
        "y": "Cosine fidelity",
    }
    basis_labels = {"legendre": "Legendre", "fourier": "Fourier"}

    fig = plot_fidelity_summary(summary, labels=labels, basis_labels=basis_labels)

    plotted_x = [line.get_xdata().tolist() for line in fig.axes[0].lines]
    assert [105] in plotted_x
    assert [120] not in plotted_x
    assert [145] in plotted_x
    plt.close(fig)


def test_load_fidelity_summary_reads_json(tmp_path) -> None:
    """Check that cached fidelity summary is loaded from JSON."""

    expected = {
        "legendre": [
            {"order": 0, "n_coefficients": 1, "mean": 0.9, "std": 0.01},
        ],
    }
    cache_path = tmp_path / "summary.json"
    cache_path.write_text(json.dumps(expected), encoding="utf-8")

    assert load_fidelity_summary(cache_path) == expected

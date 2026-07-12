"""Utilities for FL map reconstruction experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src import (
    FourierTransformer2D,
    LegendreTransformer2D,
    MomentumTransformer2D,
    ZernikeTransformer2D,
)


DATASET_DIR = Path("CD_HM_dataset")
RESULTS_DIR = Path("fl_reconstruction_results")
EMISSION_WAVELENGTHS = np.arange(375, 576, 1)
EXCITATION_WAVELENGTHS = np.arange(280, 411, 5)
ORDERS = (1, 2, 4, 8, 16)
BASIS_TYPES = ("momentum", "legendre", "fourier", "zernike")
RECONSTRUCTION_BASIS_TYPES = ("legendre", "fourier", "zernike")
FOURIER_RECONSTRUCTION_ORDERS = tuple(range(0, 8))
DEFAULT_RECONSTRUCTION_ORDERS = tuple(range(0, 18))
PLOT_DEFAULT_MAX_ORDER = 13

COLORS = {
    "momentum": "#1f77b4",
    "legendre": "#ff7f0e",
    "fourier": "#2ca02c",
    "zernike": "#d62728",
}

BASIS_LABELS_EN = {
    "momentum": "Polynomial",
    "legendre": "Legendre",
    "fourier": "Fourier",
    "zernike": "Zernike",
}

BASIS_LABELS_RU = {
    "momentum": "Полиномиальный",
    "legendre": "Лежандр",
    "fourier": "Фурье",
    "zernike": "Цернике",
}


def set_matplotlib_style() -> None:
    """Apply the shared matplotlib style used by project figures."""

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["axes.titlesize"] = 18
    mpl.rcParams["axes.labelsize"] = 15
    mpl.rcParams["legend.fontsize"] = 14
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12


def load_fl_map(
    map_id: int,
    dataset_dir: Path | str = DATASET_DIR,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Load one photoluminescence excitation-emission matrix.

    Args:
        map_id: Numeric map identifier, for example ``1`` for ``1.csv``.
        dataset_dir: Directory with FL map CSV files.
        fill_value: Value used after replacing negative intensities with NaN.

    Returns:
        Array of shape ``(27, 201)`` with cleaned intensity values.
    """

    csv_path = Path(dataset_dir) / f"{map_id}.csv"
    data_frame = pd.read_csv(csv_path)
    data_frame = data_frame.astype(float)
    data_frame[data_frame < 0] = np.nan
    data_frame = data_frame.drop("EX Wavelength/EM Wavelength", axis=1)
    return np.nan_to_num(
        data_frame.to_numpy(dtype=float, copy=True),
        nan=fill_value,
        posinf=fill_value,
        neginf=fill_value,
    )


def load_all_fl_maps(
    dataset_dir: Path | str = DATASET_DIR,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """Load all numeric FL map CSV files from a dataset directory.

    Args:
        dataset_dir: Directory with files named like ``1.csv``.
        show_progress: Whether to show a tqdm progress bar.

    Returns:
        Tuple with stacked maps of shape ``(n_maps, 27, 201)`` and map IDs.
    """

    map_paths = sorted(
        Path(dataset_dir).glob("*.csv"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else 10**9,
    )
    numeric_paths = [path for path in map_paths if path.stem.isdigit()]
    map_ids = [int(path.stem) for path in numeric_paths]
    iterator = (
        tqdm(map_ids, desc="Loading FL maps", unit="map")
        if show_progress
        else map_ids
    )
    maps = [load_fl_map(map_id, dataset_dir=dataset_dir) for map_id in iterator]
    return np.stack(maps, axis=0), map_ids


def get_transformer(basis_type: str, order: int) -> object:
    """Create a transformer for a basis and order.

    Args:
        basis_type: One of ``momentum``, ``legendre``, ``fourier``, ``zernike``.
        order: Maximum expansion order.

    Returns:
        Transformer instance from ``src.py``.
    """

    if basis_type == "momentum":
        return MomentumTransformer2D(order=order)
    if basis_type == "legendre":
        return LegendreTransformer2D(order=order)
    if basis_type == "fourier":
        return FourierTransformer2D(order=order)
    if basis_type == "zernike":
        return ZernikeTransformer2D(
            order=order,
            x_bounds=(
                float(EMISSION_WAVELENGTHS.min()),
                float(EMISSION_WAVELENGTHS.max()),
            ),
            y_bounds=(
                float(EXCITATION_WAVELENGTHS.min()),
                float(EXCITATION_WAVELENGTHS.max()),
            ),
        )
    raise ValueError(f"Unknown basis type: {basis_type}")


def transform_coefficients(maps: np.ndarray, basis_type: str, order: int) -> np.ndarray:
    """Compute expansion coefficients for one basis.

    Args:
        maps: Input maps with shape ``(batch, height, width)`` or ``(height, width)``.
        basis_type: Basis name.
        order: Maximum expansion order.

    Returns:
        Coefficient matrix of shape ``(batch, n_components)``.
    """

    batch_maps = np.asarray(maps, dtype=float)
    if batch_maps.ndim == 2:
        batch_maps = batch_maps[None, :, :]
    transformer = get_transformer(basis_type, order)
    return transformer.fit_transform(batch_maps)


def reconstruct_map(map_data: np.ndarray, basis_type: str, order: int) -> np.ndarray:
    """Reconstruct one FL map after basis decomposition.

    Args:
        map_data: Input FL map with shape ``(27, 201)``.
        basis_type: Basis name.
        order: Maximum expansion order.

    Returns:
        Reconstructed map with the same shape as ``map_data``.
    """

    sample = np.asarray(map_data, dtype=float)
    transformer = get_transformer(basis_type, order)
    coeffs = transformer.fit_transform(sample[None, :, :])[0]

    if basis_type == "momentum":
        return reconstruct_momentum(sample, transformer, coeffs)
    if basis_type == "legendre":
        return reconstruct_legendre(sample.shape, transformer, coeffs)
    if basis_type == "fourier":
        return reconstruct_fourier(sample.shape, transformer, coeffs)
    if basis_type == "zernike":
        return reconstruct_zernike(sample.shape, transformer, coeffs)
    raise ValueError(f"Unknown basis type: {basis_type}")


def reconstruct_batch(maps: np.ndarray, basis_type: str, order: int) -> np.ndarray:
    """Reconstruct a batch of FL maps.

    Args:
        maps: Input maps with shape ``(batch, height, width)``.
        basis_type: Basis name.
        order: Maximum expansion order.

    Returns:
        Reconstructed maps with the same shape as ``maps``.
    """

    return np.stack(
        [reconstruct_map(map_data, basis_type, order) for map_data in maps],
        axis=0,
    )


def reconstruct_legendre(
    shape: tuple[int, int],
    transformer: LegendreTransformer2D,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Reconstruct a 2D map from Legendre coefficients.

    Args:
        shape: Target ``(height, width)``.
        transformer: Fitted Legendre transformer.
        coeffs: One-dimensional coefficient array.

    Returns:
        Reconstructed 2D map.
    """

    height, width = shape
    x_norm = np.linspace(-1, 1, width)
    y_norm = np.linspace(-1, 1, height)
    reconstructed = np.zeros(shape, dtype=float)
    for index, (m_order, n_order) in enumerate(transformer.order_pairs):
        p_m = np.polynomial.legendre.Legendre.basis(m_order)(x_norm)[None, :]
        p_n = np.polynomial.legendre.Legendre.basis(n_order)(y_norm)[:, None]
        reconstructed += coeffs[index] * p_m * p_n
    return reconstructed


def reconstruct_fourier(
    shape: tuple[int, int],
    transformer: FourierTransformer2D,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Reconstruct a 2D map from Fourier coefficients.

    Args:
        shape: Target ``(height, width)``.
        transformer: Fitted Fourier transformer.
        coeffs: One-dimensional coefficient array.

    Returns:
        Reconstructed 2D map.
    """

    height, width = shape
    x_norm = np.linspace(-1, 1, width)
    y_norm = np.linspace(-1, 1, height)
    reconstructed = np.zeros(shape, dtype=float)
    index = 0
    for m_order, n_order in transformer.order_pairs:
        a_mn = coeffs[index]
        index += 1

        cos_mx = np.cos(m_order * np.pi * x_norm)[None, :]
        sin_mx = np.sin(m_order * np.pi * x_norm)[None, :]
        cos_ny = np.cos(n_order * np.pi * y_norm)[:, None]
        sin_ny = np.sin(n_order * np.pi * y_norm)[:, None]

        if m_order == 0 and n_order == 0:
            reconstructed += a_mn / 4
        elif m_order == 0 or n_order == 0:
            reconstructed += 0.5 * a_mn * cos_mx * cos_ny
        else:
            reconstructed += a_mn * cos_mx * cos_ny

        if n_order != 0:
            b_mn = coeffs[index]
            index += 1
            if m_order == 0:
                reconstructed += 0.5 * b_mn * sin_ny
            else:
                reconstructed += b_mn * cos_mx * sin_ny

        if m_order != 0:
            c_mn = coeffs[index]
            index += 1
            if n_order == 0:
                reconstructed += 0.5 * c_mn * sin_mx
            else:
                reconstructed += c_mn * sin_mx * cos_ny

        if m_order != 0 and n_order != 0:
            d_mn = coeffs[index]
            index += 1
            reconstructed += d_mn * sin_mx * sin_ny

    return reconstructed


def reconstruct_zernike(
    shape: tuple[int, int],
    transformer: ZernikeTransformer2D,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Reconstruct a 2D map from Zernike coefficients.

    Args:
        shape: Target ``(height, width)``.
        transformer: Fitted Zernike transformer with cached center/radius.
        coeffs: One-dimensional coefficient array.

    Returns:
        Reconstructed 2D map.
    """

    height, width = shape
    x_values = np.linspace(transformer.x_bounds[0], transformer.x_bounds[1], width)
    y_values = np.linspace(transformer.y_bounds[0], transformer.y_bounds[1], height)
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    center_x, center_y = transformer.last_centers[0]
    radius = transformer.last_radii[0]
    reconstructed = np.zeros(shape, dtype=float)
    for index, (n_order, m_order) in enumerate(transformer.modes):
        basis = transformer._zernike_poly(
            n_order,
            m_order,
            x_grid,
            y_grid,
            center_x,
            center_y,
            radius,
        )
        reconstructed += coeffs[index] * basis
    return reconstructed


def reconstruct_momentum(
    map_data: np.ndarray,
    transformer: MomentumTransformer2D,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Reconstruct a 2D map from polynomial moment constraints.

    Args:
        map_data: Original map, used only for target shape.
        transformer: Fitted momentum transformer.
        coeffs: One-dimensional moment feature array.

    Returns:
        Least-squares reconstruction on the original grid.
    """

    height, width = map_data.shape
    x_values = np.linspace(transformer.x_min, transformer.x_max, width)
    y_values = np.linspace(transformer.y_min, transformer.y_max, height)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    dx_dy = (transformer.x_max - transformer.x_min) / width
    dx_dy *= (transformer.y_max - transformer.y_min) / height

    center_x = coeffs[1] if len(coeffs) > 1 else 0.0
    center_y = coeffs[2] if len(coeffs) > 2 else 0.0
    rows = []
    targets = []

    for index, (x_order, y_order) in enumerate(transformer.order_pairs):
        if x_order + y_order == 0:
            basis = np.ones_like(x_grid)
            target = coeffs[index]
        elif x_order == 1 and y_order == 0:
            basis = x_grid
            target = coeffs[index]
        elif x_order == 0 and y_order == 1:
            basis = y_grid
            target = coeffs[index]
        else:
            basis = (x_grid - center_x) ** x_order * (y_grid - center_y) ** y_order
            power = x_order + y_order
            target = np.sign(coeffs[index]) * np.abs(coeffs[index]) ** power
        rows.append((basis * dx_dy).ravel())
        targets.append(target)

    matrix = np.vstack(rows)
    target_vector = np.asarray(targets, dtype=float)
    solution, *_ = np.linalg.lstsq(matrix, target_vector, rcond=None)
    return solution.reshape(height, width)


def cosine_fidelity(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute cosine similarity between two maps.

    Args:
        original: Original map.
        reconstructed: Reconstructed map.

    Returns:
        Cosine similarity. Returns ``0.0`` when either norm is zero.
    """

    left = np.asarray(original, dtype=float).ravel()
    right = np.asarray(reconstructed, dtype=float).ravel()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def coefficient_count(basis_type: str, order: int) -> int:
    """Get the number of coefficients generated by a basis/order pair.

    Args:
        basis_type: Basis name.
        order: Maximum expansion order.

    Returns:
        Number of output coefficients.
    """

    return int(get_transformer(basis_type, order).n_components)


def get_reconstruction_orders(basis_type: str) -> tuple[int, ...]:
    """Get reconstruction orders for a basis.

    Args:
        basis_type: Basis name.

    Returns:
        Fourier orders ``0..7`` and default orders ``0..17`` otherwise.
    """

    if basis_type == "fourier":
        return FOURIER_RECONSTRUCTION_ORDERS
    return DEFAULT_RECONSTRUCTION_ORDERS


def load_fidelity_summary(cache_path: Path | str) -> dict[str, list[dict[str, float]]]:
    """Load cached fidelity summary from JSON.

    Args:
        cache_path: Path to a JSON summary file.

    Returns:
        Nested summary keyed by basis type.
    """

    return json.loads(Path(cache_path).read_text(encoding="utf-8"))


def compute_fidelity_summary(
    orders: Iterable[int] | None = None,
    basis_types: Iterable[str] = RECONSTRUCTION_BASIS_TYPES,
    dataset_dir: Path | str = DATASET_DIR,
    cache_path: Path | str | None = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> dict[str, list[dict[str, float]]]:
    """Compute mean and standard deviation of reconstruction fidelity.

    Args:
        orders: Expansion orders to evaluate. If None, use basis-specific
            reconstruction orders.
        basis_types: Basis names to evaluate.
        dataset_dir: Directory with FL maps.
        cache_path: Optional JSON path for storing/loading results.
        use_cache: Whether to load existing cached results.
        verbose: Whether to print progress messages.

    Returns:
        Nested summary keyed by basis type.
    """

    if cache_path is not None:
        cache_file = Path(cache_path)
        if use_cache and cache_file.exists():
            if verbose:
                print(
                    f"Loading cached fidelity summary from {cache_file}",
                    flush=True,
                )
            return json.loads(cache_file.read_text(encoding="utf-8"))

    dataset_path = Path(dataset_dir)
    numeric_paths = sorted(
        (path for path in dataset_path.glob("*.csv") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if verbose:
        print(f"Loading FL maps from {dataset_path}", flush=True)
        print(f"Found {len(numeric_paths)} numeric FL map files", flush=True)

    maps, _ = load_all_fl_maps(
        dataset_dir=dataset_path,
        show_progress=verbose,
    )
    if verbose:
        print(f"Loaded {len(maps)} FL maps with shape {maps.shape}", flush=True)
    summary: dict[str, list[dict[str, float]]] = {}
    explicit_orders = tuple(orders) if orders is not None else None

    for basis_type in basis_types:
        summary[basis_type] = []
        basis_orders = (
            explicit_orders
            if explicit_orders is not None
            else get_reconstruction_orders(basis_type)
        )
        for order in basis_orders:
            n_coefficients = coefficient_count(basis_type, order)
            if verbose:
                print(
                    f"Computing fidelity: basis={basis_type}, "
                    f"order={order}, n_coefficients={n_coefficients}",
                    flush=True,
                )
            scores = []
            progress_label = f"{basis_type}, order={order}"
            iterator = (
                tqdm(maps, desc=progress_label, unit="map")
                if verbose
                else maps
            )
            for map_data in iterator:
                reconstructed = reconstruct_map(map_data, basis_type, order)
                scores.append(cosine_fidelity(map_data, reconstructed))
            scores_array = np.asarray(scores, dtype=float)
            summary[basis_type].append(
                {
                    "order": int(order),
                    "n_coefficients": n_coefficients,
                    "mean": float(np.mean(scores_array)),
                    "std": float(np.std(scores_array)),
                }
            )

    if cache_path is not None:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if verbose:
            print(f"Saved fidelity summary to {cache_file}", flush=True)

    return summary


def plot_reconstruction_pair(
    original: np.ndarray,
    reconstructed: np.ndarray,
    title: str,
    labels: dict[str, str],
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Plot original and reconstructed FL maps side by side.

    Args:
        original: Original FL map.
        reconstructed: Reconstructed FL map.
        title: Figure title.
        labels: Text labels for title, axes, and panel names.
        figsize: Figure size in inches.

    Returns:
        Matplotlib figure.
    """

    set_matplotlib_style()
    vmin = float(np.nanmin(original))
    vmax = float(np.nanmax(original))
    extent = [
        float(EMISSION_WAVELENGTHS.min()),
        float(EMISSION_WAVELENGTHS.max()),
        float(EXCITATION_WAVELENGTHS.min()),
        float(EXCITATION_WAVELENGTHS.max()),
    ]

    fig = plt.figure(figsize=figsize)
    grid_spec = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=(1, 1, 0.05),
        wspace=0.20,
    )
    left_axis = fig.add_subplot(grid_spec[0, 0])
    right_axis = fig.add_subplot(
        grid_spec[0, 1],
        sharex=left_axis,
        sharey=left_axis,
    )
    colorbar_axis = fig.add_subplot(grid_spec[0, 2])
    axes = np.array([left_axis, right_axis])
    fig.suptitle(title, fontsize=18, fontweight="bold")

    for axis_index, (axis, data, panel_title) in enumerate(
        zip(
            axes,
            (original, reconstructed),
            (labels["original"], labels["reconstructed"]),
        )
    ):
        image = axis.imshow(
            data,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(panel_title)
        axis.set_xlabel(labels["emission"])
        if axis_index == 0:
            axis.set_ylabel(labels["excitation"])
        else:
            axis.set_ylabel("")
            axis.tick_params(axis="y", labelleft=False)
        axis.tick_params(direction="out", length=5, width=1.5)
        for spine in axis.spines.values():
            spine.set_linewidth(1.5)

    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(labels["intensity"])
    fig.subplots_adjust(top=0.84)
    return fig


def plot_fidelity_summary(
    summary: dict[str, list[dict[str, float]]],
    labels: dict[str, str],
    basis_labels: dict[str, str],
    figsize: tuple[float, float] = (10, 6),
    default_max_order: int = PLOT_DEFAULT_MAX_ORDER,
) -> plt.Figure:
    """Plot fidelity as a function of coefficient count.

    Args:
        summary: Output of ``compute_fidelity_summary``.
        labels: Axis and title labels.
        basis_labels: Human-readable basis labels.
        figsize: Figure size in inches.
        default_max_order: Maximum plotted order for non-Fourier bases.

    Returns:
        Matplotlib figure.
    """

    set_matplotlib_style()
    fig, axis = plt.subplots(figsize=figsize)

    for basis_type in RECONSTRUCTION_BASIS_TYPES:
        basis_summary = summary.get(basis_type, [])
        if basis_type != "fourier":
            basis_summary = [
                item for item in basis_summary
                if item["order"] <= default_max_order
            ]
        if len(basis_summary) == 0:
            continue
        x_values = [item["n_coefficients"] for item in basis_summary]
        y_values = np.asarray([item["mean"] for item in basis_summary], dtype=float)
        std_values = np.asarray([item["std"] for item in basis_summary], dtype=float)
        lower_errors = np.minimum(std_values, np.maximum(y_values, 0.0))
        upper_errors = np.minimum(std_values, np.maximum(1.0 - y_values, 0.0))
        y_errors = np.vstack([lower_errors, upper_errors])
        axis.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            marker="o",
            lw=1.8,
            ms=5,
            capsize=3,
            color=COLORS[basis_type],
            label=basis_labels[basis_type],
        )

    axis.set_title(labels["title"], fontweight="bold")
    axis.set_xlabel(labels["x"])
    axis.set_ylabel(labels["y"])
    axis.set_ylim(top=1.0)
    axis.grid(which="major", axis="both", linestyle="--", linewidth=0.7, alpha=0.6)
    axis.legend(frameon=True)
    fig.tight_layout()
    return fig

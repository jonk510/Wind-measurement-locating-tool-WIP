"""
Wind Measurement Location Optimiser — Streamlit web app
========================================================
Run with:  streamlit run app.py
"""

import io
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import streamlit as st

warnings.filterwarnings("ignore")

try:
    import requests
    from pyproj import Transformer as _ProjTransformer
    _HAS_DEM = True
except ImportError:
    _HAS_DEM = False


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TERRAIN_COEFFICIENTS = {
    "simple":   {"horizontal": 0.000005, "vertical": 0.0003},
    "moderate": {"horizontal": 0.00001,  "vertical": 0.00065},
    "complex":  {"horizontal": 0.000015, "vertical": 0.001},
}

INSTRUMENT_TYPES = {
    "mast":  {"label": "Met Mast", "cost_aud": 400_000, "meas_uncertainty_pct": 1.0},
    "lidar": {"label": "LiDAR",    "cost_aud": 240_000, "meas_uncertainty_pct": 2.0},
    "sodar": {"label": "SoDAR",    "cost_aud":  90_000, "meas_uncertainty_pct": 3.0},
}

_IEC_THRESHOLDS = [
    ("simple",   {"max": 3.0,  "p90": 2.0}),
    ("moderate", {"max": 10.0, "p90": 7.0}),
    ("complex",  {"max": 999,  "p90": 999}),
]


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty functions
# ──────────────────────────────────────────────────────────────────────────────

def horiz_uncertainty_pct(distances_m, h_coeff):
    return distances_m * h_coeff * 100.0


def vert_uncertainty_pct(delta_z_m, v_coeff):
    return np.abs(delta_z_m) * v_coeff * 100.0


# ──────────────────────────────────────────────────────────────────────────────
# IEC 61400-1 terrain assessment
# ──────────────────────────────────────────────────────────────────────────────

def assess_terrain_complexity(xyz, wtg, assessment_radius_m=2000.0,
                               grid_resolution=200):
    pts    = xyz[["X", "Y"]].values.astype(float)
    vals   = xyz["Z"].values.astype(float)
    wtg_xy = wtg[["X", "Y"]].values.astype(float)

    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    xi = np.linspace(xmin, xmax, grid_resolution)
    yi = np.linspace(ymin, ymax, grid_resolution)
    xx, yy = np.meshgrid(xi, yi)
    elev = griddata(pts, vals, (xx, yy), method="linear")

    dx = (xmax - xmin) / (grid_resolution - 1)
    dy = (ymax - ymin) / (grid_resolution - 1)
    dz_dx, dz_dy = np.gradient(elev, dx, dy)
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))

    centroid = wtg_xy.mean(axis=0)
    dist_to_centroid = np.sqrt((xx - centroid[0]) ** 2 +
                               (yy - centroid[1]) ** 2)
    zone = (dist_to_centroid <= assessment_radius_m) & np.isfinite(slope_deg)
    if zone.sum() < 10:
        zone = np.isfinite(slope_deg)

    s = slope_deg[zone]
    stats = {
        "mean_slope_deg":      float(np.mean(s)),
        "max_slope_deg":       float(np.max(s)),
        "p90_slope_deg":       float(np.percentile(s, 90)),
        "p50_slope_deg":       float(np.percentile(s, 50)),
        "assessment_radius_m": float(assessment_radius_m),
        "n_grid_pts_assessed": int(zone.sum()),
    }
    for label, thresh in _IEC_THRESHOLDS:
        if (stats["max_slope_deg"] <= thresh["max"] and
                stats["p90_slope_deg"] <= thresh["p90"]):
            return label, stats
    return "complex", stats


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis class
# ──────────────────────────────────────────────────────────────────────────────

class WindMeasurementAnalyser:

    def __init__(self, xyz, wtg, n_clusters, terrain="moderate",
                 existing_locs=None, existing_types=None,
                 new_instrument_counts=None):
        self.xyz = xyz
        self.wtg = wtg

        if existing_locs is not None and len(existing_locs) > 0:
            self.existing_locs = np.atleast_2d(existing_locs).astype(float)
            self.n_existing    = len(self.existing_locs)
        else:
            self.existing_locs = None
            self.n_existing    = 0

        if self.n_existing > 0:
            self._existing_types = (
                [t.lower().strip() for t in existing_types]
                if existing_types is not None
                else ["mast"] * self.n_existing
            )
        else:
            self._existing_types = []

        if new_instrument_counts is not None:
            self._new_instrument_counts = {
                k: int(v) for k, v in new_instrument_counts.items()}
            self.n_new = sum(self._new_instrument_counts.values())
        else:
            self.n_new = n_clusters
            self._new_instrument_counts = {
                "mast": n_clusters, "lidar": 0, "sodar": 0}

        self.n = self.n_existing + self.n_new

        terrain = terrain.lower().strip()
        if terrain not in TERRAIN_COEFFICIENTS:
            raise ValueError(f"terrain must be one of {list(TERRAIN_COEFFICIENTS)}")
        self.terrain = terrain
        self.h_coeff = TERRAIN_COEFFICIENTS[terrain]["horizontal"]
        self.v_coeff = TERRAIN_COEFFICIENTS[terrain]["vertical"]

        self.labels     = None
        self.meas_locs  = None
        self.meas_elevs = None
        self.wtg_elevs  = None
        self.mast_types = None

        self._build_elev_interp()
        self._build_slope_interp()

    def _build_elev_interp(self):
        pts  = self.xyz[["X", "Y"]].values.astype(float)
        vals = self.xyz["Z"].values.astype(float)
        self._elev_pts  = pts
        self._elev_vals = vals
        self._elev_tree = cKDTree(pts)

    def get_elevation(self, xy):
        xy   = np.atleast_2d(xy).astype(float)
        elev = griddata(self._elev_pts, self._elev_vals, xy, method="linear")
        mask = np.isnan(elev)
        if mask.any():
            _, idx = self._elev_tree.query(xy[mask])
            elev[mask] = self._elev_vals[idx]
        return elev

    def _build_slope_interp(self, grid_resolution=200):
        pts  = self._elev_pts
        vals = self._elev_vals
        xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
        ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        xi = np.linspace(xmin, xmax, grid_resolution)
        yi = np.linspace(ymin, ymax, grid_resolution)
        xx, yy = np.meshgrid(xi, yi)
        elev = griddata(pts, vals, (xx, yy), method="linear")
        dx = (xmax - xmin) / (grid_resolution - 1)
        dy = (ymax - ymin) / (grid_resolution - 1)
        dz_dx, dz_dy = np.gradient(elev, dx, dy)
        slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
        flat_pts  = np.column_stack([xx.ravel(), yy.ravel()])
        flat_slp  = slope.ravel()
        valid = np.isfinite(flat_slp)
        self._slope_pts  = flat_pts[valid]
        self._slope_vals = flat_slp[valid]
        self._slope_tree = cKDTree(self._slope_pts)

    def get_slope(self, xy):
        xy    = np.atleast_2d(xy).astype(float)
        slope = griddata(self._slope_pts, self._slope_vals, xy, method="linear")
        mask  = np.isnan(slope)
        if mask.any():
            _, idx = self._slope_tree.query(xy[mask])
            slope[mask] = self._slope_vals[idx]
        return slope

    def cluster_turbines(self):
        coords = self.wtg[["X", "Y"]].values.astype(float)
        if self.n_existing == 0:
            km = KMeans(n_clusters=self.n, random_state=42, n_init=20)
            km.fit(coords)
            tree = cKDTree(km.cluster_centers_)
            _, self.labels = tree.query(coords)
            return self.labels

        if self.n_new > 0:
            km = KMeans(n_clusters=self.n_new, random_state=42, n_init=20)
            km.fit(coords)
            new_centres = km.cluster_centers_
        else:
            new_centres = np.empty((0, 2))

        centres = np.vstack([self.existing_locs, new_centres])
        for _ in range(300):
            tree = cKDTree(centres)
            _, labels = tree.query(coords)
            prev = centres.copy()
            for k in range(self.n_existing, self.n):
                mask = labels == k
                if mask.sum() > 0:
                    centres[k] = coords[mask].mean(axis=0)
            if np.allclose(centres, prev, atol=1.0):
                break

        self.labels = labels
        return self.labels

    def optimise_measurement_locations(self):
        coords = self.wtg[["X", "Y"]].values.astype(float)
        self.meas_locs = np.zeros((self.n, 2))
        margin = 2000.0

        if self.n_existing > 0:
            self.meas_locs[:self.n_existing] = self.existing_locs

        for k in range(self.n_existing, self.n):
            cluster_xy = coords[self.labels == k]
            if len(cluster_xy) == 0:
                self.meas_locs[k] = coords.mean(axis=0)
                continue
            if len(cluster_xy) == 1:
                self.meas_locs[k] = cluster_xy[0]
                continue
            centroid = cluster_xy.mean(axis=0)
            bounds = [
                (cluster_xy[:, 0].min() - margin, cluster_xy[:, 0].max() + margin),
                (cluster_xy[:, 1].min() - margin, cluster_xy[:, 1].max() + margin),
            ]
            def _obj(xy, pts=cluster_xy):
                return float(np.sum(np.sum((pts - xy) ** 2, axis=1)))
            res = minimize(_obj, centroid, method="L-BFGS-B", bounds=bounds,
                           options={"ftol": 1e-6, "gtol": 1e-5})
            self.meas_locs[k] = res.x

        self.meas_elevs = self.get_elevation(self.meas_locs)
        self.wtg_elevs  = self.get_elevation(self.wtg[["X", "Y"]].values)
        self.assign_instrument_types()
        return self.meas_locs

    def assign_instrument_types(self):
        counts     = self._new_instrument_counts
        n_mast_new = counts.get("mast",  0)
        n_lidar    = counts.get("lidar", 0)

        new_indices = list(range(self.n_existing, self.n))
        n_new_total = len(new_indices)

        if n_new_total == 0:
            self.mast_types = list(self._existing_types)
            return self.mast_types

        new_positions = self.meas_locs[new_indices]
        wtg_xy        = self.wtg[["X", "Y"]].values.astype(float)
        centroid      = wtg_xy.mean(axis=0)

        cent_dists   = np.linalg.norm(new_positions - centroid, axis=1)
        wtg_slopes   = self.get_slope(wtg_xy)
        median_slope = float(np.nanmedian(wtg_slopes))
        mast_slopes  = self.get_slope(new_positions)
        terrain_rep  = np.abs(mast_slopes - median_slope)

        def _norm01(arr):
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo) if hi > lo + 1e-9 else np.zeros_like(arr)

        score = 0.6 * _norm01(cent_dists) + 0.4 * _norm01(terrain_rep)
        order = np.argsort(score)

        mast_set  = set(order[:n_mast_new].tolist())
        remaining = order[n_mast_new:].tolist()

        cluster_unc = {}
        for j in remaining:
            k    = new_indices[j]
            idxs = np.where(self.labels == k)[0]
            if len(idxs) > 0:
                h = np.array([horiz_uncertainty_pct(
                    np.linalg.norm(wtg_xy[i] - self.meas_locs[k]), self.h_coeff)
                    for i in idxs])
                v = np.array([vert_uncertainty_pct(
                    abs(self.wtg_elevs[i] - self.meas_elevs[k]), self.v_coeff)
                    for i in idxs])
                cluster_unc[j] = float(np.mean(np.sqrt(h ** 2 + v ** 2)))
            else:
                cluster_unc[j] = 0.0

        rem_sorted = sorted(remaining,
                            key=lambda j: cluster_unc[j], reverse=True)
        lidar_set  = set(rem_sorted[:n_lidar])

        new_types = []
        for j in range(n_new_total):
            if j in mast_set:
                new_types.append("mast")
            elif j in lidar_set:
                new_types.append("lidar")
            else:
                new_types.append("sodar")

        self.mast_types = list(self._existing_types) + new_types
        return self.mast_types

    def compute_uncertainties(self):
        coords = self.wtg[["X", "Y"]].values.astype(float)
        h_unc = np.zeros(len(coords))
        v_unc = np.zeros(len(coords))
        i_unc = np.zeros(len(coords))
        for i, (xy, lbl, elev) in enumerate(
                zip(coords, self.labels, self.wtg_elevs)):
            dist     = float(np.linalg.norm(xy - self.meas_locs[lbl]))
            dz       = float(abs(elev - self.meas_elevs[lbl]))
            h_unc[i] = horiz_uncertainty_pct(dist, self.h_coeff)
            v_unc[i] = vert_uncertainty_pct(dz, self.v_coeff)
            if self.mast_types:
                i_unc[i] = INSTRUMENT_TYPES[self.mast_types[lbl]
                                            ]["meas_uncertainty_pct"]
        return h_unc, v_unc, i_unc

    def build_heatmap_grids(self, resolution=150):
        xmin, xmax = self.xyz["X"].min(), self.xyz["X"].max()
        ymin, ymax = self.xyz["Y"].min(), self.xyz["Y"].max()
        xi = np.linspace(xmin, xmax, resolution)
        yi = np.linspace(ymin, ymax, resolution)
        xx, yy = np.meshgrid(xi, yi)
        grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

        elev_grid = griddata(self._elev_pts, self._elev_vals,
                             grid_pts, method="linear"
                             ).reshape(resolution, resolution)

        meas_tree = cKDTree(self.meas_locs)
        dists, nearest_k = meas_tree.query(grid_pts)

        grid_elevs  = self.get_elevation(grid_pts)
        meas_e_grid = self.meas_elevs[nearest_k]
        dz_grid     = np.abs(grid_elevs - meas_e_grid)

        h_grid = horiz_uncertainty_pct(dists, self.h_coeff
                                       ).reshape(resolution, resolution)
        v_grid = vert_uncertainty_pct(dz_grid, self.v_coeff
                                      ).reshape(resolution, resolution)

        if self.mast_types:
            inst_flat = np.array([
                INSTRUMENT_TYPES[self.mast_types[k]]["meas_uncertainty_pct"]
                for k in nearest_k
            ])
            inst_grid = inst_flat.reshape(resolution, resolution)
        else:
            inst_grid = np.zeros((resolution, resolution))

        return xx, yy, h_grid, v_grid, elev_grid, inst_grid


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

_CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#17becf",
    "#bcbd22", "#8c564b", "#e377c2", "#7f7f7f",
]


def _draw_terrain(ax, xx, yy, elev_grid, n_levels=18):
    valid = ~np.isnan(elev_grid)
    if not valid.any():
        return
    lo = np.nanpercentile(elev_grid, 2)
    hi = np.nanpercentile(elev_grid, 98)
    levels = np.linspace(lo, hi, n_levels)
    ax.contourf(xx, yy, elev_grid, levels=levels, cmap="terrain", alpha=0.30)
    ax.contour(xx, yy, elev_grid, levels=levels,
               colors="grey", linewidths=0.35, alpha=0.55)


def _format_axis(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
    ax.set_xlabel("Easting (m)", fontsize=8)
    ax.set_ylabel("Northing (m)", fontsize=8)
    ax.tick_params(labelsize=7)


def _scatter_masts(ax, meas_locs, meas_elevs=None, radius_m=5000,
                   n_existing=0, mast_types=None):
    for k, loc in enumerate(meas_locs):
        col = _CLUSTER_COLORS[k % len(_CLUSTER_COLORS)]
        if k < n_existing:
            marker, size = "s", 200
            tag = f"E{k + 1}"
        else:
            marker, size = "*", 320
            tag = f"M{k - n_existing + 1}"
        ax.scatter(*loc, marker=marker, s=size, c=col,
                   edgecolors="black", linewidths=0.9, zorder=7)
        parts = [tag]
        if mast_types is not None and k < len(mast_types):
            parts.append(INSTRUMENT_TYPES[mast_types[k]]["label"])
        if meas_elevs is not None:
            parts.append(f"{meas_elevs[k]:.0f} m")
        ax.annotate("\n".join(parts), xy=loc, xytext=(6, 4),
                    textcoords="offset points", fontsize=7,
                    fontweight="bold", color=col)
        ax.add_patch(Circle(loc, radius=radius_m, fill=False,
                            edgecolor=col, linewidth=1.2, linestyle="--",
                            alpha=0.7, zorder=6))


def build_figure(analyser, h_unc, v_unc, i_unc,
                 xx, yy, h_grid, v_grid, inst_grid, elev_grid):
    """Build and return the four-panel matplotlib figure."""
    wtg_xy     = analyser.wtg[["X", "Y"]].values.astype(float)
    labels     = analyser.labels
    meas_locs  = analyser.meas_locs
    n          = analyser.n
    n_existing = analyser.n_existing
    n_new      = analyser.n_new
    mast_types = analyser.mast_types

    if n_existing > 0:
        title_n = (f"N = {n_new} new + {n_existing} existing "
                   f"= {n} total mast{'s' if n > 1 else ''}")
    else:
        title_n = f"N = {n} measurement location{'s' if n > 1 else ''}"

    combined_grid = np.sqrt(h_grid ** 2 + v_grid ** 2 + inst_grid ** 2)
    combined_unc  = np.sqrt(h_unc  ** 2 + v_unc  ** 2 + i_unc   ** 2)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Wind Measurement Location Optimisation  —  {title_n}  |  "
        f"Terrain: {analyser.terrain}",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28,
                  left=0.07, right=0.97, top=0.93, bottom=0.05)

    # Panel 1 : Cluster map
    ax0 = fig.add_subplot(gs[0, 0])
    _draw_terrain(ax0, xx, yy, elev_grid)
    ax0.set_title("Turbine Clusters & Measurement Locations",
                  fontsize=10, fontweight="bold")
    for k in range(n):
        mask = labels == k
        col  = _CLUSTER_COLORS[k % len(_CLUSTER_COLORS)]
        for xy in wtg_xy[mask]:
            ax0.plot([xy[0], meas_locs[k, 0]], [xy[1], meas_locs[k, 1]],
                     color=col, lw=0.7, alpha=0.45, zorder=3)
        ax0.scatter(wtg_xy[mask, 0], wtg_xy[mask, 1],
                    c=col, s=55, marker="^", zorder=5,
                    edgecolors="black", linewidths=0.5,
                    label=f"Cluster {k+1}  ({mask.sum()} WTGs)")
    _scatter_masts(ax0, meas_locs, analyser.meas_elevs,
                   n_existing=n_existing, mast_types=mast_types)
    legend_extras = []
    if n_existing > 0:
        legend_extras.append(
            Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
                   markeredgecolor="black", markersize=10, label="Existing mast"))
    legend_extras += [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=12,
               label="New mast (optimised)" if n_existing > 0 else "Optimal mast"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=9, label="Wind turbine"),
        Line2D([0], [0], color="grey", linewidth=1.2, linestyle="--",
               label="5 km radius"),
    ]
    ax0.legend(handles=ax0.get_legend_handles_labels()[0] + legend_extras,
               loc="upper left", fontsize=7, framealpha=0.85)
    _format_axis(ax0)

    # Panel 2 : Horizontal uncertainty
    ax1 = fig.add_subplot(gs[0, 1])
    _draw_terrain(ax1, xx, yy, elev_grid)
    ax1.set_title("Horizontal Extrapolation Uncertainty",
                  fontsize=10, fontweight="bold")
    vmax_h = np.nanpercentile(h_grid, 98)
    im1 = ax1.pcolormesh(xx, yy, h_grid, cmap="YlOrRd", alpha=0.72,
                          shading="auto", vmin=0, vmax=max(vmax_h, 0.01))
    fig.colorbar(im1, ax=ax1, shrink=0.82, pad=0.02).set_label(
        "Uncertainty (%)", fontsize=8)
    ax1.scatter(wtg_xy[:, 0], wtg_xy[:, 1], c=h_unc, cmap="YlOrRd",
                s=70, marker="^", edgecolors="black", linewidths=0.6,
                zorder=5, vmin=0, vmax=max(h_unc.max(), 0.01))
    for xy, u in zip(wtg_xy, h_unc):
        ax1.annotate(f"{u:.1f}%", xy=xy, xytext=(4, 4),
                     textcoords="offset points", fontsize=6, fontweight="bold")
    _scatter_masts(ax1, meas_locs, n_existing=n_existing, mast_types=mast_types)
    _format_axis(ax1)

    # Panel 3 : Vertical uncertainty
    ax2 = fig.add_subplot(gs[1, 0])
    _draw_terrain(ax2, xx, yy, elev_grid)
    ax2.set_title("Vertical Extrapolation Uncertainty",
                  fontsize=10, fontweight="bold")
    vmax_v = np.nanpercentile(v_grid, 98)
    im2 = ax2.pcolormesh(xx, yy, v_grid, cmap="PuBuGn", alpha=0.72,
                          shading="auto", vmin=0, vmax=max(vmax_v, 0.01))
    fig.colorbar(im2, ax=ax2, shrink=0.82, pad=0.02).set_label(
        "Uncertainty (%)", fontsize=8)
    ax2.scatter(wtg_xy[:, 0], wtg_xy[:, 1], c=v_unc, cmap="PuBuGn",
                s=70, marker="^", edgecolors="black", linewidths=0.6,
                zorder=5, vmin=0, vmax=max(v_unc.max(), 0.01))
    for xy, u in zip(wtg_xy, v_unc):
        ax2.annotate(f"{u:.1f}%", xy=xy, xytext=(4, 4),
                     textcoords="offset points", fontsize=6, fontweight="bold")
    _scatter_masts(ax2, meas_locs, n_existing=n_existing, mast_types=mast_types)
    _format_axis(ax2)

    # Panel 4 : Combined RSS uncertainty
    ax3 = fig.add_subplot(gs[1, 1])
    _draw_terrain(ax3, xx, yy, elev_grid)
    ax3.set_title("Combined Uncertainty  √(H² + V² + I²)",
                  fontsize=10, fontweight="bold")
    vmax_c = np.nanpercentile(combined_grid, 98)
    im3 = ax3.pcolormesh(xx, yy, combined_grid, cmap="RdPu", alpha=0.72,
                          shading="auto", vmin=0, vmax=max(vmax_c, 0.01))
    fig.colorbar(im3, ax=ax3, shrink=0.82, pad=0.02).set_label(
        "Total Uncertainty (%)", fontsize=8)
    ax3.scatter(wtg_xy[:, 0], wtg_xy[:, 1], c=combined_unc, cmap="RdPu",
                s=70, marker="^", edgecolors="black", linewidths=0.6,
                zorder=5, vmin=0, vmax=max(combined_unc.max(), 0.01))
    for xy, u in zip(wtg_xy, combined_unc):
        ax3.annotate(f"{u:.1f}%", xy=xy, xytext=(4, 4),
                     textcoords="offset points", fontsize=6, fontweight="bold")
    _scatter_masts(ax3, meas_locs, n_existing=n_existing, mast_types=mast_types)
    _format_axis(ax3)

    # ── Summary note ─────────────────────────────────────────────────────────
    total_cost = sum(
        INSTRUMENT_TYPES[t]["cost_aud"]
        for t in (mast_types[n_existing:] if mast_types else [])
    )
    mean_combined = float(combined_unc.mean())
    note = (f"Campaign cost (new instruments):  ${total_cost:,.0f} AUD"
            f"     |     "
            f"Mean wind speed uncertainty:  {mean_combined:.2f} %"
            f"  (mean RSS √(H²+V²+I²) across all WTGs)")
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=10,
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbe6",
                       edgecolor="#c8a800", linewidth=1.2, alpha=0.92))

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip().lstrip("﻿").upper() for c in df.columns]
    df = df.dropna(how="all")   # remove completely empty trailing rows
    return df


def _mast_csv_bytes(analyser):
    rows = []
    total_cost = 0
    for k, (loc, elev) in enumerate(zip(analyser.meas_locs, analyser.meas_elevs)):
        n_wtg  = int((analyser.labels == k).sum())
        itype  = analyser.mast_types[k] if analyser.mast_types else "mast"
        info   = INSTRUMENT_TYPES[itype]
        if k < analyser.n_existing:
            tag, status = f"E{k+1}", "Existing"
            cost = 0
        else:
            tag, status = f"M{k - analyser.n_existing + 1}", "New"
            cost = info["cost_aud"]
        total_cost += cost
        rows.append({
            "Mast":          tag,
            "Status":        status,
            "Instrument":    info["label"],
            "X":             round(loc[0], 1),
            "Y":             round(loc[1], 1),
            "Z_terrain_m":   round(float(elev), 1),
            "WTGs_assigned": n_wtg,
            "Meas_unc_pct":  info["meas_uncertainty_pct"],
            "Cost_AUD":      cost,
        })
    df = pd.DataFrame(rows)
    total_row = {c: "" for c in df.columns}
    total_row["Mast"]     = "TOTAL"
    total_row["Cost_AUD"] = total_cost
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    return df.to_csv(index=False).encode()


def _fig_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────────────────────────────────────────
# SRTM DEM auto-download
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_srtm_elevation(wtg_xy: np.ndarray, epsg_code: int,
                         buffer_m: float = 5000.0,
                         grid_n: int = 35) -> pd.DataFrame:
    """Download SRTM 30 m elevation over the WTG bounding box via OpenTopoData.

    Cached by Streamlit so re-running with the same inputs skips the API calls.
    """
    if not _HAS_DEM:
        raise ImportError(
            "Install 'requests' and 'pyproj':  pip install requests pyproj")

    xmin = wtg_xy[:, 0].min() - buffer_m
    xmax = wtg_xy[:, 0].max() + buffer_m
    ymin = wtg_xy[:, 1].min() - buffer_m
    ymax = wtg_xy[:, 1].max() + buffer_m

    xi = np.linspace(xmin, xmax, grid_n)
    yi = np.linspace(ymin, ymax, grid_n)
    xx, yy = np.meshgrid(xi, yi)
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])

    transformer = _ProjTransformer.from_crs(
        f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(grid_xy[:, 0], grid_xy[:, 1])

    if not (np.all(np.isfinite(lons)) and np.all(np.isfinite(lats))):
        raise ValueError(
            f"Coordinate transformation failed (EPSG:{epsg_code} → WGS84 produced "
            f"non-finite values). Check that the EPSG code matches the coordinate "
            f"system of your WTG file.")

    elevations = []
    batch_size = 100
    n_pts = len(lats)
    for start in range(0, n_pts, batch_size):
        batch_lats = lats[start:start + batch_size]
        batch_lons = lons[start:start + batch_size]
        locations = "|".join(
            f"{lat:.6f},{lon:.6f}" for lat, lon in zip(batch_lats, batch_lons))
        resp = requests.get(
            f"https://api.opentopodata.org/v1/srtm30m?locations={locations}",
            timeout=30)
        resp.raise_for_status()
        for r in resp.json()["results"]:
            elev = r.get("elevation")
            elevations.append(float(elev) if elev is not None else 0.0)
        if start + batch_size < n_pts:
            time.sleep(1.1)

    return pd.DataFrame({
        "X": grid_xy[:, 0],
        "Y": grid_xy[:, 1],
        "Z": np.array(elevations),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit app
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="JK's Wind Measurement Location Optimiser",
                       layout="wide")
    st.title("JK's Wind Measurement Location Optimiser")
    st.caption(
        "Finds optimal anemometer mast positions that minimise horizontal, "
        "vertical and instrument extrapolation uncertainty across a wind farm layout."
    )

    with st.sidebar:
        st.header("Input Files")
        wtg_file  = st.file_uploader("WTG Locations File (CSV)", type=["csv", "txt"])
        xyz_file  = st.file_uploader(
            "XYZ Terrain File (CSV, optional)", type=["csv", "txt"],
            help="Leave blank to auto-download SRTM 30 m DEM using the EPSG code below.")
        meas_file = st.file_uploader(
            "Existing Measurements (CSV, optional)", type=["csv", "txt"],
            help="X/Y columns required; optional Type column (mast/lidar/sodar).")

        st.header("Coordinate System")
        epsg_code = st.number_input(
            "EPSG Code", min_value=1000, max_value=99999, value=7850, step=1,
            help="Used to convert WTG coordinates to lat/lon for SRTM download. "
                 "Default 7850 = GDA2020 Zone 50 (W Australia). "
                 "Ignored when an XYZ terrain file is uploaded.")

        st.header("New Instruments")
        st.caption("Met Masts $400k · 1 % | LiDARs $240k · 2 % | SoDARs $90k · 3 %")
        n_mast  = st.number_input("Met Masts",  min_value=0, max_value=10, value=1, step=1)
        n_lidar = st.number_input("LiDARs",     min_value=0, max_value=10, value=1, step=1)
        n_sodar = st.number_input("SoDARs",     min_value=0, max_value=10, value=1, step=1)
        n_new   = int(n_mast + n_lidar + n_sodar)

        st.header("Parameters")
        diameter_m = st.number_input("Turbine rotor diameter (m)",
                                     min_value=10.0, value=150.0, step=5.0)
        resolution = st.slider("Heat-map grid resolution",
                               min_value=50, max_value=300, value=150, step=10,
                               help="Higher = sharper map, slower to compute")

        st.header("Terrain Classification")
        terrain_mode = st.radio("Method",
                                ["Auto-detect (IEC 61400-1)", "Manual override"],
                                index=0)
        manual_terrain = None
        if terrain_mode == "Manual override":
            manual_terrain = st.selectbox("Terrain type",
                                          ["simple", "moderate", "complex"])

        st.divider()
        files_ready = wtg_file is not None and n_new >= 1
        run_btn = st.button("▶  Run Analysis", type="primary",
                            disabled=not files_ready,
                            use_container_width=True)
        if not files_ready:
            st.caption("Upload WTG locations file and add at least 1 new instrument.")

    if not run_btn:
        st.info(
            "Upload your WTG locations file and configure instruments in the sidebar, "
            "then click **▶ Run Analysis**.  \n"
            "Terrain elevation is sourced from an uploaded XYZ file **or** "
            "auto-downloaded from SRTM 30 m via the EPSG code.")
        return

    with st.spinner("Loading data…"):
        try:
            wtg = _load_csv(wtg_file)
            wtg = wtg.dropna(subset=["X", "Y"])
            xyz = _load_csv(xyz_file) if xyz_file is not None else None
            existing_locs  = None
            existing_types = None
            if meas_file is not None:
                meas_df = _load_csv(meas_file)
                meas_df = meas_df.dropna(subset=["X", "Y"])
                existing_locs = meas_df[["X", "Y"]].values.astype(float)
                if "TYPE" in meas_df.columns:
                    existing_types = (meas_df["TYPE"]
                                      .str.lower().str.strip().tolist())
        except Exception as e:
            st.error(f"Failed to load files: {e}")
            return

    if xyz is None:
        with st.spinner(
            f"Downloading SRTM 30 m DEM from OpenTopoData "
            f"(EPSG:{epsg_code}, 5 km buffer) — ~15 s…"
        ):
            try:
                wtg_xy = wtg[["X", "Y"]].values.astype(float)
                xyz = fetch_srtm_elevation(wtg_xy, int(epsg_code))
                st.success(
                    f"SRTM DEM downloaded: {len(xyz):,} elevation points "
                    f"(EPSG:{epsg_code}).")
            except Exception as e:
                st.error(f"DEM download failed: {e}")
                return

    n_new = max(1, min(n_new, len(wtg)))
    new_instrument_counts = {"mast": int(n_mast), "lidar": int(n_lidar),
                             "sodar": int(n_sodar)}

    with st.spinner("Assessing terrain complexity (IEC 61400-1)…"):
        auto_terrain, slope_stats = assess_terrain_complexity(
            xyz, wtg, assessment_radius_m=5 * diameter_m)

    terrain = manual_terrain if terrain_mode == "Manual override" else auto_terrain

    st.subheader("IEC 61400-1 Terrain Assessment")
    cols = st.columns(6)
    cols[0].metric("Auto-classification", auto_terrain.upper())
    cols[1].metric("Assessment radius", f"{slope_stats['assessment_radius_m']:.0f} m")
    cols[2].metric("Mean slope",   f"{slope_stats['mean_slope_deg']:.1f}°")
    cols[3].metric("Median slope", f"{slope_stats['p50_slope_deg']:.1f}°")
    cols[4].metric("P90 slope",    f"{slope_stats['p90_slope_deg']:.1f}°")
    cols[5].metric("Max slope",    f"{slope_stats['max_slope_deg']:.1f}°")

    if terrain_mode == "Manual override":
        st.info(f"Terrain overridden to **{terrain}** (auto: {auto_terrain})")
    else:
        st.success(f"Auto-detected terrain: **{auto_terrain}**  "
                   f"(simple ≤3°/p90≤2° | moderate ≤10°/p90≤7° | complex >10°)")

    if existing_locs is not None:
        st.info(f"{len(existing_locs)} existing mast(s) loaded — "
                f"solving for {n_new} additional new instrument(s).")

    with st.spinner("Clustering turbines, optimising positions, "
                    "assigning instrument types…"):
        analyser = WindMeasurementAnalyser(
            xyz, wtg, n_new, terrain=terrain,
            existing_locs=existing_locs, existing_types=existing_types,
            new_instrument_counts=new_instrument_counts,
        )
        analyser._slope_stats = slope_stats
        analyser.cluster_turbines()
        analyser.optimise_measurement_locations()

    with st.spinner("Computing uncertainties and building heat-map grids…"):
        h_unc, v_unc, i_unc = analyser.compute_uncertainties()
        xx, yy, h_grid, v_grid, elev_grid, inst_grid = \
            analyser.build_heatmap_grids(resolution)

    with st.spinner("Rendering figure…"):
        fig = build_figure(analyser, h_unc, v_unc, i_unc,
                           xx, yy, h_grid, v_grid, inst_grid, elev_grid)

    # ── Key metrics banner ────────────────────────────────────────────────────
    combined_unc_all = np.sqrt(h_unc ** 2 + v_unc ** 2 + i_unc ** 2)
    total_cost_new   = sum(
        INSTRUMENT_TYPES[t]["cost_aud"]
        for t in (analyser.mast_types[analyser.n_existing:]
                  if analyser.mast_types else [])
    )
    kb1, kb2 = st.columns(2)
    kb1.metric("Campaign cost (new instruments)",
               f"${total_cost_new:,.0f} AUD")
    kb2.metric("Expected mean wind speed uncertainty",
               f"{combined_unc_all.mean():.2f} %",
               help="Mean RSS √(H²+V²+I²) across all turbines")

    st.subheader("Results")
    st.pyplot(fig, use_container_width=True)

    # ── Mast coordinates table ────────────────────────────────────────────────
    st.subheader("Measurement Locations")
    n_e        = analyser.n_existing
    mast_types = analyser.mast_types
    total_cost = 0
    mast_rows  = []
    for k, (loc, elev) in enumerate(zip(analyser.meas_locs, analyser.meas_elevs)):
        itype  = mast_types[k] if mast_types else "mast"
        info   = INSTRUMENT_TYPES[itype]
        if k < n_e:
            tag, status = f"E{k+1}", "Existing"
            cost = 0
        else:
            tag, status = f"M{k - n_e + 1}", "New"
            cost = info["cost_aud"]
        total_cost += cost
        mast_rows.append({
            "Mast":          tag,
            "Status":        status,
            "Instrument":    info["label"],
            "Easting X (m)": round(loc[0], 1),
            "Northing Y (m)":round(loc[1], 1),
            "Elevation (m)": round(float(elev), 1),
            "WTGs assigned": int((analyser.labels == k).sum()),
            "Meas unc (%)":  info["meas_uncertainty_pct"],
            "Cost (AUD)":    f"${cost:,.0f}" if cost > 0 else "—",
        })
    st.dataframe(pd.DataFrame(mast_rows), use_container_width=True, hide_index=True)
    st.metric("Total new instrument cost", f"${total_cost:,.0f} AUD")

    # ── Per-turbine uncertainty breakdown ─────────────────────────────────────
    with st.expander("Per-turbine uncertainty breakdown"):
        wtg_xy       = analyser.wtg[["X", "Y"]].values.astype(float)
        combined_unc = np.sqrt(h_unc ** 2 + v_unc ** 2 + i_unc ** 2)
        wtg_rows = []
        for i in range(len(wtg_xy)):
            k    = analyser.labels[i]
            dist = np.linalg.norm(wtg_xy[i] - analyser.meas_locs[k])
            dz   = abs(analyser.wtg_elevs[i] - analyser.meas_elevs[k])
            tag  = f"E{k+1}" if k < n_e else f"M{k - n_e + 1}"
            wtg_rows.append({
                "WTG":            f"WTG {i+1}",
                "Mast":           tag,
                "H distance (m)": round(dist, 0),
                "H unc (%)":      round(h_unc[i], 2),
                "ΔElev (m)":      round(dz, 1),
                "V unc (%)":      round(v_unc[i], 2),
                "I unc (%)":      round(i_unc[i], 2),
                "RSS unc (%)":    round(combined_unc[i], 2),
            })
        st.dataframe(pd.DataFrame(wtg_rows), use_container_width=True,
                     hide_index=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean H unc",   f"{h_unc.mean():.2f} %",
                  delta=f"max {h_unc.max():.2f} %", delta_color="off")
        c2.metric("Mean V unc",   f"{v_unc.mean():.2f} %",
                  delta=f"max {v_unc.max():.2f} %", delta_color="off")
        c3.metric("Mean I unc",   f"{i_unc.mean():.2f} %",
                  delta=f"max {i_unc.max():.2f} %", delta_color="off")
        c4.metric("Mean RSS unc", f"{combined_unc.mean():.2f} %",
                  delta=f"max {combined_unc.max():.2f} %", delta_color="off")

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.subheader("Downloads")
    dl1, dl2 = st.columns(2)
    dl1.download_button("⬇  Download figure (PNG)",
                        data=_fig_png_bytes(fig),
                        file_name="wind_measurement_results.png",
                        mime="image/png",
                        use_container_width=True)
    dl2.download_button("⬇  Download mast coordinates (CSV)",
                        data=_mast_csv_bytes(analyser),
                        file_name="wind_measurement_locations.csv",
                        mime="text/csv",
                        use_container_width=True)

    plt.close(fig)


if __name__ == "__main__":
    main()

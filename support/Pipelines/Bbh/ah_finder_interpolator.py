"""Interpolate AthenaK Z4c dumps into ADM tensors for an AH finder.

This module is intentionally self-contained and only requires NumPy.  It reads
AthenaK binary output files, interpolates the Z4c variables at Cartesian
coordinates, converts them to ADM quantities, and computes Christoffels/Ricci
from finite differences of the interpolated spatial metric.

Example
-------
    from ah_finder_interpolator import AthenaKAdmInterpolator

    interp = AthenaKAdmInterpolator("MBH_3_2.z4c_3d.00000.bin")
    tensors = interp([[0.0, 0.0, 0.0], [14.6, -3.8, 0.0]])
    print(tensors["SpatialMetric"].shape)  # (2, 3, 3)
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np

DEFAULT_FILENAME = os.environ.get(
    "MBH_Z4C_FILE",
    "/work1/eliasmost/jiaxiwu/MBH_3_2_3d/bin/MBH_3_2.z4c_3d.00000.bin",
)

REQUIRED_Z4C_VARIABLES = (
    "z4c_chi",
    "z4c_gxx",
    "z4c_gxy",
    "z4c_gxz",
    "z4c_gyy",
    "z4c_gyz",
    "z4c_gzz",
    "z4c_Khat",
    "z4c_Axx",
    "z4c_Axy",
    "z4c_Axz",
    "z4c_Ayy",
    "z4c_Ayz",
    "z4c_Azz",
    "z4c_Theta",
)

TENSOR_NAMES = (
    "SpatialMetric",
    "InverseSpatialMetric",
    "ExtrinsicCurvature",
    "SpatialChristoffelSecondKind",
    "SpatialRicci",
)

# Stencil offsets (in units of h) for the 19-point FD stencil used in
# _metric_derivatives.  Order: center, ±x, ±y, ±z, ±x±y, ±x±z, ±y±z.
_STENCIL_OFFSETS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.0, -1.0, 1.0],
        [0.0, -1.0, -1.0],
    ]
)


@dataclass(frozen=True)
class MeshBlock:
    data_offset: int
    shape: tuple[int, int, int]
    logical: tuple[int, int, int, int]
    geometry: tuple[float, float, float, float, float, float]

    @property
    def level(self) -> int:
        return self.logical[3]

    @property
    def dx(self) -> np.ndarray:
        nx3, nx2, nx1 = self.shape
        x0, x1, y0, y1, z0, z1 = self.geometry
        return np.array([(x1 - x0) / nx1, (y1 - y0) / nx2, (z1 - z0) / nx3])

    def contains(self, coord: np.ndarray) -> bool:
        x, y, z = coord
        x0, x1, y0, y1, z0, z1 = self.geometry
        return x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1


def _parse_key_value(line: str) -> tuple[str, str]:
    key, value = line.split("=", 1)
    return key.strip(), value.strip()


def _get_from_header(header: list[str], blockname: str, keyname: str) -> str:
    blockname = blockname.strip()
    keyname = keyname.strip()
    if not blockname.startswith("<"):
        blockname = "<" + blockname
    if not blockname.endswith(">"):
        blockname += ">"

    current_block = "<none>"
    for entry in header:
        if entry.startswith("<"):
            current_block = entry
            continue
        key, value = _parse_key_value(entry)
        if current_block == blockname and key == keyname:
            return value.split("#", 1)[0].strip()
    raise KeyError(f"no parameter called {blockname}/{keyname}")


class AthenaKAdmInterpolator:
    """Callable interpolator returning SpECTRE-style ADM tensor names.

    Parameters
    ----------
    filename:
        AthenaK binary Z4c dump.
    chi_psi_power:
        AthenaK Z4c convention, ``chi = psi**chi_psi_power``.  The MBH dumps
        use the default ``-4``, so ``psi^4 = chi^-1``.
    chi_floor:
        Floor used for non-differentiated chi near punctures.
    fd_points_per_cell:
        Finite-difference offset, in local cell widths, for metric derivatives.
    region:
        Optional ``(xmin, xmax, ymin, ymax, zmin, zmax)``.  Only meshblocks that
        overlap this box are loaded, useful if Nils only needs one horizon.
    """

    def __init__(
        self,
        filename: str = DEFAULT_FILENAME,
        *,
        chi_psi_power: float = -4.0,
        chi_floor: float = 1.0e-5,
        fd_points_per_cell: float = 1.0,
        region: tuple[float, float, float, float, float, float] | None = None,
        verbose: bool = False,
    ) -> None:
        self.filename = filename
        self.chi_psi_power = chi_psi_power
        self.chi_floor = chi_floor
        self.fd_points_per_cell = fd_points_per_cell
        self.verbose = verbose

        metadata = self._read_metadata()
        self.header = metadata["header"]
        self.time = metadata["time"]
        self.cycle = metadata["cycle"]
        self.var_names = metadata["var_names"]
        self.varsizebytes = metadata["varsizebytes"]
        self.locsizebytes = metadata["locsizebytes"]
        self.nvars = metadata["nvars"]
        self.nghost = metadata["nghost"]
        self.root_grid_size = metadata["root_grid_size"]
        self.root_grid_extent = metadata["root_grid_extent"]

        missing = sorted(set(REQUIRED_Z4C_VARIABLES) - set(self.var_names))
        if missing:
            raise ValueError(f"{filename} is missing required variables: {missing}")
        self._var_indices = [self.var_names.index(v) for v in REQUIRED_Z4C_VARIABLES]
        self._var_index = {name: i for i, name in enumerate(REQUIRED_Z4C_VARIABLES)}

        blocks = metadata["blocks"]
        if region is not None:
            blocks = [block for block in blocks if _boxes_overlap(block.geometry, region)]
        if not blocks:
            raise ValueError("no meshblocks selected")
        self.blocks = blocks
        self._load_selected_variables()
        self._build_block_map()

    def __call__(
        self,
        coords: Iterable[Iterable[float]] | np.ndarray,
        tensor_names: Iterable[str] = TENSOR_NAMES,
    ) -> dict[str, np.ndarray]:
        return self.evaluate(coords, tensor_names=tensor_names)

    def evaluate(
        self,
        coords: Iterable[Iterable[float]] | np.ndarray,
        tensor_names: Iterable[str] = TENSOR_NAMES,
    ) -> dict[str, np.ndarray]:
        coords = np.asarray(coords, dtype=float)
        if coords.ndim == 1:
            coords = coords[None, :]
        if coords.shape[1] != 3:
            raise ValueError("coords must have shape (N, 3) or (3,)")

        requested = tuple(tensor_names)
        unknown = sorted(set(requested) - set(TENSOR_NAMES))
        if unknown:
            raise ValueError(f"unknown tensor names: {unknown}")

        out: dict[str, list[np.ndarray]] = {name: [] for name in requested}
        for coord in coords:
            adm = self._adm_at(coord)
            need_derivatives = (
                "SpatialChristoffelSecondKind" in requested or "SpatialRicci" in requested
            )
            if need_derivatives:
                gamma, inv_gamma, dgamma, ddgamma = self._metric_derivatives(coord)
                christoffel = self._christoffel(inv_gamma, dgamma)
                ricci = self._ricci(inv_gamma, dgamma, ddgamma, christoffel)
            for name in requested:
                if name == "SpatialMetric":
                    out[name].append(adm["gamma"])
                elif name == "InverseSpatialMetric":
                    out[name].append(adm["inverse_gamma"])
                elif name == "ExtrinsicCurvature":
                    out[name].append(adm["extrinsic_curvature"])
                elif name == "SpatialChristoffelSecondKind":
                    out[name].append(christoffel)
                elif name == "SpatialRicci":
                    out[name].append(ricci)
        return {name: np.stack(values, axis=0) for name, values in out.items()}

    def _read_metadata(self) -> dict[str, object]:
        blocks: list[MeshBlock] = []
        with open(self.filename, "rb") as fp:
            fp.seek(0, os.SEEK_END)
            filesize = fp.tell()
            fp.seek(0)

            code_header = fp.readline().split()
            if not code_header or code_header[0] != b"Athena":
                raise TypeError(f"{self.filename} is not an Athena binary output")
            version = code_header[-1].split(b"=")[-1]
            if version != b"1.1":
                raise TypeError(f"unsupported Athena binary version {version!r}")

            pheader_count = int(fp.readline().split(b"=")[-1])
            pheader = {}
            for _ in range(pheader_count - 1):
                key, value = _parse_key_value(fp.readline().decode("utf-8").strip())
                pheader[key] = value

            nvars = int(fp.readline().split(b"=")[-1])
            var_names = [v.decode("utf-8") for v in fp.readline().split()[1:]]
            header_size = int(fp.readline().split(b"=")[-1])
            header = [
                line.decode("utf-8").split("#", 1)[0].strip()
                for line in fp.read(header_size).split(b"\n")
            ]
            header = [line for line in header if line]

            locsizebytes = int(pheader["size of location"])
            varsizebytes = int(pheader["size of variable"])
            if locsizebytes not in (4, 8) or varsizebytes not in (4, 8):
                raise ValueError("only 4- or 8-byte AthenaK locations/variables are supported")

            nghost = int(_get_from_header(header, "<mesh>", "nghost"))
            nx1_root = int(_get_from_header(header, "<mesh>", "nx1"))
            nx2_root = int(_get_from_header(header, "<mesh>", "nx2"))
            nx3_root = int(_get_from_header(header, "<mesh>", "nx3"))
            extent = (
                float(_get_from_header(header, "<mesh>", "x1min")),
                float(_get_from_header(header, "<mesh>", "x1max")),
                float(_get_from_header(header, "<mesh>", "x2min")),
                float(_get_from_header(header, "<mesh>", "x2max")),
                float(_get_from_header(header, "<mesh>", "x3min")),
                float(_get_from_header(header, "<mesh>", "x3max")),
            )
            loc_dtype = np.float64 if locsizebytes == 8 else np.float32
            value_dtype = np.float64 if varsizebytes == 8 else np.float32

            while fp.tell() < filesize:
                mb_index = np.frombuffer(fp.read(24), dtype=np.int32).astype(np.int64) - nghost
                nx1 = int(mb_index[1] - mb_index[0] + 1)
                nx2 = int(mb_index[3] - mb_index[2] + 1)
                nx3 = int(mb_index[5] - mb_index[4] + 1)
                logical = tuple(int(x) for x in np.frombuffer(fp.read(16), dtype=np.int32))
                geometry = tuple(float(x) for x in np.frombuffer(fp.read(6 * locsizebytes), dtype=loc_dtype))
                data_offset = fp.tell()
                blocks.append(MeshBlock(data_offset, (nx3, nx2, nx1), logical, geometry))
                fp.seek(nvars * nx1 * nx2 * nx3 * value_dtype().nbytes, os.SEEK_CUR)

        return {
            "time": float(pheader["time"]),
            "cycle": int(pheader["cycle"]),
            "varsizebytes": varsizebytes,
            "locsizebytes": locsizebytes,
            "nvars": nvars,
            "var_names": var_names,
            "header": header,
            "nghost": nghost,
            "root_grid_size": (nx1_root, nx2_root, nx3_root),
            "root_grid_extent": extent,
            "blocks": blocks,
        }

    def _load_selected_variables(self) -> None:
        dtype = np.float64 if self.varsizebytes == 8 else np.float32
        selected = np.empty(
            (len(self.blocks), len(self._var_indices), *self.blocks[0].shape),
            dtype=dtype,
        )
        with open(self.filename, "rb") as fp:
            for block_i, block in enumerate(self.blocks):
                if block.shape != self.blocks[0].shape:
                    raise ValueError("mixed output meshblock shapes are not supported yet")
                if self.verbose and (block_i % 100 == 0 or block_i == len(self.blocks) - 1):
                    print(f"loading meshblock {block_i + 1}/{len(self.blocks)}")
                nx3, nx2, nx1 = block.shape
                fp.seek(block.data_offset)
                data = np.fromfile(fp, dtype=dtype, count=self.nvars * nx1 * nx2 * nx3)
                data = data.reshape(self.nvars, nx3, nx2, nx1)
                selected[block_i] = data[self._var_indices]
        self.data = selected

    def _build_block_map(self) -> None:
        """Build O(1) block lookup keyed by (level, lx1, lx2, lx3)."""
        block_nx3, block_nx2, block_nx1 = self.blocks[0].shape
        nx1_r, nx2_r, nx3_r = self.root_grid_size
        blk_nx0 = nx1_r // block_nx1
        blk_ny0 = nx2_r // block_nx2
        blk_nz0 = nx3_r // block_nx3
        self._block_map: dict[tuple[int, int, int, int], int] = {}
        for i, block in enumerate(self.blocks):
            lx1, lx2, lx3, level = block.logical
            self._block_map[(level, lx1, lx2, lx3)] = i
        max_level = max(b.level for b in self.blocks)
        x0_r, x1_r, y0_r, y1_r, z0_r, z1_r = self.root_grid_extent
        # Precompute per-level (scale_x, scale_y, scale_z, nx, ny, nz) tuples.
        self._level_params = [
            (
                (blk_nx0 << lv) / (x1_r - x0_r),
                (blk_ny0 << lv) / (y1_r - y0_r),
                (blk_nz0 << lv) / (z1_r - z0_r),
                blk_nx0 << lv,
                blk_ny0 << lv,
                blk_nz0 << lv,
            )
            for lv in range(max_level, -1, -1)
        ]
        self._x0_r = x0_r
        self._y0_r = y0_r
        self._z0_r = z0_r
        self._max_level = max_level

    def _find_block_index(self, coord: np.ndarray) -> int:
        x, y, z = float(coord[0]), float(coord[1]), float(coord[2])
        x0_r = self._x0_r
        y0_r = self._y0_r
        z0_r = self._z0_r
        block_map = self._block_map
        for lv, (sx, sy, sz, nx, ny, nz) in enumerate(self._level_params):
            level = self._max_level - lv
            lx1 = math.floor((x - x0_r) * sx)
            lx2 = math.floor((y - y0_r) * sy)
            lx3 = math.floor((z - z0_r) * sz)
            if 0 <= lx1 < nx and 0 <= lx2 < ny and 0 <= lx3 < nz:
                if (idx := block_map.get((level, lx1, lx2, lx3))) is not None:
                    return idx
        raise ValueError(f"coordinate {coord.tolist()} is outside the loaded meshblocks")

    def _z4c_at(self, coord: np.ndarray) -> dict[str, float]:
        block_i = self._find_block_index(coord)
        block = self.blocks[block_i]
        values = self._interp_block_values(block_i, block, coord)
        return {name: float(values[i]) for i, name in enumerate(REQUIRED_Z4C_VARIABLES)}

    def _interp_block_values(
        self, block_i: int, block: MeshBlock, coord: np.ndarray
    ) -> np.ndarray:
        x0, x1, y0, y1, z0, z1 = block.geometry
        nx3, nx2, nx1 = block.shape
        dx = (x1 - x0) / nx1
        dy = (y1 - y0) / nx2
        dz = (z1 - z0) / nx3

        ux = (coord[0] - (x0 + 0.5 * dx)) / dx
        uy = (coord[1] - (y0 + 0.5 * dy)) / dy
        uz = (coord[2] - (z0 + 0.5 * dz)) / dz
        i0 = max(0, min(nx1 - 2, math.floor(ux)))
        j0 = max(0, min(nx2 - 2, math.floor(uy)))
        k0 = max(0, min(nx3 - 2, math.floor(uz)))
        tx = ux - i0
        ty = uy - j0
        tz = uz - k0

        arr = self.data[block_i]
        omtx = 1.0 - tx
        omty = 1.0 - ty
        omtz = 1.0 - tz
        c0 = (arr[:, k0, j0, i0] * omtx + arr[:, k0, j0, i0 + 1] * tx) * omty + (
            arr[:, k0, j0 + 1, i0] * omtx + arr[:, k0, j0 + 1, i0 + 1] * tx
        ) * ty
        c1 = (arr[:, k0 + 1, j0, i0] * omtx + arr[:, k0 + 1, j0, i0 + 1] * tx) * omty + (
            arr[:, k0 + 1, j0 + 1, i0] * omtx + arr[:, k0 + 1, j0 + 1, i0 + 1] * tx
        ) * ty
        return c0 * omtz + c1 * tz

    def _interp_block_values_batch(
        self, block_i: int, block: MeshBlock, coords: np.ndarray
    ) -> np.ndarray:
        """Interpolate at N coordinates within a single block.

        coords: shape (N, 3). Returns shape (N, n_vars).
        Points outside the block are clamped to its boundary.
        """
        x0, x1, y0, y1, z0, z1 = block.geometry
        nx3, nx2, nx1 = block.shape
        dx = (x1 - x0) / nx1
        dy = (y1 - y0) / nx2
        dz = (z1 - z0) / nx3

        ux = (coords[:, 0] - (x0 + 0.5 * dx)) / dx
        uy = (coords[:, 1] - (y0 + 0.5 * dy)) / dy
        uz = (coords[:, 2] - (z0 + 0.5 * dz)) / dz
        i0 = np.clip(np.floor(ux).astype(np.intp), 0, nx1 - 2)
        j0 = np.clip(np.floor(uy).astype(np.intp), 0, nx2 - 2)
        k0 = np.clip(np.floor(uz).astype(np.intp), 0, nx3 - 2)
        tx = (ux - i0)[:, np.newaxis]
        ty = (uy - j0)[:, np.newaxis]
        tz = (uz - k0)[:, np.newaxis]

        arr = self.data[block_i]
        omtx = 1.0 - tx
        omty = 1.0 - ty
        omtz = 1.0 - tz
        # arr[:, k0, j0, i0] has shape (n_vars, N); transpose to (N, n_vars)
        c0 = (arr[:, k0, j0, i0].T * omtx + arr[:, k0, j0, i0 + 1].T * tx) * omty + (
            arr[:, k0, j0 + 1, i0].T * omtx + arr[:, k0, j0 + 1, i0 + 1].T * tx
        ) * ty
        c1 = (arr[:, k0 + 1, j0, i0].T * omtx + arr[:, k0 + 1, j0, i0 + 1].T * tx) * omty + (
            arr[:, k0 + 1, j0 + 1, i0].T * omtx + arr[:, k0 + 1, j0 + 1, i0 + 1].T * tx
        ) * ty
        return c0 * omtz + c1 * tz

    def _adm_at(self, coord: np.ndarray) -> dict[str, np.ndarray]:
        block_i = self._find_block_index(coord)
        return self._adm_from_vals(self._interp_block_values(block_i, self.blocks[block_i], coord))

    def _adm_from_vals(self, v: np.ndarray) -> dict[str, np.ndarray]:
        chi = float(v[0])
        if chi < self.chi_floor:
            chi = self.chi_floor
        psi4 = chi ** (4.0 / self.chi_psi_power)
        gxx = float(v[1]) * psi4
        gxy = float(v[2]) * psi4
        gxz = float(v[3]) * psi4
        gyy = float(v[4]) * psi4
        gyz = float(v[5]) * psi4
        gzz = float(v[6]) * psi4
        gamma = np.array([[gxx, gxy, gxz], [gxy, gyy, gyz], [gxz, gyz, gzz]])
        trace_k = float(v[7]) + 2.0 * float(v[14])
        axx = float(v[8]) * psi4
        axy = float(v[9]) * psi4
        axz = float(v[10]) * psi4
        ayy = float(v[11]) * psi4
        ayz = float(v[12]) * psi4
        azz = float(v[13]) * psi4
        atilde_scaled = np.array([[axx, axy, axz], [axy, ayy, ayz], [axz, ayz, azz]])
        extrinsic_curvature = atilde_scaled + (trace_k / 3.0) * gamma
        inverse_gamma = _inv3x3_sym(gamma)
        return {
            "gamma": gamma,
            "inverse_gamma": inverse_gamma,
            "extrinsic_curvature": extrinsic_curvature,
        }

    def _adm_from_z4c(self, z4c: dict[str, float]) -> dict[str, np.ndarray]:
        chi = max(z4c["z4c_chi"], self.chi_floor)
        psi4 = chi ** (4.0 / self.chi_psi_power)
        gtilde = _sym_matrix(
            z4c["z4c_gxx"],
            z4c["z4c_gxy"],
            z4c["z4c_gxz"],
            z4c["z4c_gyy"],
            z4c["z4c_gyz"],
            z4c["z4c_gzz"],
        )
        atilde = _sym_matrix(
            z4c["z4c_Axx"],
            z4c["z4c_Axy"],
            z4c["z4c_Axz"],
            z4c["z4c_Ayy"],
            z4c["z4c_Ayz"],
            z4c["z4c_Azz"],
        )
        gamma = psi4 * gtilde
        trace_k = z4c["z4c_Khat"] + 2.0 * z4c["z4c_Theta"]
        extrinsic_curvature = psi4 * atilde + (trace_k / 3.0) * gamma
        inverse_gamma = _inv3x3_sym(gamma)
        return {
            "gamma": gamma,
            "inverse_gamma": inverse_gamma,
            "extrinsic_curvature": extrinsic_curvature,
        }

    def _spatial_metric_at(self, coord: np.ndarray) -> np.ndarray:
        block_i = self._find_block_index(coord)
        v = self._interp_block_values(block_i, self.blocks[block_i], coord)
        chi = float(v[0])
        if chi < self.chi_floor:
            chi = self.chi_floor
        psi4 = chi ** (4.0 / self.chi_psi_power)
        return np.array(
            [
                [float(v[1]) * psi4, float(v[2]) * psi4, float(v[3]) * psi4],
                [float(v[2]) * psi4, float(v[4]) * psi4, float(v[5]) * psi4],
                [float(v[3]) * psi4, float(v[5]) * psi4, float(v[6]) * psi4],
            ]
        )

    def _metric_derivatives(
        self, coord: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        block_i = self._find_block_index(coord)
        block = self.blocks[block_i]
        h = self.fd_points_per_cell * block.dx
        hx, hy, hz = float(h[0]), float(h[1]), float(h[2])
        all_pts = coord + _STENCIL_OFFSETS * h
        vals = self._interp_block_values_batch(block_i, block, all_pts)

        # Vectorized gamma computation for all 19 stencil points at once.
        # vals[:, 0]=chi, 1=gxx, 2=gxy, 3=gxz, 4=gyy, 5=gyz, 6=gzz
        psi4 = np.maximum(vals[:, 0], self.chi_floor) ** (4.0 / self.chi_psi_power)
        g6 = vals[:, 1:7] * psi4[:, np.newaxis]  # (19, 6)
        # gammas[k] is the 3×3 spatial metric at stencil point k; shape (19, 3, 3)
        gammas = np.empty((19, 3, 3))
        gammas[:, 0, 0] = g6[:, 0]
        gammas[:, 0, 1] = gammas[:, 1, 0] = g6[:, 1]
        gammas[:, 0, 2] = gammas[:, 2, 0] = g6[:, 2]
        gammas[:, 1, 1] = g6[:, 3]
        gammas[:, 1, 2] = gammas[:, 2, 1] = g6[:, 4]
        gammas[:, 2, 2] = g6[:, 5]

        gamma = gammas[0]
        inverse_gamma = _inv3x3_sym(gamma)
        dgamma = np.empty((3, 3, 3))
        dgamma[0] = (gammas[1] - gammas[2]) / (2.0 * hx)
        dgamma[1] = (gammas[3] - gammas[4]) / (2.0 * hy)
        dgamma[2] = (gammas[5] - gammas[6]) / (2.0 * hz)
        ddgamma = np.empty((3, 3, 3, 3))
        ddgamma[0, 0] = (gammas[1] - 2.0 * gamma + gammas[2]) / (hx * hx)
        ddgamma[1, 1] = (gammas[3] - 2.0 * gamma + gammas[4]) / (hy * hy)
        ddgamma[2, 2] = (gammas[5] - 2.0 * gamma + gammas[6]) / (hz * hz)
        mixed = (gammas[7] - gammas[8] - gammas[9] + gammas[10]) / (4.0 * hx * hy)
        ddgamma[0, 1] = mixed
        ddgamma[1, 0] = mixed
        mixed = (gammas[11] - gammas[12] - gammas[13] + gammas[14]) / (4.0 * hx * hz)
        ddgamma[0, 2] = mixed
        ddgamma[2, 0] = mixed
        mixed = (gammas[15] - gammas[16] - gammas[17] + gammas[18]) / (4.0 * hy * hz)
        ddgamma[1, 2] = mixed
        ddgamma[2, 1] = mixed
        return gamma, inverse_gamma, dgamma, ddgamma

    @staticmethod
    def _christoffel(inverse_gamma: np.ndarray, dgamma: np.ndarray) -> np.ndarray:
        # sym[i,j,l] = dgamma[i,j,l] + dgamma[j,i,l] - dgamma[l,i,j]
        sym = dgamma + dgamma.transpose(1, 0, 2) - dgamma.transpose(1, 2, 0)
        return 0.5 * np.einsum("al,ijl->aij", inverse_gamma, sym)

    @staticmethod
    def _ricci(
        inverse_gamma: np.ndarray,
        dgamma: np.ndarray,
        ddgamma: np.ndarray,
        christoffel: np.ndarray,
    ) -> np.ndarray:
        # d_inverse[m,a,b] = -inv[a,p] * dgamma[m,p,q] * inv[q,b]
        d_inverse = -np.einsum("ap,mpq,qb->mab", inverse_gamma, dgamma, inverse_gamma)

        # sym_dg[i,j,l] = dgamma[i,j,l] + dgamma[j,i,l] - dgamma[l,i,j]
        sym_dg = dgamma + dgamma.transpose(1, 0, 2) - dgamma.transpose(1, 2, 0)

        # sym_ddg[m,i,j,l] = ddgamma[m,i,j,l] + ddgamma[m,j,i,l] - ddgamma[m,l,i,j]
        sym_ddg = ddgamma + ddgamma.transpose(0, 2, 1, 3) - ddgamma.transpose(0, 2, 3, 1)

        d_christoffel = 0.5 * (
            np.einsum("mal,ijl->maij", d_inverse, sym_dg)
            + np.einsum("al,mijl->maij", inverse_gamma, sym_ddg)
        )

        ricci = np.einsum("mmij->ij", d_christoffel)
        ricci -= np.einsum("jkik->ij", d_christoffel)
        ricci += np.einsum("kij,lkl->ij", christoffel, christoffel)
        ricci -= np.einsum("lik,kjl->ij", christoffel, christoffel)
        return 0.5 * (ricci + ricci.T)


def adm_quantities(
    coords: Iterable[Iterable[float]] | np.ndarray,
    tensor_names: Iterable[str] = TENSOR_NAMES,
) -> dict[str, np.ndarray]:
    """Cached convenience function for Nils' AH-finder hook."""

    return _default_interpolator()(coords, tensor_names=tensor_names)


@lru_cache(maxsize=1)
def _default_interpolator() -> AthenaKAdmInterpolator:
    return AthenaKAdmInterpolator(DEFAULT_FILENAME)


def _index_and_fraction(u: float, n: int) -> tuple[int, float]:
    if n < 2:
        raise ValueError("cannot interpolate along an axis with fewer than two cells")
    i0 = int(np.floor(u))
    i0 = max(0, min(n - 2, i0))
    return i0, float(u - i0)


def _sym_matrix(xx: float, xy: float, xz: float, yy: float, yz: float, zz: float) -> np.ndarray:
    return np.array(
        [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
        dtype=float,
    )


def _inv3x3_sym(m: np.ndarray) -> np.ndarray:
    """Analytic inverse of a 3×3 symmetric matrix via cofactor expansion."""
    a, b, c = m[0, 0], m[0, 1], m[0, 2]
    d, e = m[1, 1], m[1, 2]
    f = m[2, 2]
    A = d * f - e * e
    B = c * e - b * f
    C = b * e - c * d
    D = a * f - c * c
    E = b * c - a * e
    F = a * d - b * b
    det = a * A + b * B + c * C
    return np.array([[A, B, C], [B, D, E], [C, E, F]]) / det


def _boxes_overlap(
    a: tuple[float, float, float, float, float, float],
    b: tuple[float, float, float, float, float, float],
) -> bool:
    return not (
        a[1] < b[0]
        or b[1] < a[0]
        or a[3] < b[2]
        or b[3] < a[2]
        or a[5] < b[4]
        or b[5] < a[4]
    )


def _summary(filename: str) -> None:
    reader = AthenaKAdmInterpolator.__new__(AthenaKAdmInterpolator)
    reader.filename = filename
    metadata = reader._read_metadata()
    blocks = metadata["blocks"]
    levels = np.array([block.level for block in blocks])
    print(f"file: {filename}")
    print(f"time: {metadata['time']:g}")
    print(f"cycle: {metadata['cycle']}")
    print(f"variables: {', '.join(metadata['var_names'])}")
    print(f"root grid: {metadata['root_grid_size']}, extent: {metadata['root_grid_extent']}")
    print(f"meshblocks: {len(blocks)}, levels: {levels.min()}..{levels.max()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", nargs="?", default=DEFAULT_FILENAME)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--coord", nargs=3, type=float, action="append", metavar=("X", "Y", "Z"))
    parser.add_argument("--region", nargs=6, type=float, metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.summary and not args.coord:
        _summary(args.filename)
        return

    interp = AthenaKAdmInterpolator(
        args.filename,
        region=tuple(args.region) if args.region else None,
        verbose=args.verbose,
    )
    coords = np.array(args.coord or [[0.0, 0.0, 0.0]], dtype=float)
    tensors = interp(coords)
    print(f"time={interp.time:g} cycle={interp.cycle} coords={coords.tolist()}")
    for name, value in tensors.items():
        print(f"{name}: shape={value.shape}")
        print(value)


if __name__ == "__main__":
    main()

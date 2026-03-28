import numpy as np
from dataclasses import dataclass
from scipy.ndimage import rotate

from dataset.mar.ct_artifact_simulator import (
    BeamHardeningArtifactSimulator,
    CompositeArtifactSimulator,
    MotionArtifactSimulator,
    RingArtifactSimulator,
    ScatterArtifactSimulator,
    TruncationArtifactSimulator,
)
from dataset.mar.physics_params import ATTEN_MODE_COL, PhysicsConfig, PhysicsParams


@dataclass
class MockGeometryConfig:
    image_size: int = 128


class MockGeometry:
    def __init__(self, config: MockGeometryConfig | None = None) -> None:
        self.config = config or MockGeometryConfig()
        self.reso = 0.0369
        self._angles = np.linspace(0.0, 180.0, 90, endpoint=False)

    def forward(self, image: np.ndarray) -> np.ndarray:
        rows = []
        for angle in self._angles:
            rotated = rotate(image, angle=float(angle), reshape=False, order=1, mode="nearest")
            rows.append(rotated.sum(axis=0))
        return np.stack(rows, axis=0).astype(np.float32)

    def fbp(self, sinogram: np.ndarray) -> np.ndarray:
        h = self.config.image_size
        recon = np.zeros((h, h), dtype=np.float32)
        for angle, row in zip(self._angles, sinogram):
            slab = np.tile(row[None, :], (h, 1))
            back = rotate(slab, angle=-float(angle), reshape=False, order=1, mode="nearest")
            recon += back
        recon /= max(len(self._angles), 1)
        return recon.astype(np.float32)


def _make_synthetic_phantom(size: int = 128) -> np.ndarray:
    y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]
    r = np.sqrt(x**2 + y**2)
    phantom = np.full((size, size), -1000.0, dtype=np.float32)
    phantom[r < size * 0.42] = 0.0
    phantom[(r < size * 0.18) & (r > size * 0.11)] = 900.0
    lesion = ((x - 16) ** 2 + (y + 12) ** 2) < (size * 0.05) ** 2
    phantom[lesion] = 60.0
    return phantom


def _make_mock_mu_table(n_energies: int = 120) -> np.ndarray:
    mu = np.zeros((n_energies, 7), dtype=np.float64)
    for col in range(7):
        base = 0.15 + col * 0.01
        mu[:, col] = base * np.exp(-0.01 * np.arange(n_energies))
    mu[69, 6] = 0.192
    return mu


def _make_mock_physics() -> PhysicsParams:
    phy = PhysicsParams.__new__(PhysicsParams)
    phy.config = PhysicsConfig()
    phy.mu_water = _make_mock_mu_table()
    phy.mu_bone = _make_mock_mu_table() * 1.6
    mu_ti = _make_mock_mu_table() * 10.0
    mu_fe = _make_mock_mu_table() * 15.0
    mu_cu = _make_mock_mu_table() * 12.0
    mu_au = _make_mock_mu_table() * 20.0
    phy.mu_metals = np.stack([mu_ti, mu_fe, mu_cu, mu_au], axis=-1)
    phy.spectrum = np.ones(120, dtype=np.float64) * 0.01
    phy.spectrum[60:80] = 0.05
    phy.energies = np.arange(20, 121)
    phy.thresh_water = 100 / 1000 * 0.192 + 0.192
    phy.thresh_bone = 1500 / 1000 * 0.192 + 0.192
    phy.metal_atten = 4.5 * mu_ti[69, ATTEN_MODE_COL]
    phy.para_bhc = np.array([[1.0], [0.001], [-0.0001]], dtype=np.float64)
    return phy


def _make_test_context():
    geo = MockGeometry(MockGeometryConfig(image_size=128))
    phy = _make_mock_physics()
    phantom = _make_synthetic_phantom(128)
    return geo, phy, phantom


def _assert_artifact_result(result):
    assert result.artifact_results
    item = result.artifact_results[0]
    assert item["ma_CT"].shape == result.gt_ct.shape
    assert item["ma_sinogram"].shape == result.poly_sinogram.shape
    assert not np.allclose(item["ma_CT"], result.poly_ct)
    assert item["severity"] in {"mild", "moderate", "severe"} or "+" in item["severity"]


def test_ring_artifact_simulator():
    geo, phy, phantom = _make_test_context()
    sim = RingArtifactSimulator(geo, phy, seed=42)
    result = sim.simulate(phantom, severity="moderate")
    _assert_artifact_result(result)


def test_motion_artifact_simulator():
    geo, phy, phantom = _make_test_context()
    sim = MotionArtifactSimulator(geo, phy, seed=42)
    result = sim.simulate(phantom, severity="moderate")
    _assert_artifact_result(result)


def test_beam_hardening_artifact_simulator():
    geo, phy, phantom = _make_test_context()
    sim = BeamHardeningArtifactSimulator(geo, phy, seed=42)
    result = sim.simulate(phantom, severity="moderate")
    _assert_artifact_result(result)


def test_scatter_artifact_simulator():
    geo, phy, phantom = _make_test_context()
    sim = ScatterArtifactSimulator(geo, phy, seed=42)
    result = sim.simulate(phantom, severity="moderate")
    _assert_artifact_result(result)


def test_truncation_artifact_simulator():
    geo, phy, phantom = _make_test_context()
    sim = TruncationArtifactSimulator(geo, phy, seed=42)
    result = sim.simulate(phantom, severity="moderate")
    _assert_artifact_result(result)


def test_composite_artifact_simulator():
    geo, phy, phantom = _make_test_context()
    ring = RingArtifactSimulator(geo, phy, seed=7)
    scatter = ScatterArtifactSimulator(geo, phy, seed=11)
    combo = CompositeArtifactSimulator(geo, phy, generators=[ring, scatter], seed=123)
    result = combo.simulate_composed(phantom, [(ring, "mild"), (scatter, "moderate")])
    _assert_artifact_result(result)
    recipe = result.artifact_results[0]["params"]["recipe"]
    assert len(recipe) == 2

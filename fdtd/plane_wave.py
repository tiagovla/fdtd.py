"""Incident plane waves in scattered-field and total-field/scattered-field form."""

import numpy as np

from .constants import FREESPACE_PERMEABILITY as MU_0
from .constants import FREESPACE_PERMITTIVITY as EPS_0
from .constants import SPEED_LIGHT
from .materials import Material
from .sources import Source


class PlaneWave:
    """A traveling wave in a homogeneous, isotropic, lossless background.

    Parameters
    ----------
    waveform : callable
        Electric amplitude in V/m as a function of retarded time in seconds.
        Must accept NumPy arrays and return finite scalar or broadcastable values.
    material : Material
        Homogeneous lossless background material carrying the incident wave.
    direction : array_like
        Propagation vector, normalized internally.
    polarization : array_like
        Electric polarization perpendicular to direction, normalized internally.
        Complex vectors can describe elliptical polarization.
    origin : array_like
        Position in meters where the waveform has zero propagation delay.
    """

    def __init__(
        self,
        waveform,
        material: Material,
        *,
        direction=(1, 0, 0),
        polarization=(0, 0, 1),
        origin=(0, 0, 0),
    ):
        if not callable(waveform):
            raise ValueError("waveform must be callable.")
        self.material = material
        self._validate_material()
        direction = self._vector(direction, "direction")
        polarization = self._vector(polarization, "polarization", complex_values=True)
        direction_norm = np.linalg.norm(direction)
        polarization_norm = np.linalg.norm(polarization)
        if not 0 < direction_norm < np.inf or not 0 < polarization_norm < np.inf:
            raise ValueError(
                "direction and polarization must be nonzero finite vectors."
            )
        direction = direction / direction_norm
        polarization = polarization / polarization_norm
        if abs(np.dot(direction, polarization)) > 1e-12:
            raise ValueError("polarization must be perpendicular to direction.")
        self.waveform = waveform
        self.direction = direction
        self.polarization = polarization
        self.origin = self._vector(origin, "origin")

    def _validate_material(self):
        material = self.material
        if not isinstance(material, Material):
            raise ValueError("material must be a Material.")
        Material._validate(
            material.name,
            material.eps_r,
            material.mu_r,
            material.sigma_e,
            material.sigma_m,
        )
        if material.sigma_e != 0 or material.sigma_m != 0:
            raise ValueError("Plane-wave background material must be lossless.")
        if not np.isfinite(self.speed) or not np.isfinite(self.impedance):
            raise ValueError(
                "Background parameters produce non-finite wave properties."
            )

    @property
    def speed(self):
        """Return propagation speed in the background material, in m/s."""
        return SPEED_LIGHT / np.sqrt(self.material.eps_r) / np.sqrt(self.material.mu_r)

    @property
    def impedance(self):
        """Return background wave impedance in ohms."""
        return (
            np.sqrt(MU_0 / EPS_0)
            * np.sqrt(self.material.mu_r)
            / np.sqrt(self.material.eps_r)
        )

    @property
    def magnetic_polarization(self):
        """Return the magnetic polarization per unit electric amplitude."""
        return np.cross(self.direction, self.polarization) / self.impedance

    @staticmethod
    def _vector(value, name, *, complex_values=False):
        if not complex_values and np.iscomplexobj(value):
            raise ValueError(f"{name} must be real.")
        vector = np.asarray(value, dtype=complex if complex_values else float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain three finite components.")
        return vector

    def _values(self, time):
        time = np.asarray(time)
        if np.iscomplexobj(time) or not np.all(np.isfinite(time)):
            raise ValueError("Incident sample times must be finite and real.")
        values = np.asarray(self.waveform(time), dtype=complex)
        try:
            values = np.broadcast_to(values, time.shape)
        except ValueError:
            raise ValueError(
                "waveform values must broadcast to the sample times."
            ) from None
        if not np.all(np.isfinite(values)):
            raise ValueError("waveform must return finite values.")
        return values

    def _field(self, positions, time, polarization):
        if np.iscomplexobj(positions):
            raise ValueError("positions must be real.")
        positions = np.asarray(positions, dtype=float)
        if (
            positions.ndim < 1
            or positions.shape[-1] != 3
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("positions must be a finite array with last dimension 3.")
        delay = np.sum((positions - self.origin) * self.direction, axis=-1) / self.speed
        return self._values(np.asarray(time) - delay)[..., None] * polarization

    def electric_field(self, positions, time):
        """Return incident E in V/m at positions (..., 3) and time in seconds."""
        return self._field(positions, time, self.polarization)

    def magnetic_field(self, positions, time):
        """Return incident H in A/m at positions (..., 3) and time in seconds."""
        return self._field(positions, time, self.magnetic_polarization)


class _PlaneWaveSource(Source):
    def __init__(self, wave, bounds, name):
        if not isinstance(wave, PlaneWave):
            raise ValueError("wave must be a PlaneWave.")
        super().__init__(*bounds, waveform=wave.waveform, name=name)
        self.wave = wave
        self.reset()

    def reset(self):
        self._electric_terms = []
        self._magnetic_terms = []
        self._ready = False

    def _background(self, magnetic):
        properties = self.grid.mu_r if magnetic else self.grid.eps_r
        conductivity = self.grid.sigma_m if magnetic else self.grid.sigma_e
        relative = self.wave.material.mu_r if magnetic else self.wave.material.eps_r
        absolute = relative * (MU_0 if magnetic else EPS_0)
        return np.isclose(properties, relative, rtol=1e-12, atol=0) & (
            conductivity * self.grid.dt <= absolute * np.finfo(float).eps
        )

    def _prepare_wave(self):
        self.reset()
        grid = self.grid
        if grid is None:
            raise RuntimeError("Register the plane-wave source with a grid first.")
        self.wave._validate_material()
        if sum(isinstance(source, _PlaneWaveSource) for source in grid.sources) != 1:
            raise ValueError(
                "Only one incident plane-wave source may define grid fields."
            )
        for axis, size in enumerate(grid.shape):
            if size == 1 and abs(self.wave.direction[axis]) > 1e-12:
                raise ValueError("Plane-wave propagation must use active grid axes.")
            if grid._periodic[axis] and (
                abs(self.wave.direction[axis]) > 1e-12
                or not np.isclose(grid._phases[axis], 1, rtol=0, atol=1e-12)
            ):
                raise ValueError(
                    "Incident waves require zero-phase transverse periodic boundaries."
                )
        courant = (
            self.wave.speed
            * grid.dt
            * np.sqrt(
                sum(1 / d**2 for d, size in zip(grid.spacing, grid.shape) if size > 1)
            )
        )
        if courant >= 1:
            raise ValueError("Time step exceeds the incident background CFL limit.")
        backgrounds = (self._background(False), self._background(True))
        for face, thickness in grid._cpml_faces.items():
            region = [slice(None)] * 3
            region["xyz".index(face[0])] = (
                slice(0, thickness + 1)
                if face[1] == "-"
                else slice(-thickness - 1, None)
            )
            if any(not np.all(background[tuple(region)]) for background in backgrounds):
                raise ValueError(
                    "Incident-wave CPML slabs must contain only background."
                )
        return backgrounds

    def _cache_term(
        self, terms, weights, target, incident, magnetic, *, axis=0, shift=0, lag=0
    ):
        coordinates = np.nonzero(weights)
        if not coordinates[0].size:
            return
        delay = np.zeros(coordinates[0].size)
        for dimension, spacing in enumerate(self.grid.spacing):
            offset = 0.5 * (
                (dimension != incident) if magnetic else dimension == incident
            )
            position = (
                coordinates[dimension] + offset + (shift if dimension == axis else 0)
            ) * spacing
            delay += (
                (position - self.wave.origin[dimension])
                * self.wave.direction[dimension]
                / self.wave.speed
            )
        indices = np.ravel_multi_index(
            (*coordinates, np.full(coordinates[0].size, target)), self.grid.E.shape
        )
        terms.append((indices, weights[coordinates].astype(complex), delay + lag))

    def _apply(self, field, terms):
        if not self._ready:
            raise RuntimeError("Prepare the plane-wave source with Grid.run() first.")
        flat = field.reshape(-1)
        for indices, weights, delay in terms:
            flat[indices] += weights * self.wave._values(self.grid.current_time - delay)

    def update_E(self):
        self._apply(self.grid.E, self._electric_terms)

    def update_H(self):
        self._apply(self.grid.H, self._magnetic_terms)

    def estimate_memory(self):
        """Cache bytes; a conservative upper bound before initial preparation."""
        if self._ready:
            return sum(
                array.nbytes
                for term in (*self._electric_terms, *self._magnetic_terms)
                for array in term
            )
        if self.grid is None:
            return 0
        entry = np.dtype(np.intp).itemsize + np.dtype(complex).itemsize + 8
        return int(self._max_terms * np.prod(self.grid.shape) * entry)


class ScatteredFieldSource(_PlaneWaveSource):
    """Drive material contrast while storing scattered E and H everywhere.

    Add the incident fields from ``wave`` at the corresponding Yee positions
    and sample times to reconstruct total fields. CPML must contain only the
    lossless background. Lumped elements require total fields and are rejected.
    """

    _max_terms = 12

    def __init__(self, wave, *, name=None):
        super().__init__(wave, (0, 0, 0, 0, 0, 0), name)

    def attach_to_grid(self):
        backgrounds = self._prepare_wave()
        if self.grid.lumped_elements:
            raise ValueError("Scattered-field grids cannot contain lumped elements.")
        for magnetic, background in enumerate(backgrounds):
            properties = self.grid.mu_r if magnetic else self.grid.eps_r
            conductivity = self.grid.sigma_m if magnetic else self.grid.sigma_e
            unit = MU_0 if magnetic else EPS_0
            relative = self.wave.material.mu_r if magnetic else self.wave.material.eps_r
            polarization = (
                self.wave.magnetic_polarization if magnetic else self.wave.polarization
            )
            terms = self._magnetic_terms if magnetic else self._electric_terms
            denominator = properties * unit + conductivity * self.grid.dt / 2
            contrast = (relative - properties) * unit
            loss = conductivity * self.grid.dt / 2
            # Subtract the background constitutive update at its two staggered times.
            for lag, numerator in (
                (0, contrast - loss),
                (self.grid.dt, -contrast - loss),
            ):
                for component, amplitude in enumerate(polarization):
                    weights = (
                        np.where(
                            background[..., component],
                            0,
                            numerator[..., component] / denominator[..., component],
                        )
                        * amplitude
                    )
                    self._cache_term(
                        terms, weights, component, component, magnetic, lag=lag
                    )
        self._ready = True


class TFSFSource(_PlaneWaveSource):
    """Inject an incident plane wave on an axis-aligned TF/SF surface.

    Grid fields are total fields inside the half-open box [minimum, maximum)
    and scattered fields outside. Bounds must align with grid nodes and leave
    a background cell between the surface and CPML or exterior grid walls.
    Scatterers and lumped elements must remain at least one cell inside the box.
    Inactive axes use zero bounds. Zero-phase transverse periodic axes must
    span the full stored node range and have no TF/SF faces.

    Incident samples use the analytic background wave. Numerical dispersion
    causes finite-grid leakage; resolve the waveform spatially and check
    convergence. Start with negligible incident fields in the box at time zero.
    """

    _max_terms = 24

    def __init__(self, x_min, y_min, z_min, x_max, y_max, z_max, *, wave, name=None):
        super().__init__(wave, (x_min, y_min, z_min, x_max, y_max, z_max), name)

    def _mask(self, magnetic, margin=0):
        mask = np.ones(self.grid.E.shape, dtype=bool)
        for axis, size in enumerate(self.grid.shape):
            if size == 1 or self.grid._periodic[axis]:
                continue
            for component in range(3):
                offset = 0.5 * ((axis != component) if magnetic else axis == component)
                positions = np.arange(size) + offset
                inside = (positions >= self.idx_s[axis] + margin) & (
                    positions < self.idx_e[axis] - margin
                )
                shape = [1, 1, 1]
                shape[axis] = size
                mask[..., component] &= inside.reshape(shape)
        return mask

    def attach_to_grid(self):
        backgrounds = self._prepare_wave()
        grid = self.grid
        lower = np.array((self.x_min, self.y_min, self.z_min)) / grid.spacing
        upper = np.array((self.x_max, self.y_max, self.z_max)) / grid.spacing
        if not np.allclose(
            lower, np.rint(lower), rtol=0, atol=1e-10
        ) or not np.allclose(upper, np.rint(upper), rtol=0, atol=1e-10):
            raise ValueError("TF/SF bounds must align with grid nodes.")
        self.idx_s, self.idx_e = np.rint(lower).astype(int), np.rint(upper).astype(int)
        for axis, size in enumerate(grid.shape):
            if size == 1:
                continue
            start, stop = self.idx_s[axis], self.idx_e[axis]
            if grid._periodic[axis]:
                if start != 0 or stop != size - 1:
                    raise ValueError("TF/SF boxes must span transverse periodic axes.")
                continue
            first = grid._cpml_faces.get("xyz"[axis] + "-", 0) + 1
            last = size - grid._cpml_faces.get("xyz"[axis] + "+", 0) - 2
            if start < first or stop > last or stop - start < 2:
                raise ValueError(
                    "TF/SF surface must leave a background cell outside CPML/walls."
                )
        for magnetic, background in enumerate(backgrounds):
            if np.any(~background & ~self._mask(magnetic, margin=1)):
                raise ValueError(
                    "Scatterers must lie at least one cell inside the TF/SF surface."
                )
        for element in grid.lumped_elements:
            for axis, bounds in enumerate(
                (
                    (element.x_min, element.x_max),
                    (element.y_min, element.y_max),
                    (element.z_min, element.z_max),
                )
            ):
                if (
                    grid.shape[axis] > 1
                    and not grid._periodic[axis]
                    and not (
                        self.idx_s[axis] + 1
                        <= bounds[0] / grid.spacing[axis]
                        <= bounds[1] / grid.spacing[axis]
                        < self.idx_e[axis] - 1
                    )
                ):
                    raise ValueError(
                        "Lumped elements must lie inside the TF/SF surface."
                    )
        electric_mask, magnetic_mask = self._mask(False), self._mask(True)
        curl_terms = (
            (1, 2, 0, 1),
            (2, 1, 0, -1),
            (2, 0, 1, 1),
            (0, 2, 1, -1),
            (0, 1, 2, 1),
            (1, 0, 2, -1),
        )
        for magnetic in (False, True):
            incident_mask, target_mask = (
                (electric_mask, magnetic_mask)
                if magnetic
                else (magnetic_mask, electric_mask)
            )
            coupling = grid.c_he if magnetic else grid.c_eh
            polarization = (
                self.wave.polarization if magnetic else self.wave.magnetic_polarization
            )
            terms = self._magnetic_terms if magnetic else self._electric_terms
            stencil = ((0, -1), (1, 1)) if magnetic else ((0, 1), (-1, -1))
            for axis, incident, target, sign in curl_terms:
                if grid.shape[axis] == 1 or polarization[incident] == 0:
                    continue
                scale = (
                    sign
                    * coupling[..., target]
                    * polarization[incident]
                    / grid.spacing[axis]
                )
                for shift, factor in stencil:
                    # The mask commutator leaves only derivative stencils crossing a face.
                    difference = target_mask[..., target].astype(np.int8) - np.roll(
                        incident_mask[..., incident], -shift, axis=axis
                    )
                    self._cache_term(
                        terms,
                        scale * factor * difference,
                        target,
                        incident,
                        not magnetic,
                        axis=axis,
                        shift=shift,
                        lag=grid.dt / 2,
                    )
        self._ready = True

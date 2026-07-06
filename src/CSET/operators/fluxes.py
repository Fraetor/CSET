# © Crown copyright, Met Office (2022-2025) and CSET contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Operators to calculate kinematic heat fluxes from covariances."""

import iris
import iris.cube
from cf_units import Unit
from iris.cube import Cube, CubeList

from CSET._common import iter_maybe
from CSET.operators._atmospheric_constants import CPD, LV, RD


def _exactly_one(matches, role):
    if len(matches) == 0:
        raise ValueError(f"sensible_heat_units could not identify a unique {role} cube")
    if len(matches) > 1:
        names = [getattr(c, "var_name", None) or c.name() for c in matches]
        raise ValueError(
            f"sensible_heat_units found multiple possible {role} cubes: {names}"
        )
    return matches[0]


def _is_p_cube(cube):
    return (
        cube.units is not None
        and not cube.units.is_unknown()
        and cube.units.is_convertible(Unit("Pa"))
    )


def _is_T_cube(cube):
    if cube.units is None or cube.units.is_unknown():
        return False
    return cube.units.is_convertible(Unit("K")) or cube.units.is_convertible(
        Unit("degC")
    )


def _is_wt_covar_cube(cube):
    if cube.units is None or cube.units.is_unknown():
        return False
    # turbulence covariance may be recorded either as K m s-1
    # or degC m s-1; these are equivalent
    return cube.units.is_convertible(Unit("K m s-1")) or cube.units.is_convertible(
        Unit("degC m s-1")
    )


def sensible_heat_flux_from_covariance(cubes, **kwargs):
    """
Convert turbulent temperature covariance into sensible heat flux.

This operator computes surface upward sensible heat flux (SHF) from
temperature covariance using:

    SHF = ρ * CPD * (w'T')

where air density is calculated from pressure and temperature via the
ideal gas law.

The required input cubes are identified primarily from their physical
units:

    - temperature covariance (e.g. K m s-1 or degC m s-1)
    - air temperature (convertible to K or degC)
    - air pressure (convertible to Pa)

If multiple physically plausible candidates are found, CF metadata
(e.g. standard names) are used as a secondary disambiguation step.
A ValueError is raised if the required cubes cannot be uniquely
identified.

Parameters
----------
cubes : Cube or CubeList
    Input cube(s) containing exactly one identifiable covariance,
    temperature and pressure cube. Additional cubes are passed through
    unchanged.

**kwargs : dict, optional
    Additional keyword arguments.

Returns
-------
Cube or CubeList
    Input cubes with the pressure, temperature and covariance cubes
    removed and a new
    ``surface_upward_sensible_heat_flux`` cube added. Unrelated cubes
    are passed through unchanged.

Notes
-----
- Pressure is converted internally to Pa and temperature to K.
- Covariance units of ``degC m s-1`` are treated as numerically
  equivalent to ``K m s-1`` because temperature offsets cancel when
  forming fluctuations.
- Input cubes are assumed to be physically compatible; no regridding or
  coordinate alignment is performed.
- Identification is unit-based, with metadata used only to resolve
  ambiguities.

Raises
------
ValueError
    If suitable pressure, temperature or covariance cubes cannot be
    uniquely identified.
"""
    from cf_units import Unit

    cubes = (
        iris.cube.CubeList(cubes)
        if not isinstance(cubes, iris.cube.CubeList)
        else cubes
    )

    # Pressure cube
    p_cand = [c for c in cubes if _is_p_cube(c)]
    if len(p_cand) > 1:
        preferred = [
            c for c in p_cand
            if c.standard_name == "air_pressure"
        ]

        if len(preferred) == 1:
            p_cand = preferred

    pressure = _exactly_one(p_cand, "pressure")

    # Temperature cube
    T_cand = [c for c in cubes if _is_T_cube(c)]
    if len(T_cand) > 1:
        preferred = [
            c for c in T_cand
            if c.standard_name == "air_temperature"
        ]

        if len(preferred) == 1:
            T_cand = preferred

    temp = _exactly_one(T_cand, "temperature")

    # Covariance cube
    covar_cand = [c for c in cubes if _is_wt_covar_cube(c)]
    if len(covar_cand) > 1:
        preferred = []
        for cube in covar_cand:
            text = " ".join(
                str(x).lower()
                for x in (
                    cube.standard_name,
                    cube.var_name,
                    cube.long_name,
                    cube.name(),
                )
                if x
            )
            if "wt" in text or "w't" in text:
                preferred.append(cube)

        if len(preferred) == 1:
            covar_cand = preferred

    wT = _exactly_one(covar_cand, "w'T' covariance")

    #
    # Unit conversions
    #
    temp_K = temp.copy()
    if temp_K.units.is_convertible(Unit("degC")):
        temp_K.convert_units("K")

    pres_Pa = pressure.copy()
    pres_Pa.convert_units("Pa")

    # Treat degC covariance numerically as K covariance
    wT_cov = wT.copy()
    if str(wT_cov.units) == "degC m s-1":
        wT_cov.units = Unit("K m s-1")

    rho_air = pres_Pa.data / (RD * temp_K.data)

    shf = wT_cov.copy()
    shf.data = CPD * rho_air * wT_cov.data
    shf.units = Unit("W m-2")
    shf.rename("surface_upward_sensible_heat_flux")
    shf.var_name = "surface_upward_sensible_heat_flux"

    used_ids = {id(wT), id(temp), id(pressure)}

    out = iris.cube.CubeList(
        c for c in cubes
        if id(c) not in used_ids
    )

    out.append(shf)

    return out[0] if len(out) == 1 else out


def latent_heat_units(
    cubes: Cube | CubeList,
    **kwargs,
) -> Cube | CubeList:
    """
    Convert covariance into latent heat flux units.

    This operator converts any cube with units convertible to kg m-2 s-1
    (i.e. water mass flux) into latent heat flux (W m-2) by multiplying
    by a constant latent heat of vaporisation.

    No attempt is made to distinguish between turbulent fluxes (e.g. w'q')
    and other water mass fluxes. This generalisation seems reasonable
    given that interpreting rainfall or dewfall, for example, as an
    equivalent heat flux is physically meaningful.

    This function operates on one or more Iris cubes. Any cube with
    units convertible to mass flux (kg m-2 s-1) is multiplied by a
    constant latent heat of vaporisation to produce a latent heat flux.
    Cubes with incompatible, missing, or unknown units are passed through
    unchanged.

    Parameters
    ----------
    cubes : Cube or CubeList
        Input cube(s), typically containing w'q' covariance or other flux-like
        quantities.

    **kwargs : dict
        Unused; accepted for interface consistency with other operators.

    Returns
    -------
    Cube or CubeList
        Output cube(s) where:
        - Cubes with units convertible to kg m-2 s-1 are converted to W m-2.
        - All other cubes are returned unchanged.
        - The return type matches the input type (single Cube or CubeList).

    Notes
    -----
    - The conversion uses a fixed latent heat of vaporisation:
          LV = 2.5 × 10^6 J kg-1
    - In reality, Lc varies with temperature (~5% variation between -20 °C
      and +40 °C). This dependency is currently neglected but could be
      included in future improvements.
    - This function does not attempt to identify specific variables; it relies
      solely on unit convertibility to determine applicability.
    """
    REQUIRED_UNITS = Unit("kg m-2 s-1")
    OUTPUT_UNITS = Unit("W m-2")

    out = iris.cube.CubeList()
    for cube in iter_maybe(cubes):
        # ACT ON MASS FLUXES
        if cube.units is None or cube.units.is_unknown():
            out.append(cube)
            continue
        if not cube.units.is_convertible(REQUIRED_UNITS):
            # e.g. if UM LE or some other diagnostic — leave untouched
            out.append(cube)
            continue

        cube_a = cube.copy()
        cube_a = cube_a * LV
        cube_a.units = cube.units * Unit("J kg-1")
        cube_a.convert_units(OUTPUT_UNITS)
        out.append(cube_a)

    return out[0] if len(out) == 1 else out

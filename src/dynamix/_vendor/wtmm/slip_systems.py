"""Crystallographic slip-system database + Nye-tensor → GND density resolver.

Pantleon 2008 framework for decomposing the measured Nye tensor alpha onto
per-slip-system dislocation densities rho_t:

    alpha_ij = sum_t b_i^t l_j^t rho_t,   rho_t >= 0.

With 6 accessible alpha components from 2D EBSD and N candidate slip systems,
this is underdetermined; we pick a physically meaningful minimum.  Two
objective choices:

    solver='nnls'  (default)
        Minimize ||A rho - alpha||_2 subject to rho >= 0
        scipy.optimize.nnls.  What MTEX does by default.

    solver='lp'
        Pantleon's original: minimize sum_t u_t rho_t subject to A rho = alpha,
        rho >= 0.  scipy.optimize.linprog with HiGHS backend.  u_t is the
        line-energy weight per slip-system type (defaults to uniform; pass
        line_energy='anisotropic' for edge-vs-screw weighting).

Slip system tables:
    QUARTZ_SLIP_SYSTEMS         ~96 systems: basal/prism/rhomb <a>, <c>, <a+c>,
                                 + steep trigonal dipyramid {2-1-11} <c+a>
                                 (Lister & Hobbs 1980 Table 1)
    FORSTERITE_SLIP_SYSTEMS     ~18 systems: (010)[100], (001)[100],
                                 (100)[001], (010)[001], {0kl}[100]
                                 pencil glide, secondary [001]
    ORTHOCLASE_SLIP_SYSTEMS     ~12 systems: standard feldspar
    BIOTITE / MUSCOVITE         layered silicate basal + weak cross

Expanded from the minimal table in
shear_zone_wtmm_v3_hobbs_ord.ipynb cell 84 using literature references
listed per-phase below.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Quartz slip systems (alpha-quartz, trigonal D3)
# ---------------------------------------------------------------------------
# References:
#   Blacic & Christie 1984, J. Geophys. Res. 89, 4223
#   Linker & Kirby 1981, J. Geophys. Res. 86, 4659
#   Mainprice & Nicolas 1989, J. Struct. Geol. 11, 175
#   Stipp et al. 2002, Geol. Soc. Spec. Pub. 200, 171 (quartz CIT map)
#   Kronenberg 1994, Rev. Mineral. 29, 123
#   Muto et al. 2011, Tectonophysics 505, 88
#   Lister & Hobbs 1980, J. Struct. Geol. 2, 355 (TBH quartzite fabrics,
#       Table 1: 42-system glide list incl. steep trigonal dipyramid <c+a>;
#       Table 2: CRSS ratios for three model quartzites)
#   Morrison-Smith 1976 (Tectonophysics 33, 43) -- {2-1-11} <c+a> in quartz
#   Blum & Morris 1977 (Phys. Chem. Mineral. 2, 17) -- high-T <c+a> slip
#
# Crystal frame convention for quartz in this module:
#   x -- along a_1 direction in the basal plane
#   y -- perpendicular to a_1 in basal plane (so y = rotate a_1 by +90)
#   z -- along the c-axis
# Burgers vectors and plane normals are given in this orthohexagonal Cartesian.
# EBSD Bunge Euler angles will rotate these into the sample frame per pixel.
#
# Rhomb plane angle from basal:  for quartz c/a ~ 1.10, the (10-11) positive-
# rhomb plane makes ~38 degrees with the basal plane.  We use 38 deg here;
# a more exact treatment would use (0001)*(10-11) = arccos(c / sqrt(a^2 + c^2))
# (c/a convention).  The LP/NNLS is insensitive to small angle errors.

def _quartz_slip_systems():
    cos30, sin30 = math.cos(math.pi / 6), math.sin(math.pi / 6)   # sqrt(3)/2, 1/2
    # Three <a> Burgers vectors in the basal plane at 120 deg
    a_vecs = [
        (1.0, 0.0, 0.0),
        (-sin30, cos30, 0.0),
        (-sin30, -cos30, 0.0),
    ]
    # Three prism {10-10}-type plane normals: same directions as the a-vectors
    prism_normals = a_vecs
    # Positive and negative rhomb plane normals, tilted ~38 deg from basal
    cos_rh = math.cos(math.radians(38.0))
    sin_rh = math.sin(math.radians(38.0))
    pos_rhomb_normals = [
        (cos_rh,            0.0,                sin_rh),
        (-cos_rh * sin30,   cos_rh * cos30,     sin_rh),
        (-cos_rh * sin30,  -cos_rh * cos30,     sin_rh),
    ]
    neg_rhomb_normals = [(x, y, -z) for (x, y, z) in pos_rhomb_normals]
    basal_n = (0.0, 0.0, 1.0)
    c_vec = (0.0, 0.0, 1.0)

    systems = []

    # ---- Basal <a> (3 variants, edge + screw) ----
    for i, a in enumerate(a_vecs):
        for t in ('edge', 'screw'):
            systems.append({'name': f'basal <a>_{i+1} {t}',
                            'b': list(a), 'n': list(basal_n),
                            'type': t, 'family': 'basal_a'})

    # ---- Prism <a> (each prism plane carries the 2 in-plane <a>'s) ----
    for j, pn in enumerate(prism_normals):
        for i, a in enumerate(a_vecs):
            if i == j:
                continue   # this <a> is perpendicular to the plane; not in-plane
            for t in ('edge', 'screw'):
                systems.append({'name': f'prism{j+1} <a>_{i+1} {t}',
                                'b': list(a), 'n': list(pn),
                                'type': t, 'family': 'prism_a'})

    # ---- Prism <c> (three prism planes, Burgers along c) ----
    for j, pn in enumerate(prism_normals):
        for t in ('edge', 'screw'):
            systems.append({'name': f'prism{j+1} <c> {t}',
                            'b': list(c_vec), 'n': list(pn),
                            'type': t, 'family': 'prism_c'})

    # ---- Positive rhomb <a> ----
    for k, rn in enumerate(pos_rhomb_normals):
        for i, a in enumerate(a_vecs):
            if i == k:
                continue
            for t in ('edge', 'screw'):
                systems.append({'name': f'posRhomb{k+1} <a>_{i+1} {t}',
                                'b': list(a), 'n': list(rn),
                                'type': t, 'family': 'rhomb_pos_a'})

    # ---- Negative rhomb <a> ----
    for k, rn in enumerate(neg_rhomb_normals):
        for i, a in enumerate(a_vecs):
            if i == k:
                continue
            for t in ('edge', 'screw'):
                systems.append({'name': f'negRhomb{k+1} <a>_{i+1} {t}',
                                'b': list(a), 'n': list(rn),
                                'type': t, 'family': 'rhomb_neg_a'})

    # ---- Positive rhomb <a+c> ----
    for k, rn in enumerate(pos_rhomb_normals):
        for i, a in enumerate(a_vecs):
            if i == k:
                continue
            b = (a[0] + c_vec[0], a[1] + c_vec[1], a[2] + c_vec[2])
            for t in ('edge', 'screw'):
                systems.append({'name': f'posRhomb{k+1} <a+c>_{i+1} {t}',
                                'b': list(b), 'n': list(rn),
                                'type': t, 'family': 'rhomb_pos_ac'})

    # ---- Negative rhomb <a+c> ----
    for k, rn in enumerate(neg_rhomb_normals):
        for i, a in enumerate(a_vecs):
            if i == k:
                continue
            b = (a[0] + c_vec[0], a[1] + c_vec[1], a[2] + c_vec[2])
            for t in ('edge', 'screw'):
                systems.append({'name': f'negRhomb{k+1} <a+c>_{i+1} {t}',
                                'b': list(b), 'n': list(rn),
                                'type': t, 'family': 'rhomb_neg_ac'})

    # ---- Steep trigonal dipyramid {2-1-11} <c+a> (Lister & Hobbs 1980) ----
    # Plane-to-basal angle ~66 deg (Morrison-Smith 1976); same 3-fold pos/neg
    # symmetry as the rhomb set but with a steeper tilt.  Each of the 3 pos and
    # 3 neg planes hosts 2 non-perpendicular <c+a> Burgers (same i != k rule).
    # L&H split these into <c+a2> and <c+a3> families by Burgers orientation.
    cos_sd = math.cos(math.radians(66.0))
    sin_sd = math.sin(math.radians(66.0))
    pos_steep_normals = [
        (cos_sd,            0.0,                sin_sd),
        (-cos_sd * sin30,   cos_sd * cos30,     sin_sd),
        (-cos_sd * sin30,  -cos_sd * cos30,     sin_sd),
    ]
    neg_steep_normals = [(x, y, -z) for (x, y, z) in pos_steep_normals]

    # For each plane k we get 2 allowed in-plane a's (i != k).  Use the *first*
    # (lower i) for the 'ac2' family and the *second* for 'ac3' -- matches
    # L&H's pairing of `<c+a2>` with one steep-plane orientation and `<c+a3>`
    # with the other.
    def _add_steep(sign_tag, normals):
        for k, rn in enumerate(normals):
            allowed = [i for i in range(3) if i != k]
            for slot, i in enumerate(allowed):
                a = a_vecs[i]
                b = (a[0] + c_vec[0], a[1] + c_vec[1], a[2] + c_vec[2])
                fam_tag = 'ac2' if slot == 0 else 'ac3'
                for t in ('edge', 'screw'):
                    systems.append({
                        'name': f'steepDipyr{sign_tag}{k+1} <c+{fam_tag}>_{i+1} {t}',
                        'b': list(b), 'n': list(rn),
                        'type': t,
                        'family': f'steep_dipyr_{sign_tag}_{fam_tag}',
                    })
    _add_steep('pos', pos_steep_normals)
    _add_steep('neg', neg_steep_normals)

    return systems


QUARTZ_SLIP_SYSTEMS = _quartz_slip_systems()


# ---------------------------------------------------------------------------
# Forsterite / Olivine slip systems (orthorhombic, Pbnm)
# ---------------------------------------------------------------------------
# References:
#   Carter & Ave'Lallemant 1970 (olivine dislocations)
#   Bai et al. 1991 (high-T creep systems)
#   Couvy et al. 2004 (high-pressure [001] slip)
#   Raterron et al. 2004, 2007 (slip-system transitions with P, T)
#   Mainprice et al. 2005 (olivine LPO review)
#   Karato & Wu 1993 (mantle deformation)
# Crystal frame (Pbnm convention):
#   x = a (short axis, ~4.75 A)
#   y = b (long axis, ~10.2 A)
#   z = c (~5.98 A)

_OLIVINE_ROOT = [
    # Primary (010)[100] "a-slip" -- dominant at low-P high-T dry conditions
    {'name': '(010)[100] edge',  'b': [1, 0, 0], 'n': [0, 1, 0], 'type': 'edge',  'family': 'oliv_a'},
    {'name': '(010)[100] screw', 'b': [1, 0, 0], 'n': [0, 1, 0], 'type': 'screw', 'family': 'oliv_a'},
    # (001)[100] second-order a-slip
    {'name': '(001)[100] edge',  'b': [1, 0, 0], 'n': [0, 0, 1], 'type': 'edge',  'family': 'oliv_a'},
    {'name': '(001)[100] screw', 'b': [1, 0, 0], 'n': [0, 0, 1], 'type': 'screw', 'family': 'oliv_a'},
    # (010)[001] c-slip -- wet / high-P high-stress deformation
    {'name': '(010)[001] edge',  'b': [0, 0, 1], 'n': [0, 1, 0], 'type': 'edge',  'family': 'oliv_c'},
    {'name': '(010)[001] screw', 'b': [0, 0, 1], 'n': [0, 1, 0], 'type': 'screw', 'family': 'oliv_c'},
    # (100)[001] c-slip
    {'name': '(100)[001] edge',  'b': [0, 0, 1], 'n': [1, 0, 0], 'type': 'edge',  'family': 'oliv_c'},
    {'name': '(100)[001] screw', 'b': [0, 0, 1], 'n': [1, 0, 0], 'type': 'screw', 'family': 'oliv_c'},
    # (001)[010] b-slip (weak)
    {'name': '(001)[010] edge',  'b': [0, 1, 0], 'n': [0, 0, 1], 'type': 'edge',  'family': 'oliv_b'},
    {'name': '(001)[010] screw', 'b': [0, 1, 0], 'n': [0, 0, 1], 'type': 'screw', 'family': 'oliv_b'},
    # Pencil glide {0kl}[100] -- multiple {0kl} planes with a-Burgers
    {'name': '(011)[100] edge',  'b': [1, 0, 0], 'n': [0, 1, 1], 'type': 'edge',  'family': 'oliv_pencil'},
    {'name': '(011)[100] screw', 'b': [1, 0, 0], 'n': [0, 1, 1], 'type': 'screw', 'family': 'oliv_pencil'},
    {'name': '(021)[100] edge',  'b': [1, 0, 0], 'n': [0, 2, 1], 'type': 'edge',  'family': 'oliv_pencil'},
    {'name': '(021)[100] screw', 'b': [1, 0, 0], 'n': [0, 2, 1], 'type': 'screw', 'family': 'oliv_pencil'},
]


FORSTERITE_SLIP_SYSTEMS = _OLIVINE_ROOT


# ---------------------------------------------------------------------------
# Orthoclase / feldspar slip systems
# ---------------------------------------------------------------------------
# References:
#   Gandais & Willaime 1984, Feldspars and Feldspathoids (NATO ASI C137)
#   Stunitz et al. 2003, Tectonophysics 372, 215
#   Menegon et al. 2006, Solid Earth 7, 285
# Frame: orthogonal approximation (feldspars are actually monoclinic or
# triclinic; this is a working-convention simplification).

ORTHOCLASE_SLIP_SYSTEMS = [
    {'name': '(010)[100] edge',  'b': [1, 0, 0], 'n': [0, 1, 0], 'type': 'edge',  'family': 'fsp_a010'},
    {'name': '(010)[100] screw', 'b': [1, 0, 0], 'n': [0, 1, 0], 'type': 'screw', 'family': 'fsp_a010'},
    {'name': '(010)[001] edge',  'b': [0, 0, 1], 'n': [0, 1, 0], 'type': 'edge',  'family': 'fsp_c010'},
    {'name': '(010)[001] screw', 'b': [0, 0, 1], 'n': [0, 1, 0], 'type': 'screw', 'family': 'fsp_c010'},
    {'name': '(001)[100] edge',  'b': [1, 0, 0], 'n': [0, 0, 1], 'type': 'edge',  'family': 'fsp_a001'},
    {'name': '(001)[100] screw', 'b': [1, 0, 0], 'n': [0, 0, 1], 'type': 'screw', 'family': 'fsp_a001'},
    {'name': '(001)[010] edge',  'b': [0, 1, 0], 'n': [0, 0, 1], 'type': 'edge',  'family': 'fsp_b001'},
    {'name': '(001)[010] screw', 'b': [0, 1, 0], 'n': [0, 0, 1], 'type': 'screw', 'family': 'fsp_b001'},
    # Albite / cleavage-plane slip for plagioclase end-member
    {'name': '(100)[001] edge',  'b': [0, 0, 1], 'n': [1, 0, 0], 'type': 'edge',  'family': 'fsp_pla'},
    {'name': '(100)[001] screw', 'b': [0, 0, 1], 'n': [1, 0, 0], 'type': 'screw', 'family': 'fsp_pla'},
]


# ---------------------------------------------------------------------------
# Biotite / Muscovite / phyllosilicates -- basal cleavage-dominated
# ---------------------------------------------------------------------------
# References: Kronenberg et al. 1990; Mares & Kronenberg 1993.

_MICA_ROOT = [
    {'name': '(001)[100] edge',  'b': [1, 0, 0], 'n': [0, 0, 1], 'type': 'edge',  'family': 'mica_basal'},
    {'name': '(001)[100] screw', 'b': [1, 0, 0], 'n': [0, 0, 1], 'type': 'screw', 'family': 'mica_basal'},
    {'name': '(001)[010] edge',  'b': [0, 1, 0], 'n': [0, 0, 1], 'type': 'edge',  'family': 'mica_basal'},
    {'name': '(001)[010] screw', 'b': [0, 1, 0], 'n': [0, 0, 1], 'type': 'screw', 'family': 'mica_basal'},
    {'name': '(001)<110> edge',  'b': [1, 1, 0], 'n': [0, 0, 1], 'type': 'edge',  'family': 'mica_basal'},
    {'name': '(001)<110> screw', 'b': [1, 1, 0], 'n': [0, 0, 1], 'type': 'screw', 'family': 'mica_basal'},
    # Weak cross slip
    {'name': '(hk0)[001] edge',  'b': [0, 0, 1], 'n': [1, 0, 0], 'type': 'edge',  'family': 'mica_cross'},
]

BIOTITE_SLIP_SYSTEMS = _MICA_ROOT
MUSCOVITE_SLIP_SYSTEMS = _MICA_ROOT


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

SLIP_SYSTEMS_DB = {
    'Quartz-new':     QUARTZ_SLIP_SYSTEMS,
    'Quartz':         QUARTZ_SLIP_SYSTEMS,
    'Quartz Alpha':   QUARTZ_SLIP_SYSTEMS,
    'Quartz low':     QUARTZ_SLIP_SYSTEMS,
    'Forsterite':     FORSTERITE_SLIP_SYSTEMS,
    'Olivine':        FORSTERITE_SLIP_SYSTEMS,
    'Enstatite':      FORSTERITE_SLIP_SYSTEMS,
    'Orthopyroxene':  FORSTERITE_SLIP_SYSTEMS,
    'Orthoclase':     ORTHOCLASE_SLIP_SYSTEMS,
    'Sanidine':       ORTHOCLASE_SLIP_SYSTEMS,
    'K-Feldspar':     ORTHOCLASE_SLIP_SYSTEMS,
    'Microcline':     ORTHOCLASE_SLIP_SYSTEMS,
    'Plagioclase':    ORTHOCLASE_SLIP_SYSTEMS,
    'Albite':         ORTHOCLASE_SLIP_SYSTEMS,
    'Anorthite':      ORTHOCLASE_SLIP_SYSTEMS,
    'Biotite':        BIOTITE_SLIP_SYSTEMS,
    'Phlogopite':     BIOTITE_SLIP_SYSTEMS,
    'Chlorite':       BIOTITE_SLIP_SYSTEMS,
    'Talc':           BIOTITE_SLIP_SYSTEMS,
    'Muscovite':      MUSCOVITE_SLIP_SYSTEMS,
    'Sericite':       MUSCOVITE_SLIP_SYSTEMS,
    'Paragonite':     MUSCOVITE_SLIP_SYSTEMS,
    'Illite':         MUSCOVITE_SLIP_SYSTEMS,
}


# ---------------------------------------------------------------------------
# Burgers vector magnitudes (metres) by phase + family.
# Used to convert Pantleon-NNLS rho output from [rad / pixel] to [m^-2]:
#   rho_si = rho_raw / (px_size_m * |b|_m)
# Phase keys match SLIP_SYSTEMS_DB; family keys match the 'family' field on
# each slip-system entry. Post-Dauphine merge in cell 24 collapses
# {pos,neg} pairs into total families (rhomb_a, rhomb_ac, steep_dipyr_*),
# so those are listed alongside their split versions for convenience.
# Sources: quartz a/c from Heard 1972 / Hobbs & Ord Table 13.1; olivine
# Pbnm cell from Brodholt & Refson 2000; albite from Smith 1974.
# ---------------------------------------------------------------------------

# Quartz alpha (P3221): a0 = 4.913 A, c0 = 5.405 A
_QUARTZ_A = 4.913e-10
_QUARTZ_C = 5.405e-10
_QUARTZ_AC = (_QUARTZ_A**2 + _QUARTZ_C**2) ** 0.5  # ~ 7.30e-10 m

_QUARTZ_BMAG = {
    'basal_a':           _QUARTZ_A,
    'prism_a':           _QUARTZ_A,
    'prism_c':           _QUARTZ_C,
    'rhomb_pos_a':       _QUARTZ_A,
    'rhomb_neg_a':       _QUARTZ_A,
    'rhomb_a':           _QUARTZ_A,    # post-Dauphine merge
    'rhomb_pos_ac':      _QUARTZ_AC,
    'rhomb_neg_ac':      _QUARTZ_AC,
    'rhomb_ac':          _QUARTZ_AC,   # post-Dauphine merge
    # steep dipyramid <a+c> family: same Burgers magnitude as rhomb <a+c>
    'steep_dipyr_pos_ac1': _QUARTZ_AC,
    'steep_dipyr_pos_ac2': _QUARTZ_AC,
    'steep_dipyr_pos_ac3': _QUARTZ_AC,
    'steep_dipyr_neg_ac1': _QUARTZ_AC,
    'steep_dipyr_neg_ac2': _QUARTZ_AC,
    'steep_dipyr_neg_ac3': _QUARTZ_AC,
    'steep_dipyr_ac1':     _QUARTZ_AC,
    'steep_dipyr_ac2':     _QUARTZ_AC,
    'steep_dipyr_ac3':     _QUARTZ_AC,
}

# Forsterite/olivine (Pbnm): a0 = 4.76 A, b0 = 10.21 A, c0 = 5.99 A
_OLIVINE_A = 4.76e-10
_OLIVINE_B = 10.21e-10
_OLIVINE_C = 5.99e-10
_OLIVINE_BMAG = {
    'oliv_a':       _OLIVINE_A,   # Burgers along [100]
    'oliv_b':       _OLIVINE_B,   # Burgers along [010]
    'oliv_c':       _OLIVINE_C,   # Burgers along [001]
    'oliv_pencil':  _OLIVINE_A,   # pencil-glide of [100]
}

# Albite/orthoclase (triclinic/monoclinic): albite cell a=8.18 A, b=12.87 A,
# c=7.11 A.  Family naming in this module: fsp_a010, fsp_c010, fsp_a001, ...
# Burgers direction is the first letter of the family suffix; plane suffix
# does not change |b|.
_ALBITE_A = 8.18e-10
_ALBITE_B = 12.87e-10
_ALBITE_C = 7.11e-10
_FELDSPAR_BMAG = {
    'fsp_a010': _ALBITE_A,
    'fsp_a001': _ALBITE_A,
    'fsp_b001': _ALBITE_B,
    'fsp_c010': _ALBITE_C,
    'fsp_pla':  _ALBITE_A,   # pencil-like; assume <a>
}

# Mica (biotite/muscovite, monoclinic): a0 ~ 5.30 A (dominant in-plane Burgers)
_MICA_A = 5.30e-10
_MICA_BMAG = {
    'mica_basal': _MICA_A,
    'mica_cross': _MICA_A,
}

BURGERS_MAGNITUDE_M = {
    'Quartz-new':     _QUARTZ_BMAG,
    'Quartz':         _QUARTZ_BMAG,
    'Quartz Alpha':   _QUARTZ_BMAG,
    'Quartz low':     _QUARTZ_BMAG,
    'Forsterite':     _OLIVINE_BMAG,
    'Olivine':        _OLIVINE_BMAG,
    'Enstatite':      _OLIVINE_BMAG,
    'Orthopyroxene':  _OLIVINE_BMAG,
    'Orthoclase':     _FELDSPAR_BMAG,
    'Sanidine':       _FELDSPAR_BMAG,
    'K-Feldspar':     _FELDSPAR_BMAG,
    'Microcline':     _FELDSPAR_BMAG,
    'Plagioclase':    _FELDSPAR_BMAG,
    'Albite':         _FELDSPAR_BMAG,
    'Anorthite':      _FELDSPAR_BMAG,
    'Biotite':        _MICA_BMAG,
    'Phlogopite':     _MICA_BMAG,
    'Chlorite':       _MICA_BMAG,
    'Talc':           _MICA_BMAG,
    'Muscovite':      _MICA_BMAG,
    'Sericite':       _MICA_BMAG,
    'Paragonite':     _MICA_BMAG,
    'Illite':         _MICA_BMAG,
}


# ---------------------------------------------------------------------------
# Euler -> rotation, line-direction, geometry matrix
# ---------------------------------------------------------------------------

def euler_to_rotmat(e1_deg, e2_deg, e3_deg):
    """Bunge ZXZ Euler angles (deg) -> 3x3 rotation matrix mapping crystal to sample."""
    p1, P, p2 = np.deg2rad(e1_deg), np.deg2rad(e2_deg), np.deg2rad(e3_deg)
    c1, s1 = np.cos(p1), np.sin(p1)
    cP, sP = np.cos(P), np.sin(P)
    c2, s2 = np.cos(p2), np.sin(p2)
    return np.array([
        [c1 * c2 - s1 * s2 * cP, -c1 * s2 - s1 * c2 * cP,  s1 * sP],
        [s1 * c2 + c1 * s2 * cP, -s1 * s2 + c1 * c2 * cP, -c1 * sP],
        [s2 * sP,                 c2 * sP,                 cP      ],
    ])


def euler_to_rotmat_batch(e1_deg, e2_deg, e3_deg):
    """Vectorized Bunge ZXZ Euler -> rotation matrices.

    Accepts array-like Euler angles of any shape; returns an array of shape
    ``(..., 3, 3)`` whose leading axes match the inputs.  ~100x faster than
    calling :func:`euler_to_rotmat` in a Python loop.
    """
    p1 = np.deg2rad(np.asarray(e1_deg, dtype=np.float64))
    P  = np.deg2rad(np.asarray(e2_deg, dtype=np.float64))
    p2 = np.deg2rad(np.asarray(e3_deg, dtype=np.float64))
    c1, s1 = np.cos(p1), np.sin(p1)
    cP, sP = np.cos(P),  np.sin(P)
    c2, s2 = np.cos(p2), np.sin(p2)
    R = np.empty(p1.shape + (3, 3), dtype=np.float64)
    R[..., 0, 0] = c1 * c2 - s1 * s2 * cP
    R[..., 0, 1] = -c1 * s2 - s1 * c2 * cP
    R[..., 0, 2] = s1 * sP
    R[..., 1, 0] = s1 * c2 + c1 * s2 * cP
    R[..., 1, 1] = -s1 * s2 + c1 * c2 * cP
    R[..., 1, 2] = -c1 * sP
    R[..., 2, 0] = s2 * sP
    R[..., 2, 1] = c2 * sP
    R[..., 2, 2] = cP
    return R


def line_direction(b, n, kind):
    """Return the line direction for an edge or screw dislocation."""
    b_arr = np.asarray(b, dtype=np.float64)
    n_arr = np.asarray(n, dtype=np.float64)
    nb = np.linalg.norm(b_arr)
    nn = np.linalg.norm(n_arr)
    if nb < 1e-30 or nn < 1e-30:
        raise ValueError("b or n has zero norm")
    b_hat = b_arr / nb
    n_hat = n_arr / nn
    if kind == 'edge':
        l = np.cross(b_hat, n_hat)
        ln = np.linalg.norm(l)
        if ln < 1e-12:
            raise ValueError(f"b and n are parallel; no edge line direction for {b}, {n}")
        return l / ln
    if kind == 'screw':
        return b_hat.copy()
    raise ValueError(f"unknown dislocation kind: {kind}")


def build_geometry_matrix(slip_systems):
    """Return A (6 x N) such that alpha_vec = A @ rho, where alpha_vec is the
    flattened 2D-accessible part [alpha_11, alpha_12, alpha_13,
    alpha_21, alpha_22, alpha_23] of the Nye tensor in the crystal frame.
    alpha_ij = sum_t b_i^t l_j^t rho_t (Nye 1953, Eq. 17 via Pantleon 2008).
    """
    n = len(slip_systems)
    A = np.zeros((6, n), dtype=np.float64)
    for s, ss in enumerate(slip_systems):
        b_hat = np.asarray(ss['b'], dtype=np.float64)
        b_hat = b_hat / max(np.linalg.norm(b_hat), 1e-30)
        l_hat = line_direction(ss['b'], ss['n'], ss['type'])
        bl = np.outer(b_hat, l_hat)   # 3x3
        A[:, s] = bl[:2, :].ravel()   # take rows 0,1 (2D-accessible), flatten
    return A


# ---------------------------------------------------------------------------
# Line-energy weights for the LP objective
# ---------------------------------------------------------------------------

_CRSS_LH1980 = {
    # Critical resolved shear stress ratios, Lister & Hobbs 1980 Table 2
    # (medium-T "Model B" quartzite column; approximate).  Lower = easier slip.
    # Used as per-system LP cost weights under mode='crss_lh1980'.
    'basal_a':              1.0,
    'prism_a':              2.0,
    'prism_c':              5.0,   # rarely glidable; placeholder between a and c+a
    'rhomb_pos_a':          3.0,
    'rhomb_neg_a':          4.0,
    'rhomb_pos_ac':         6.0,
    'rhomb_neg_ac':         7.0,
    'steep_dipyr_pos_ac2':  10.0,
    'steep_dipyr_pos_ac3':  10.0,
    'steep_dipyr_neg_ac2':  10.0,
    'steep_dipyr_neg_ac3':  10.0,
}


def line_energy_weights(slip_systems, mode='uniform', poisson_ratio=0.25):
    """Return a (N,) vector of per-system line-energy weights for the LP cost.

    mode :
        'uniform'       all weights = 1.
        'character'     u_edge / u_screw = 1 / (1 - nu).  Default edge-heavier.
        'anisotropic'   character + slight family-level multiplier (basal ~0.7,
                        prism ~1.0, rhomb ~1.3, pencil ~1.2) -- approximate,
                        reflects relative line energies in the respective
                        slip planes.
        'crss_lh1980'   Lister & Hobbs 1980 Table 2 CRSS ratios for the
                        medium-T quartzite model (basal <a> = 1, prism <a> = 2,
                        rhomb <a> ~3-4, rhomb <c+a> ~6-7, steep dipyr <c+a>
                        ~10).  Biases LP/NNLS solutions toward low-CRSS systems.
                        Quartz-specific; for non-quartz phases reduces to the
                        'uniform' vector.
    """
    n = len(slip_systems)
    u = np.ones(n, dtype=np.float64)
    if mode == 'uniform':
        return u
    # character: edges cost more than screws
    edge_screw_ratio = 1.0 / max(1.0 - poisson_ratio, 0.05)
    for i, ss in enumerate(slip_systems):
        u[i] = edge_screw_ratio if ss['type'] == 'edge' else 1.0
    if mode == 'character':
        return u
    if mode == 'anisotropic':
        family_mult = {
            'basal_a':   0.7,
            'prism_a':   1.0,
            'prism_c':   1.2,
            'rhomb_pos_a':   1.3, 'rhomb_neg_a':   1.3,
            'rhomb_pos_ac':  1.5, 'rhomb_neg_ac':  1.5,
            'oliv_a':    1.0, 'oliv_c': 1.3, 'oliv_b': 1.5, 'oliv_pencil': 1.2,
            'fsp_a010':  1.0, 'fsp_c010': 1.1, 'fsp_a001': 1.0, 'fsp_b001': 1.2,
            'fsp_pla':   1.1,
            'mica_basal': 0.6, 'mica_cross': 2.0,
        }
        for i, ss in enumerate(slip_systems):
            u[i] *= family_mult.get(ss.get('family', ''), 1.0)
        return u
    if mode == 'crss_lh1980':
        u = np.ones(n, dtype=np.float64)
        for i, ss in enumerate(slip_systems):
            u[i] = _CRSS_LH1980.get(ss.get('family', ''), 1.0)
        return u
    raise ValueError(f"unknown line_energy mode: {mode}")


# ---------------------------------------------------------------------------
# Per-pixel resolution
# ---------------------------------------------------------------------------

def _solve_one_pixel(A, alpha_vec, solver, line_energy_w, lp_options):
    """Solve one pixel's (A @ rho = alpha_vec, rho >= 0) problem."""
    if solver == 'nnls':
        from scipy.optimize import nnls
        rho, _res = nnls(A, alpha_vec)
        return rho
    if solver == 'lp':
        from scipy.optimize import linprog
        n = A.shape[1]
        res = linprog(c=line_energy_w,
                      A_eq=A, b_eq=alpha_vec,
                      bounds=[(0, None)] * n,
                      method=lp_options.get('method', 'highs'))
        if res.success:
            return res.x
        # Infeasible: alpha can't be exactly reproduced.  Fall back to NNLS.
        from scipy.optimize import nnls
        rho, _ = nnls(A, alpha_vec)
        return rho
    raise ValueError(f"solver must be 'nnls' or 'lp', got {solver!r}")


def resolve_gnd_per_pixel(nye_tensor_sample, euler1_deg, euler2_deg, euler3_deg,
                          phase_id_map, phase_name_lookup,
                          slip_db=None,
                          solver='nnls',
                          line_energy='uniform',
                          poisson_ratio=0.25,
                          lp_options=None,
                          progress=False,
                          n_jobs=1):
    """Resolve Nye tensor onto slip-system densities rho_t per pixel.

    Parameters
    ----------
    nye_tensor_sample : ndarray
        Shape (ny, nx, 2, 3) or (ny, nx, 3, 3).  The 2-row form is the
        measured d/dx, d/dy rows (i.e., alpha_ij for i in {1,2,3}, j in {1,2}).
        The 3-row form assumes alpha's 3rd row is zero or unmeasured.
    euler*_deg : ndarray (ny, nx)
        Bunge ZXZ Euler angles per pixel (degrees).
    phase_id_map : ndarray (ny, nx) of int
        Phase ID per pixel (0 = no phase).
    phase_name_lookup : callable or dict
        phase_id -> phase name string.  If callable, called with pid.
        If dict, looked up as phase_name_lookup[pid].
    slip_db : dict, optional
        Phase-name -> list-of-slip-systems.  Defaults to SLIP_SYSTEMS_DB.
    solver : {'nnls', 'lp'}
        Optimizer choice.
    line_energy : {'uniform', 'character', 'anisotropic'}
        Cost weight for the LP objective.  Ignored for NNLS.
    poisson_ratio : float
        For 'character' weights.
    lp_options : dict or None
        Extra kwargs forwarded to scipy.linprog (e.g. {'method': 'highs'}).
    progress : bool
        Print pixel count when each phase finishes.
    n_jobs : int
        Number of parallel workers for the per-pixel solver loop.  ``1``
        (default) runs serially; ``-1`` uses all CPUs.  Requires ``joblib``
        when > 1.  Speedup is typically ~n_cores for LP, ~0.5-0.7 n_cores
        for NNLS (dominated by Python overhead at very small per-pixel cost).

    Returns
    -------
    results : dict
        Mapping (phase_id, system_name) -> float32 density map (ny, nx).
        Zero where phase doesn't match or pixel was skipped.
    meta : dict
        Diagnostic info: per-phase solver choice, system counts, residuals.
    """
    if slip_db is None:
        slip_db = SLIP_SYSTEMS_DB
    if lp_options is None:
        lp_options = {}

    nye_arr = np.asarray(nye_tensor_sample)
    if nye_arr.ndim == 4 and nye_arr.shape[-2:] == (2, 3):
        ny, nx = nye_arr.shape[:2]
        _nye_is_2x3 = True
    elif nye_arr.ndim == 4 and nye_arr.shape[-2:] == (3, 3):
        ny, nx = nye_arr.shape[:2]
        _nye_is_2x3 = False
    else:
        raise ValueError(f"Unexpected nye_tensor_sample shape {nye_arr.shape}; "
                         "expected (ny, nx, 2, 3) or (ny, nx, 3, 3)")

    # phase_name_lookup as callable
    if callable(phase_name_lookup):
        _lookup_name = phase_name_lookup
    else:
        _lookup_name = lambda pid: phase_name_lookup.get(int(pid), None)

    phase_ids = [int(p) for p in np.unique(phase_id_map) if int(p) > 0]

    # Optional parallel backend
    _Parallel = None
    _delayed  = None
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
            _Parallel, _delayed = Parallel, delayed
        except ImportError:
            print('  joblib not available; falling back to serial (n_jobs=1)')
            n_jobs = 1

    results = {}
    meta = {
        'solver': solver, 'line_energy': line_energy,
        'phase_counts': {},
    }

    e1 = np.asarray(euler1_deg, dtype=np.float64)
    e2 = np.asarray(euler2_deg, dtype=np.float64)
    e3 = np.asarray(euler3_deg, dtype=np.float64)

    import time as _time
    for pid in phase_ids:
        pname = _lookup_name(pid)
        if pname is None or pname not in slip_db:
            continue
        systems = slip_db[pname]
        n_sys = len(systems)
        A = build_geometry_matrix(systems)
        u_w = line_energy_weights(systems, mode=line_energy,
                                  poisson_ratio=poisson_ratio)
        rho_maps = np.zeros((n_sys, ny, nx), dtype=np.float32)
        mask = (phase_id_map == pid)
        ys, xs = np.where(mask)
        n_px = len(ys)
        if n_px == 0:
            continue

        t0 = _time.time()

        # ---- Batch 1: Euler -> R (3x3 each) ----
        R_batch = euler_to_rotmat_batch(e1[ys, xs], e2[ys, xs], e3[ys, xs])   # (N, 3, 3)

        # ---- Batch 2: alpha in sample frame (N, 3, 3) ----
        alpha_sample = np.zeros((n_px, 3, 3), dtype=np.float64)
        if _nye_is_2x3:
            alpha_sample[:, :2, :] = nye_arr[ys, xs]
        else:
            alpha_sample[:, :, :]  = nye_arr[ys, xs]

        # ---- Batch 3: rotate to crystal frame a_c = R^T @ a_s @ R ----
        # einsum: a_c[p,i,k] = R[p,j,i] * a_s[p,j,l] * R[p,l,k]
        alpha_crystal = np.einsum('pji,pjl,plk->pik',
                                   R_batch, alpha_sample, R_batch)
        # Keep rows i in {0,1} (measurable), flatten to 6-vector for the solver
        alpha_vec_batch = alpha_crystal[:, :2, :].reshape(n_px, 6)

        t_prep = _time.time() - t0

        # ---- Solver loop (optionally parallel) ----
        #
        # NOTE: interrupting a joblib+loky job (Ctrl-C / Jupyter stop) can
        # crash the kernel because SIGINT cannot propagate into scipy.nnls /
        # linprog C code, and zombie worker processes may leave loky in a
        # half-dead state.  Workarounds, from safest to fastest:
        #   n_jobs=1               : serial, interrupt works cleanly
        #   n_jobs=-1 w/ threading : shares memory, but GIL-bound for NNLS
        #   n_jobs=-1 w/ loky      : fastest; but DO NOT interrupt mid-run
        t1 = _time.time()
        if n_jobs == 1:
            rhos = [
                _solve_one_pixel(A, alpha_vec_batch[k], solver, u_w, lp_options)
                for k in range(n_px)
            ]
        else:
            # Context-manager form: joblib tears down the worker pool cleanly
            # on normal exit AND on KeyboardInterrupt (fires __exit__), which
            # gives workers a chance to respond to SIGTERM instead of leaving
            # zombies when Jupyter sends the interrupt.
            try:
                with _Parallel(n_jobs=n_jobs, batch_size='auto',
                               backend='loky') as _pool:
                    rhos = _pool(
                        _delayed(_solve_one_pixel)(A, alpha_vec_batch[k],
                                                   solver, u_w, lp_options)
                        for k in range(n_px)
                    )
            except KeyboardInterrupt:
                print('  interrupt caught; joblib workers terminated')
                raise
        t_solve = _time.time() - t1

        # ---- Unpack rho vectors into per-system maps ----
        rho_mat = np.asarray(rhos, dtype=np.float32)   # (N, n_sys)
        for s in range(n_sys):
            rho_maps[s, ys, xs] = rho_mat[:, s]
        for s, ss in enumerate(systems):
            results[(pid, ss['name'])] = rho_maps[s]
        meta['phase_counts'][pid] = {
            'name': pname, 'n_systems': n_sys, 'n_pixels_solved': n_px,
            'prep_sec': t_prep, 'solve_sec': t_solve,
        }
        if progress:
            print(f'  phase {pid} ({pname}): {n_px} pixels, {n_sys} systems, '
                  f'solver={solver}, n_jobs={n_jobs}  '
                  f'(prep {t_prep:.1f}s, solve {t_solve:.1f}s)')
    return results, meta


def aggregate_by_family(results, slip_db=None, phase_name_lookup=None, meta=None):
    """Sum density maps across slip systems sharing the same 'family' label.
    Returns dict (phase_id, family) -> float32 density map.

    phase_name_lookup : callable or dict, optional
        Maps phase_id -> phase_name. Required when slip-system *names* collide
        across phases (e.g. '(010)[100] edge' lives in BOTH olivine and
        feldspar but with different family labels). Without it, the function
        falls back to first-match insertion-order scan and WILL mislabel.
    meta : dict, optional
        The `meta` dict returned by `resolve_gnd_per_pixel` — used as a
        backup to recover (pid -> pname) if no explicit lookup is given.
    """
    if slip_db is None:
        slip_db = SLIP_SYSTEMS_DB

    # Resolve a pid->pname lookup
    _pid2name = None
    if phase_name_lookup is not None:
        if callable(phase_name_lookup):
            _pid2name = phase_name_lookup
        else:
            _pid2name = lambda pid: phase_name_lookup.get(int(pid), None)
    elif meta is not None and 'phase_counts' in meta:
        _meta_map = {int(pid): info.get('name')
                     for pid, info in meta['phase_counts'].items()}
        _pid2name = lambda pid: _meta_map.get(int(pid), None)

    agg = {}
    for (pid, sname), rho in results.items():
        family = None
        # Phase-aware path: dispatch to the correct sub-DB
        if _pid2name is not None:
            pname = _pid2name(pid)
            if pname is not None and pname in slip_db:
                for ss in slip_db[pname]:
                    if ss['name'] == sname:
                        family = ss.get('family', sname); break
        # Fallback (legacy, ambiguous): first match across the whole DB
        if family is None:
            for _pname, systems in slip_db.items():
                for ss in systems:
                    if ss['name'] == sname:
                        family = ss.get('family', sname); break
                if family is not None: break
        if family is None:
            family = sname
        key = (pid, family)
        if key not in agg:
            agg[key] = np.zeros_like(rho)
        agg[key] += rho
    return agg

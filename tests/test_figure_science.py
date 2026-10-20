# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import numpy as np
import pytest

from dynamix.core.figure_fits import manual_slope, range_fit
from dynamix.core.figure_samples import direction_samples
from dynamix.core.orientation import Orientation, Frame
from dynamix.core.orientation_net import project, plane_pole
from dynamix.core.space_scale import dense_covariance, skeleton_covariance, self_similarity, local_energy


def test_manual_and_range_have_distinct_uncertainty_and_known_slope():
    points = [[-4, 10], [-1, 4]]
    m = manual_slope(points, factor=-1)
    assert m["value"] == 2 and "se" not in m
    assert manual_slope(points[::-1], factor=-1)["value"] == 2
    assert np.isnan(manual_slope([[0, 1], [0, 2]])["slope"])
    x = np.arange(10.)
    y = np.where(x<=4, 3*x, 12+(x-4))
    assert range_fit(x,y,0,4)["slope"] == pytest.approx(3)
    assert range_fit(x,y,5,9)["slope"] == pytest.approx(1)


def test_nets_and_plane_poles():
    trends = Orientation(np.array([0,90,180,270]), Frame.TRUE)
    for projection in ("schmidt", "wulff"):
        x,y = project(trends,0,projection=projection)
        assert x == pytest.approx([0,1,0,-1],abs=1e-12)
        assert y == pytest.approx([1,0,-1,0],abs=1e-12)
        x,y = project(trends,90,projection=projection)
        assert np.hypot(x,y) == pytest.approx(np.zeros(4),abs=1e-12)
    o,p = plane_pole(Orientation(0,Frame.TRUE),90)
    assert o.degrees == 270 and p==0
    with pytest.raises(ValueError): project(Orientation(0,Frame.PIXEL),0)


def test_all_chain_points_include_endpoints_but_not_singletons():
    x = np.arange(7)
    e = dict(x=x,y=np.zeros(7,int),mod=np.array([1,2,3,5,3,2,1]),arg=np.zeros(7),line_id=np.zeros(7,int))
    result = {"extrema":[e],"scales":[1.]}
    all_points = direction_samples(result,"wtmm")[0]
    peaks = direction_samples(result,"wtmmm")[0]
    assert all_points["point_id"].tolist() == list(range(7))
    assert peaks["point_id"].tolist() == [3]
    assert len(result["extrema"][0]["x"])==7


def test_dense_and_sparse_match_direct_masked_nonwrapping_pairs():
    rng = np.random.default_rng(2)
    a,b = rng.normal(size=(5,6)),rng.normal(size=(5,6))
    a[1,1] = b[2,3] = np.nan
    edges = np.array([.5,1.5,2.5,4.5])
    dense = dense_covariance(a,b,edges,min_pairs=1)
    coords = np.indices(a.shape)
    def sample(v):
        return dict(x=coords[1].ravel(),y=coords[0].ravel(),mod=np.exp(v.ravel()))
    sparse = skeleton_covariance(sample(a),sample(b),edges,min_pairs=1)
    sums,counts = np.zeros(3),np.zeros(3)
    ca,cb = a-np.nanmean(a),b-np.nanmean(b)
    for iy,ix in np.argwhere(np.isfinite(a)):
        for jy,jx in np.argwhere(np.isfinite(b)):
            r = np.hypot(jy-iy,jx-ix)
            k = np.searchsorted(edges,r,side="right")-1
            if r>0 and 0<=k<3:
                sums[k] += ca[iy,ix]*cb[jy,jx]
                counts[k] += 1
    assert dense["counts"] == pytest.approx(counts)
    assert dense["covariance"] == pytest.approx(sums/counts,abs=1e-10)
    assert sparse["covariance"] == pytest.approx(sums/counts,abs=1e-10)
    shifted = dense_covariance(a+20,b-7,edges,min_pairs=1)
    assert shifted["covariance"] == pytest.approx(dense["covariance"],abs=1e-10)


def test_kernel_identifies_deterministic_scale_multiplier():
    rng = np.random.default_rng(4)
    v = np.exp(rng.normal(0,.2,1000))
    p = np.linspace(-2,2,41)
    result = self_similarity([v,2*v],[1,2],[(0,1)],p)
    row = result["rows"][0]
    assert row["logabs"] == pytest.approx(np.zeros(p.size),abs=1e-12)
    assert row["phase"] == pytest.approx(-p*np.log(2),abs=1e-12)
    assert row["ratio"][20] == 1


def test_energy_uses_full_support_and_amplitude_rescaling():
    m = np.ones((32,32))
    first = local_energy(m,2)
    second = local_energy(m*3,2)
    good = np.isfinite(first)
    assert not good[0].any() and good[16,16]
    assert second[good]-first[good] == pytest.approx(np.full(good.sum(),np.log(3)))


def test_matching_marginals_do_not_imply_matching_spatial_covariance():
    y,x = np.indices((32,32))
    magnitude = .7*np.sin(x/5)+.3*np.cos(y/7)
    shuffled = np.random.default_rng(55).permutation(magnitude.ravel()).reshape(magnitude.shape)
    p = np.linspace(-1,1,21)
    one = self_similarity([np.exp(magnitude.ravel()),2*np.exp(magnitude.ravel())],[1,2],[(0,1)],p)
    other = self_similarity([np.exp(shuffled.ravel()),2*np.exp(shuffled.ravel())],[1,2],[(0,1)],p)
    assert one["rows"][0]["ratio"] == pytest.approx(other["rows"][0]["ratio"],abs=1e-12)
    edges = [0.5,1.5,3.5]
    correlated = dense_covariance(magnitude,magnitude,edges)
    uncorrelated = dense_covariance(shuffled,shuffled,edges)
    assert correlated["covariance"][0]>.1
    assert abs(uncorrelated["covariance"][0])<.03

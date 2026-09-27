Golden files for the LastWave dwtrans2d port (src/dynamix/core/mz_lastwave).

Each file is one run of the authors' C (LastWave package_dwtrans2d: dwt2d, dwt2r, extrema2, e2recons;
Bacry, Mallat, Zhong, Hwang et al.; GPL-2.0-or-later), extracted verbatim and compiled with
-ffp-contract=off, in its default periodic mode, on a synthetic input:

  noise64   white noise, 64 x 64, J = 4
  disc64    anti-aliased disc (radius 20), 64 x 64, J = 4
  square64  Gaussian-blurred square, 64 x 64, J = 3
  fbm128    doubly integrated white noise, 128 x 128, J = 5

"_lw_clip" runs use LastWave's own reconstruction settings (decay a = 1/5.8^(2/scale), clipping on) and
hold, per level l = 1..J: S_l, Wx_l, Wy_l, M_l, A_l (after dwt2d, including its level scaling),
extmask_l, extmagn_l (normalised magnitudes as extrema2 stores them), extmag_l (denormalised, as the
reconstruction uses them) and extarg_l; plus rec (the dwt2r identity reconstruction) and it_0, it_1,
it_5, it_20 (the reconstructed image after the initial pass and after 1, 5 and 20 iterations).

"_k1_noclip" runs use decay a = exp(-1/scale) (the paper's constant) with clipping off and hold only
input and it_0, it_1, it_5, it_20: the transform and the extrema do not depend on these settings.

Every file also holds input, J, periodic, kappa ("lw" or "1") and clip.

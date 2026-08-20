# The rank-3 regulariser

Upstream's volume-to-section solver reports one regularisation energy and descends another.
This page writes down what the term should be, what the code computes instead, and why the
difference reads as a coding slip rather than a modelling choice.

Line references are into `STalign/STalign.py` at the pinned commit
`b2068edc98974efa54537eca194736e177bbe11d`. This is ledger row
[D11](STALIGN_DIVERGENCES.md).

## What the term is

LDDMM penalises the time-dependent velocity field $v(t, x)$, $x \in \mathbb{R}^d$, by a
Sobolev norm — the deformation is smooth because rough velocities are expensive:

$$
E_R \;=\; \frac{1}{2\sigma_R^2}\int_0^1 \lVert v(t,\cdot)\rVert_V^2 \,\mathrm{d}t ,
\qquad
\lVert v \rVert_V^2 \;=\; \int L(\xi)\,\lvert \hat v(\xi) \rvert^2 \,\mathrm{d}\xi .
$$

The weight $L$ is the differential operator's symbol in frequency. Upstream builds it on the
velocity grid's own frequency axes (`:1388`), with $a$ the smoothing length and $p$ its power:

$$
L(\xi) \;=\; \Bigg( 1 + 2a^2 \sum_{k=1}^{d} \frac{1 - \cos\!\big(2\pi \xi_k \Delta_k\big)}{\Delta_k^{2}} \Bigg)^{\!2p}
$$

which is the finite-difference Laplacian's symbol, so $L$ grows with frequency along **every**
axis $k = 1 \dots d$. That sum over all $d$ axes is the part to hold on to.

Discretised on a grid of $N_k$ samples per axis with spacing $\Delta_k$, write
$N = \prod_k N_k$ and $\Delta V = \prod_k \Delta_k$. Parseval's identity for the DFT,
$\sum_x \lvert v(x) \rvert^2 = N^{-1} \sum_\xi \lvert \hat v(\xi) \rvert^2$, turns the norm into

$$
E_R \;=\; \frac{\Delta V}{2\,\sigma_R^{2}}\cdot\frac{1}{N}\sum_{\xi} L(\xi)\,\big\lvert \hat v(\xi) \big\rvert^{2} .
$$

Three things travel together in that expression: the transform runs over all $d$ spatial axes,
$L$ is indexed by $d$ frequencies, and the normalisation divides by all $d$ sizes. Break any
one of them and the expression stops being a norm.

## What the code computes

At **rank 2** (`:1193`), with `v` of shape $(n_t, N_1, N_2, 2)$:

```python
ER = torch.sum(torch.sum(torch.abs(torch.fft.fftn(v,dim=(1,2)))**2,dim=(0,-1))*LL)*DV/2.0/v.shape[1]/v.shape[2]/sigmaR**2
```

Axes 1 and 2 *are* both spatial axes, `LL` is indexed by both frequencies, and the division is
by $N_1 N_2 = N$. This is the formula above, exactly.

At **rank 3** (`:1504`), with `v` of shape $(n_t, N_1, N_2, N_3, 3)$, the line is
**byte-identical**:

```python
ER = torch.sum(torch.sum(torch.abs(torch.fft.fftn(v,dim=(1,2)))**2,dim=(0,-1))*LL)*DV/2.0/v.shape[1]/v.shape[2]/sigmaR**2
```

Now `dim=(1,2)` is two of *three* spatial axes, and the divisor is $N_1 N_2$ with $N_3$
missing. The quantity actually computed is

$$
\tilde E_R \;=\; \frac{\Delta V}{2\,\sigma_R^{2}\,N_1 N_2}
\sum_{\xi_1}\sum_{\xi_2}\sum_{x_3}
L\big(\xi_1,\xi_2,\underbrace{x_3}_{\text{a position}}\big)\;
\big\lvert \hat v(\xi_1,\xi_2,x_3) \big\rvert^{2} .
$$

The third argument of $L$ is a **spatial index** being read as a frequency index. `LL` has
shape $(N_1, N_2, N_3)$ and the array it multiplies has shape $(N_1, N_2, N_3)$, so the
elementwise product broadcasts happily and nothing raises — but the third axis of one is
frequency and the third axis of the other is position. There is no norm, Sobolev or otherwise,
that $\tilde E_R$ discretises.

## Why this reads as a bug

**The gradient was generalised and the energy was not.** The smoothing applied to this
energy's own gradient is `dim=(1,2)` at rank 2 (`:1215`) and `dim=(1,2,3)` at rank 3
(`:1527`). Whoever extended the file to three dimensions updated the gradient line and left
the energy line untouched — and the energy line is byte-identical to its rank-2 counterpart. So
upstream reports $\tilde E_R$ while descending a gradient smoothed over all three axes. A
deliberate change of objective would have moved both lines, or neither.

**Neither reading of Parseval closes.** Suppose the two-axis transform were intended. A
partial Parseval over axes 1 and 2 divides by $N_1 N_2$ — which the code does — but then $L$
must be indexed on those two axes only, and it is not. Suppose instead the three-axis $L$ were
intended. Then the transform must span three axes and the divisor must include $N_3$, and
neither holds. The line is not a consistent discretisation of any objective.

**The discrepancy is not a constant, so it cannot be absorbed into $\sigma_R$.** The ratio
$\tilde E_R / E_R$ depends on the field's spectrum. Measured by sweeping the smoothness of a
random field on a $6\times7\times8$ velocity grid:

| Gaussian smoothing of $v$ | $\tilde E_R / E_R$ |
| --- | --- |
| none (white noise) | 0.98 |
| $\sigma = 0.5$ voxels | 1.20 |
| $\sigma = 1.0$ | 2.89 |
| $\sigma = 1.5$ | 4.17 |
| $\sigma = 2.0$ | 4.91 |

White noise spreads energy evenly across frequencies, so dropping one axis' transform costs
almost nothing. A smooth field concentrates energy at low frequency along every axis —
including the one never transformed — and the reading diverges. A *fitted* velocity field is
smooth by construction, because the Sobolev kernel $K = 1/L$ makes it so, which is why the
defect is invisible on a toy field and large on a real fit. On the reference velocity field in
this repository's fixtures the ratio is **≈ 5.4**.

## What it costs

Measured on `starmap-allen3Datlas-alignment` at 4000 iterations, one H100, float64:

| | $E_R$ (three-axis, correct) | $\tilde E_R$ (upstream) |
| --- | --- | --- |
| value on the fitted field | **0.744** | **2081** |
| share of the total objective | 0.001 % | 2.6 % |
| inflation, same field | — | 3460–8880× |

The inflation is dominated by the missing $\div N_3$, which is $\div 114$ on this grid.

The practical consequence is not that the number is wrong but that it changes whether the
regulariser does anything at all. At the notebook's $\sigma_R = 10^8$ the *correct* term
contributes one part in $10^5$ of the objective — it is numerically switched off, and the fit
runs effectively unregularised. Upstream's inflated term lands in the range where it bites.
The fitted velocity fields show it directly: rms $\lvert v \rvert$ of **20.7** for the correct
term against **9.4** and **9.9** for two runs of upstream. Restoring the weight the defect
supplies by accident takes $\sigma_R \approx 10^{6}$, roughly 100× stronger than the
notebook's value.

Reproducing $\tilde E_R$ in the port moves the fitted velocity field by a factor of 30 or more:
`v` relative L2 goes **1.802 → 0.046** on `merfish-allen3Datlas` and **0.930 → 0.163** on
`starmap-allen3Datlas`. It also makes the reported objective climb while the fit improves —
squidpy's total drop flips from **+3.66 %** to **−5.15 %** when it reports the two-axis energy
while descending the three-axis gradient, which is the mismatch made visible from the other
side.

## Scope

Rank 3 only. At rank 2, `dim=(1,2)` is every spatial axis, so `:1193` is correct and all
fifteen two-dimensional notebooks in the comparison are untouched. Of the seventeen pinned
notebooks, only `merfish-allen3Datlas-alignment` and `starmap-allen3Datlas-alignment` reach
`LDDMM_3D_to_slice` at all.

squidpy uses every spatial axis in both places, and pins that against upstream's own line in
`tests/test_stalign_reference.py::test_slice_regularizer_axes_diverge_from_upstream`.

# Loss-function engineering for DS102 (F1-macro), trained strictly from scratch

`loss_lab/` adds modular, unit-tested loss functions (linear/NN and LightGBM), a
from-scratch trainer, macro-F1 post-processing and a **fully nested** cross-validated
ablation. Nothing in v1–v60 is modified; existing scripts keep working.

> **Tóm tắt (VI).** Thư mục này thử nghiệm các hàm loss cho mô hình tuyến tính TF-IDF và LightGBM của DS102, tất cả học từ đầu (không pretrained), thống kê loss/ngưỡng chỉ tính trong phần train của từng fold (CV lồng nhau 5×3). Kết quả và khuyến nghị nằm ở mục 6–7.

<!-- RESULTS_TLDR -->

---

## 1. Technical audit

### 1.1 Problem and metric
* **Modality / task:** Vietnamese e-commerce reviews, NLP (plus light tabular metadata) → **single-label 4-class classification** of `spam_label`: 0 no-spam, 1 fake review, 2 review on brand/shop only, 3 non-review.
* **Metric:** **F1-macro** over labels 0–3 (`CONSTRAINTS.md`, `README.md`). Every class counts 25 % regardless of size, so the 1.5 % fake class weighs as much as the 73 % majority class.
* **Data (SHA-256 identical to the hashes in the repo manifests):** train 14 306, dev 1 590, test 3 974 rows; columns `comment, rating, product_name, description, category, categories, num_sold, num_review`.

### 1.2 Target distribution and label quality

| | class 0 | class 1 (fake) | class 2 (brand-only) | class 3 (non-review) |
|---|---|---|---|---|
| train count (%) | 10 512 (73.5) | **211 (1.47)** | 1 103 (7.7) | 2 480 (17.3) |
| dev count | 1 183 | **17** | 120 | 270 |
| median comment tokens | 21 | **50** | 19 | 11 |

* **Imbalance:** 49.8 : 1 (majority : fake). sklearn `balanced` weights give the fake class **47×** the weight of class 0 inside a fold; sqrt-balanced 6.9×; effective-number (β = 0.999) 6.1×.
* **Heavy tails:** comment length skew 1.89, excess kurtosis 5.33; fakes are long (copy-pasted product descriptions, template praise), non-reviews short.
* **Label noise / boundary ambiguity:** identical normalised texts carry conflicting labels — e.g. "rất hài lòng" → {0,0,3,3,3}, "hài lòng" → {2,3}, "giá tốt" → {2,3}. The 0↔2 (product vs. shop/delivery review) boundary is semantically fuzzy and dominates the error mass of the best ensemble (443 class-2 → 0 and 338 class-0 → 2 OOF errors in `output_v36_experiment`).
* **Template campaigns:** a large share of fakes have a near-duplicate (token-Jaccard ≥ 0.6) fake in the fitting data (108/211 under the repo's 3-fold split, `CLAUDE_REVIEW.md`), which inflates class-1 OOF scores; the repo therefore also tracks **novel-fake F1** (class-1 F1 on rows without such a template). Folds of this study hold 33/49/38/43/48 fakes.
* 99.4 % of dev/test product names occur in train; 13 dev and 39 test comments occur verbatim in train.

### 1.3 Current baseline and its losses
Best leaderboard file: **V11 experimental, Kaggle 0.77436** = 0.7·[0.5·softmax(3·v2) + 0.5·CatBoost-text] + 0.3·latent-CatBoost (SVD-128), then class multipliers [1, 1.2, 1.2, 0.9].

| component | model (all from scratch) | training loss |
|---|---|---|
| v2 linear (largest weight) | `LinearSVC(C=0.5, class_weight='balanced')` on FeaturesV2 (word 1–3 g + char 2–5 g TF-IDF + product/category/rating/dense meta) ×0.75, NB-SVM ×0.25 | **L2-regularised squared hinge, one-vs-rest**, liblinear weighting (positives × w_c, negatives × 1) |
| CatBoost text / latent | CatBoost `MultiClass` | softmax cross-entropy, `SqrtBalanced` weights |
| LightGBM experiments (v18/v20) | built-in `multiclass` | softmax CE, sqrt-balanced weights |
| CNN / MLP experiments (v14, v37, H2) | PyTorch from random init | weighted CE; class-balanced focal in H2 (pooled OOF 0.7246 vs 0.7241 — no gain) |

Macro-F1 never enters training: it is chased post hoc with class multipliers/offsets picked on OOF.

## 2. Mathematical diagnosis

### 2.1 Surrogate–metric gap
Let $\eta_c(x)=P(y=c\mid x)$, $\pi_c$ the class rate and $q_c$ the predicted rate. Macro-F1 is $\frac1K\sum_c \frac{2\,C_{cc}}{\pi_c+q_c}$ (confusion $C$ in probabilities): **non-decomposable** (a ratio of sums), **piecewise constant** in the parameters (argmax), and its optimal classifier is *not* argmax. Linearising around the optimum gives

$$h^\star(x)=\arg\max_c \frac{2\eta_c(x)-F1_c^\star}{\pi_c+q_c}\qquad(\text{binary case: threshold } \eta>F1^\star/2),$$

i.e. a per-class rescaling/threshold that depends on the achievable F1 of each class.

* **Squared hinge OvR with `balanced` weights** is a cost-sensitive surrogate with $c_{FN}/c_{FP}=w_c$, so each sub-problem estimates the level set $\eta_c = 1/(1+w_c)$: thresholds **[0.746, 0.059, 0.237, 0.407]** for classes 0–3, versus the F1-optimal $\approx F1_c^\star/2$ (≈ 0.3 for fakes). The hinge is not a proper scoring rule — it identifies only $\operatorname{sign}(\eta_c-t_c)$ at its own $t_c$ — so post-hoc threshold moves to the F1-optimal level rely on ranking information the loss never promised. The four OvR margins are also trained independently and are not on a common scale (the repo's `softmax(3·margin)` is a hand-set temperature).
* **Class-weighted CE** has minimiser $p_w(c\mid x)\propto w_c\,\eta_c(x)$: argmax then thresholds at weight-induced, not F1-induced, levels; correcting it is again a per-class offset.
* **Consequence:** a *proper* loss (CE family) gives consistent posterior estimates on which a tuned per-class offset is a consistent plug-in rule for F-measures (Koyejo et al. 2014; Narasimhan et al. 2015). A direct differentiable surrogate (soft macro-F1) instead bakes the threshold into training.

### 2.2 From-scratch optimisation dynamics
* At zero initialisation every logit is 0 ($p=1/4$). With `balanced` weights the first L-BFGS directions are dominated by the 170-odd fakes carrying 47× weight — exactly the rows that contain template campaigns — so early updates memorise template n-grams.
* The model has $d\approx 1.3\text{–}1.7\times10^5$ features for $n\approx 7.6\text{–}11.4\times10^3$ rows: the data are almost linearly separable. In this regime CE drives margins (and confidence) up on already-separated easy rows, limited only by L2; **importance weights lose influence as regularisation weakens** (the solution approaches the max-margin direction, which ignores weights; Soudry et al. 2018, Byrd & Lipton 2019), whereas **additive logit adjustments change the margin constraints** $z_y-z_c\ge\tau\log(\pi_c/\pi_y)$ and keep enlarging rare-class margins (Menon et al. 2021).
* Squared hinge grows **quadratically** with violation, so the label-conflicting short reviews (§1.2) exert outsized pull on the boundary; CE grows linearly; focal *up*-weights hard (possibly mislabeled) rows relative to easy ones.
* Overconfidence: with weak L2, softmax outputs saturate on train, which makes raw argmax thresholds brittle across folds — hence nested offsets and temperature scaling below.

## 3. Candidate losses (all implemented in `losses.py`, analytic gradients)
Notation: $z\in\mathbb R^K$ logits, $p=\mathrm{softmax}(z)$ via log-sum-exp, $t$ true class, weights normalised so $\sum_i w_{y_i}=n$; every statistic ($w$, $\pi$, $n_c$) comes from the **training part of the current fold**.

| id | loss | equation | gradient wrt $z_j$ |
|---|---|---|---|
| **B0** baseline | squared hinge OvR (repo loss) | $\sum_c W_{c}\max(0,1-T_c z_c)^2$, $T_c=\pm1$, $W_c=w_c$ if $t=c$ else 1 | $-2W_jT_j\max(0,1-T_jz_j)$ |
| B1 reference | class-weighted CE | $-w_t\log p_t$ | $w_t(p_j-[j=t])$ |
| **C1** | class-balanced focal | $-\alpha_t(1-p_t)^\gamma\log p_t$, $\alpha_c\propto\frac{1-\beta}{1-\beta^{n_c}}$, $\gamma=2,\beta=0.999$ | $\alpha_t G(p_t)(p_j-[j=t])$, $G(u)=(1-u)^\gamma-\gamma u(1-u)^{\gamma-1}\log u\ge0$ |
| **C2** | logit-adjusted CE | $-\log\mathrm{softmax}(z+\tau\log\pi)_t$, $\tau=1$; predict with $z$ | $\mathrm{softmax}(z+\tau\log\pi)_j-[j=t]$ |
| **C3** | composite, annealed | $\alpha_s\,\mathrm{CE}_w+(1-\alpha_s)\big(1-\tfrac1K\sum_c\tfrac{2TP_c}{\sum_ip_{ic}+N_c}\big)$ | soft-F1 part: $\partial F1_c/\partial p_{ic}=\frac{2}{D_c}\big(y_{ic}-\tfrac{F1_c}{2}\big)$, then through softmax |
| **C4** | composite, annealed | same with LA-CE as base | — |

* **Soft macro-F1's gradient** $\frac{2}{D_c}(y_{ic}-F1_c/2)$ re-weights each class by $1/D_c$ (automatic balancing) and pushes every row towards the F1-optimal threshold $F1_c/2$ — the same structure as $h^\star$ in §2.1. It is non-decomposable, so the trainer is **full-batch**: every gradient is exact for the whole fold, not a biased mini-batch estimate.
* **Schedule** $\alpha_s = \alpha_{\min}+(1-\alpha_{\min})\tfrac12(1+\cos\tfrac{\pi s}{S-1})$, $S=4$, $\alpha_{\min}=0.25$ → $\alpha=[1, 0.81, 0.44, 0.25]$: stage 0 is pure base loss from zero init (stable, convex for CE/LA), later stages warm-start and sharpen on the metric (continuation instead of re-weighting inside one L-BFGS run, which would corrupt its curvature pairs).
* **GBDT (LightGBM custom `fobj`)**: class-balanced focal with exact gradient $g_{ij}$ above and exact diagonal Hessian
  $H_{ij}=\alpha_t\big[-G'(u)\,u\,(\,[j=t]-p_j)^2+G(u)\,p_j(1-p_j)\big]$,
  $G'(u)=-2\gamma(1-u)^{\gamma-1}-\gamma(1-u)^{\gamma-1}\log u+\gamma(\gamma-1)u(1-u)^{\gamma-2}\log u$.
  Focal loss is not convex in the logits ($H$ can be negative for badly misclassified rows), so the Newton Hessian is floored at $10^{-6}\alpha_t$; it is multiplied by $K/(K-1)$ to match LightGBM's built-in multiclass step (with that factor, γ = 0 reproduces built-in `multiclass` to 3×10⁻⁹). The paired `feval` is the exact competition macro-F1. Logit adjustment for GBDT is a training-only `init_score` of $\tau\log\pi$.

## 4. Implementation and verification

| file | content |
|---|---|
| `losses.py` | all losses (`SoftmaxCrossEntropy` + label smoothing, `FocalLoss`, `LogitAdjustedCrossEntropy`, `SoftMacroF1Loss`, `SquaredHingeOvR`, `CompositeLoss`), weight schemes, macro-F1, LightGBM `focal_grad_hess` / `make_lgb_focal_objective` / `lgb_macro_f1_eval` |
| `linear_model.py` | `LinearLossClassifier`: $z=XW+b$, zero (or Glorot) init, full-batch L-BFGS, C-per-sample-sum convention of LinearSVC, staged composite schedule; builds the loss from the training labels only |
| `postprocess.py` | per-class logit offsets by coordinate ascent with `scipy.optimize.brute` (macro-F1 is a step function), temperature scaling with `scipy.optimize.minimize_scalar` |
| `data.py` | repo `load_data`/`prepare`/`FeaturesV2` reused; grouped splits; macOS guard |
| `run_ablation.py`, `run_gbdt_ablation.py` | nested CV benchmarks with checkpoints and manifests |
| `analyze_strata.py` | post-hoc class-1 template/novel strata (repo definition) |
| `pilot_grid.py` | grid-range pilot on inner data only |
| `test_losses.py` | 20 unit checks |

Numerical stability: log-softmax via log-sum-exp; $1-p$ computed as `-expm1(log p)` and clamped at `eps = 1e-7` before negative powers; soft-F1 denominators $+10^{-7}$; non-finite logits rejected; every objective/gradient checked finite during optimisation.

`python -m unittest test_losses -v` (inside `loss_lab/`) checks, among others:
* analytic gradients of **all 10 loss variants** vs central finite differences (relative error < 1e-6, weighted and unweighted) and non-zero gradient variance;
* focal γ = 0 ≡ CE, LA τ = 0 ≡ CE, composite α ∈ {0,1} endpoints, softmax-shift invariance (rows of ∂L/∂z sum to 0);
* finiteness at logits up to ±1e8; soft-F1 → hard F1 when saturated; macro-F1 ≡ sklearn;
* LightGBM focal: gradient vs finite differences, **exact Hessian vs finite differences of the gradient** for γ ∈ {0, 0.5, 1, 2, 3}, **Hessian used > 0**, γ = 0 Hessian ≡ $p(1-p)$, and custom CE ≡ built-in `multiclass`;
* the squared-hinge replica matches sklearn/liblinear `LinearSVC` decision values (max |Δ| < 2e-3);
* class weights are fold statistics (sum to n, rarer ⇒ heavier, β → 1 ⇒ inverse frequency).

A mutation test (dropping the focal modulating derivative, or the ½ in the soft-F1 gradient) makes the suite fail, so the checks are sensitive.

## 5. Ablation protocol
* **Fixed for every row:** 4 CSVs, `prepare`, outer **5-fold `StratifiedGroupKFold`** grouped by normalised comment (seed 102), **inner 3-fold** split of each outer-train part (seed 102), `FeaturesV2` refitted on every (outer or inner) training part, linear model with zero init and full-batch L-BFGS, identical post-processing.
* **Nested selection, no leakage:** for each outer fold and loss: C chosen on inner OOF macro-F1 (after offsets); offsets $b$ and temperature $T$ fitted on inner OOF at $C^\star$; model refitted on the whole outer-train part; the outer fold is scored once. Class weights, priors and effective numbers are recomputed inside every fit from its own training labels. Dev is never used for any choice: dev predictions average the five outer models' logits, with offsets tuned on pooled train OOF.
* **Pre-registration:** losses and their hyper-parameters (γ = 2, β = 0.999, τ = 1, S = 4, α_min = 0.25) fixed from the literature; C grids set by `pilot_grid.py` on an inner split of outer-fold-0's training part (never an outer validation fold). After the pilot, the composite family was extended with an LA base (C4) because the pilot showed soft-F1 helps raw but not post-processed F1 on a CE base; no other change.
* **Reported per row:** final training loss and validation loss (each in its own loss), validation NLL raw and temperature-scaled, outer-fold macro-F1 raw and post-processed (mean ± std over folds), pooled OOF macro-F1, per-class F1, novel/template fake F1, dev macro-F1, paired per-fold deltas and a **paired cluster bootstrap** (2 000 draws over comment groups) against the controlled baseline B0 and the as-deployed v2 linear component R0.

## 6. Results

All numbers are out-of-fold macro-F1 on the 14 306 training rows (5 outer folds, seed 102) unless marked *dev*. "nested-post" = offsets from inner OOF (strict); "cross-fit" = offsets for fold k tuned on the other outer folds' OOF (scale-matched: those models are the same size; fold k's labels are never used). Train/val loss are each in the row's **own** loss units (not comparable across rows); val NLL is the common calibration yardstick. Full tables: `output/ablation_seed102/ablation_table.md`, `output/gbdt_seed102/ablation_table.md`.

### 6.1 Linear text model (FeaturesV2, zero init, full-batch L-BFGS)

| Row | C* per fold | Train loss | Val loss | Val NLL raw / T-scaled | OOF F1 raw | OOF F1 nested-post | OOF F1 cross-fit | Per-class F1 nested-post [0,1,2,3] | Novel-fake F1 (TP/96, FP) | Dev F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 repo v2 (SVM 0.75 + NB-SVM 0.25), as deployed | 0.5 (fixed) | — | — | 0.452 / 0.300 | 0.7651 ± 0.0247 | 0.7603 ± 0.0207 | 0.7629 ± 0.0216 | 0.932, 0.647, 0.591, 0.872 | 0.329 (26, 36) | 0.7266 |
| R1 sklearn LinearSVC, as deployed | 0.5 (fixed) | 0.1212 | 1.1830 | 0.462 / 0.304 | 0.7623 ± 0.0245 | 0.7603 ± 0.0197 | 0.7611 ± 0.0230 | 0.932, 0.647, 0.587, 0.873 | 0.331 (26, 35) | 0.7207 |
| **B0 squared hinge OvR, balanced (baseline loss)** | 0.125–0.25 | 0.3036 | 1.1255 | 0.519 / 0.305 | **0.7660 ± 0.0228** | 0.7610 ± 0.0216 | **0.7642 ± 0.0236** | 0.931, 0.649, 0.591, 0.870 | 0.341 (28, 40) | 0.7199 |
| B1 class-weighted CE | 2 | 0.1073 | 0.5827 | 0.341 / 0.334 | 0.7509 ± 0.0226 | 0.7603 ± 0.0217 | 0.7558 ± 0.0195 | 0.930, 0.648, 0.593, 0.869 | 0.353 (30, 44) | 0.7066 |
| C1 class-balanced focal | 8 | 0.0095 | 0.1993 | 0.326 / **0.292** | 0.7464 ± 0.0223 | 0.7555 ± 0.0215 | 0.7521 ± 0.0247 | 0.930, 0.637, 0.584, 0.867 | 0.369 (31, 41) | 0.7230 |
| C2 logit-adjusted CE | 2 (one fold 8) | 0.0935 | 0.2927 | 0.493 / 0.481 | 0.6810 ± 0.0205 | 0.7588 ± 0.0226 | 0.7610 ± 0.0235 | 0.930, 0.637, 0.593, 0.872 | 0.345 (30, 48) | 0.7093 |
| C3 composite CE_bal → soft-F1 | 2 (one fold 8) | 0.1169 | 0.4473 | **0.298** / 0.295 | 0.7586 ± 0.0203 | **0.7637 ± 0.0253** | 0.7541 ± 0.0224 | 0.931, 0.660, 0.595, 0.866 | **0.375** (30, 34) | 0.7206 |
| C4 composite LA → soft-F1 | 2 (one fold 8) | 0.1276 | 0.3235 | 0.301 / 0.298 | 0.7581 ± 0.0158 | 0.7598 ± 0.0206 | 0.7553 ± 0.0163 | 0.931, 0.655, 0.588, 0.865 | 0.372 (29, 31) | 0.7216 |

Paired against the baseline loss B0 (fold deltas and a 2 000-draw cluster bootstrap over comment groups):

| Row | nested-post Δ [95 % CI] (folds > B0, P(Δ>0)) | cross-fit Δ [95 % CI] (folds > B0, P(Δ>0)) |
|---|---|---|
| R0 | −0.0001 [−0.0060, +0.0059] (3/5, 0.50) | −0.0008 [−0.0077, +0.0058] (2/5, 0.40) |
| R1 | −0.0005 [−0.0075, +0.0064] (2/5, 0.45) | −0.0029 [−0.0097, +0.0039] (2/5, 0.19) |
| B1 | −0.0005 [−0.0065, +0.0057] (3/5, 0.43) | **−0.0090 [−0.0159, −0.0023]** (1/5, 0.00) |
| C1 | −0.0057 [−0.0151, +0.0033] (2/5, 0.10) | **−0.0122 [−0.0212, −0.0032]** (0/5, 0.00) |
| C2 | −0.0022 [−0.0082, +0.0038] (2/5, 0.24) | −0.0034 [−0.0110, +0.0041] (2/5, 0.18) |
| C3 | +0.0027 [−0.0041, +0.0095] (3/5, 0.77) | **−0.0102 [−0.0174, −0.0029]** (1/5, 0.00) |
| C4 | −0.0008 [−0.0084, +0.0067] (3/5, 0.42) | **−0.0091 [−0.0174, −0.0009]** (2/5, 0.02) |

Reading:
* **No candidate loss beats the repo's squared hinge on the linear model.** The best raw and the best cross-fit scores both belong to B0; C3's +0.0027 under nested offsets is inside its CI and flips sign under cross-fit offsets. Under the scale-matched protocol, weighted CE, CB-focal and both composites are *significantly worse*.
* **Focal loss is the worst choice** on this architecture under both protocols, matching the repo's earlier H2 CNN result.
* **What the theory predicted, and did happen:** soft-F1 moves CE's own decision rule toward macro-F1 (+0.0077 raw over B1) and gives the best-calibrated raw probabilities (NLL 0.298 without temperature); logit adjustment needs re-thresholding (raw 0.681 → 0.761 cross-fit) because τ = 1 targets balanced error, not F1. Neither beats the hinge, whose implicit OvR thresholds already sit close to the F1 optimum here.
* **Novel fakes:** the gentler objectives (CB-focal, composites) trade some template memorisation for slightly better novel-fake F1 (0.341 → 0.375, mostly fewer false positives: 40 → 31–34). With 96 novel fakes this is a hint, not evidence.
* **Regularisation, not loss:** nested selection chose C = 0.125–0.25 for the hinge in every fold (repo uses 0.5): +0.0037 raw / +0.0031 cross-fit over R1, not significant.
* Loss-diverse averaging (`explore_blends.py`, hypothesis-generating only): no pair of these linear models beats B0 by more than +0.002 cross-fit.

### 6.2 LightGBM on latent SVD-128 features (repo v18 parameters, macro-F1 `feval` early stopping on the inner fold)

| Row | Rounds per fold | Train loss | Val loss | Val NLL raw / T | OOF F1 raw | OOF F1 nested-post | OOF F1 cross-fit | Per-class F1 nested-post | Dev F1 |
|---|---|---|---|---|---|---|---|---|---|
| G0 built-in multiclass + SqrtBalanced (repo recipe) | 83–104 | 0.1817 | 0.5017 | 0.329 / 0.316 | 0.7410 ± 0.0217 | 0.7387 ± 0.0172 | 0.7440 ± 0.0214 | 0.917, 0.646, 0.517, 0.874 | 0.6808 |
| G1 same loss as custom objective (RNG noise floor) | 73–111 | 0.1839 | 0.5037 | 0.330 / 0.317 | 0.7362 ± 0.0197 | 0.7392 ± 0.0169 | 0.7375 ± 0.0183 | 0.919, 0.643, 0.524, 0.871 | 0.6934 |
| G2 class-balanced focal, exact Hessian | 72–112 | 0.0438 | 0.2107 | 0.429 / 0.324 | 0.7084 ± 0.0213 | 0.7364 ± 0.0215 | 0.7385 ± 0.0214 | 0.920, 0.637, 0.520, 0.869 | 0.6680 |
| **G3 logit-adjusted CE (τ log π init_score)** | 390–495 | 0.0145 | 0.3429 | 0.348 / 0.348 | 0.7369 ± 0.0197 | **0.7501 ± 0.0153** | **0.7565 ± 0.0195** | 0.923, 0.649, **0.547**, 0.880 | 0.6906 |
| G0-LL built-in + SqrtBalanced, log-loss ES | 170–188 | 0.0795 | 0.5233 | 0.305 / 0.305 | 0.7315 ± 0.0195 | 0.7443 ± 0.0159 | 0.7514 ± 0.0222 | 0.918, 0.646, 0.537, 0.877 | 0.7073 |
| G4 unweighted CE, log-loss ES (plug-in) | 132–148 | 0.0931 | 0.3161 | 0.316 / 0.316 | 0.6765 ± 0.0179 | 0.7416 ± 0.0175 | 0.7374 ± 0.0189 | 0.921, 0.636, 0.534, 0.874 | 0.6934 |

| Row | Δ vs G0 nested [95 % CI] (folds) | Δ vs G0 cross-fit [95 % CI] (folds) | Δ vs G0-LL nested (folds, P) | Δ vs G0-LL cross-fit (folds, P) |
|---|---|---|---|---|
| G1 (noise floor) | +0.0007 [−0.0044, +0.0057] (3/5) | −0.0049 [−0.0122, +0.0024] (1/5) | — | — |
| G2 focal | −0.0020 [−0.0109, +0.0063] (2/5) | −0.0050 [−0.0142, +0.0042] (1/5) | −0.0083 (1/5, 0.03) | −0.0126 (1/5, 0.00) |
| **G3 LA** | **+0.0114 [+0.0032, +0.0196] (5/5)** | **+0.0128 [+0.0046, +0.0212] (5/5)** | +0.0050 [−0.0031, +0.0131] (5/5, 0.89) | +0.0051 [−0.0022, +0.0128] (5/5, 0.91) |
| G0-LL | +0.0063 [+0.0011, +0.0118] (4/5) | +0.0077 [+0.0014, +0.0142] (5/5) | — | — |
| G4 | +0.0029 [−0.0058, +0.0114] (4/5) | −0.0061 [−0.0151, +0.0026] (1/5) | −0.0034 (1/5, 0.22) | −0.0137 (0/5, 0.00) |

Reading:
* **Logit adjustment is the one loss change that wins**: +0.011–0.013 over the repo-style objective on every fold with CIs above zero, and +0.005 over the stronger log-loss-early-stopped baseline on every fold (CI still spans zero). Its gain concentrates on the weakest class (class 2 F1 0.517 → 0.547).
* **Mechanism:** with row-weighted CE, raw-argmax macro-F1 on the inner fold peaks after ~90 rounds, so metric-based early stopping halts an under-fitted booster; G0-LL shows ~half of G3's gain is recovered just by stopping on log-loss. LA keeps the booster's own decision rule aligned with the rare classes throughout training, so the metric keeps improving for ~450 rounds, and it adds additive rare-class margins without inflating 170 fake rows into 47× (or 6.9×) heavier gradients and Hessians — the leaf statistics stay unweighted.
* **Focal loss again does not help** (G2), and the custom-objective machinery is validated: G1 (identical loss) differs from G0 only by LightGBM's column-sampling RNG, which bounds the noise floor at ≈ ±0.005 on the mean.

<!-- RESULTS_REPLICATION -->

## 8. Using it in the existing pipeline (nothing existing is modified)

* **LightGBM / any GBDT with logit adjustment** — the recommended change. Drop the class/row weights and pass a training-only init score; predict raw scores without it, then softmax and tune per-class offsets on OOF:
  ```python
  from loss_lab.losses import log_class_prior          # fold-local prior from y_train only
  init = np.tile(log_class_prior(y_tr, 4), (len(y_tr), 1))
  booster = lgb.train(params | {'objective': 'multiclass'}, lgb.Dataset(X_tr, y_tr, init_score=init), ...)
  z = booster.predict(X_new, raw_score=True)          # no init score here: LA decisions
  ```
  CatBoost: `Pool(X_tr, y_tr, baseline=init)` with `auto_class_weights` removed, and `predict(..., prediction_type='RawFormulaVal')` on a baseline-free Pool (`run_catboost_la.py`).
* **Custom LightGBM objective with exact Hessian:** `make_lgb_focal_objective(gamma, class_weight, hess_scale=K/(K-1))` + `feval=lgb_macro_f1_eval`; LightGBM does not apply Dataset weights to custom gradients, the objective does.
* **Any differentiable model (numpy, or wrap in `torch.autograd.Function`)**: every loss in `losses.py` returns `(value, dL/dz)`; `LinearLossClassifier(spec, C)` is a drop-in for `LinearSVC` (`fit` / `decision_function`).
* **Post-processing:** tune offsets on OOF from models of the *same size* as the deployed ones (fold-averaged inference + offsets tuned on pooled OOF, as the dev evaluation here does). Offsets tuned on smaller inner models transfer poorly because margins grow with training size (§6.1, R0 raw 0.7651 → nested 0.7603).

## 9. Reproduce (Linux: hosted Colab or any Linux box; macOS is refused on purpose)

```bash
cd kaggle_ds102
python -m unittest discover -s loss_lab -p 'test_*.py' -v           # 20 checks, ~5 s
python loss_lab/pilot_grid.py      --data-dir DATA --out loss_lab/output
python loss_lab/run_ablation.py    --data-dir DATA --out loss_lab/output/ablation_seed102 --seed 102
python loss_lab/run_gbdt_ablation.py --data-dir DATA --out loss_lab/output/gbdt_seed102 --seed 102 \
       --configs G0_builtin_multiclass_sqrtbal G1_custom_ce_sqrtbal G2_custom_cb_focal G3_logit_adjusted \
                 G0LL_builtin_sqrtbal_loglossES G4_builtin_unweighted_loglossES
python loss_lab/run_catboost_la.py --data-dir DATA --out loss_lab/output/catboost_la_seed102
python loss_lab/analyze_strata.py  --data-dir DATA --run loss_lab/output/ablation_seed102
python loss_lab/explore_blends.py  --run loss_lab/output/ablation_seed102
```
`DATA` holds the four competition CSVs (their SHA-256 are checked into every `manifest.json`). Runtimes on a 4-vCPU Linux container: linear ablation ≈ 65 min (4.0 CPU-h), GBDT ablation ≈ 25 min + 15 min for the two robustness rows. Libraries: numpy 2.3, scipy 1.18, scikit-learn 1.7.2, LightGBM 4.7.0, CatBoost 1.2.10 (all open source). Runs resume from per-(config, fold) checkpoints; the manifest of the code that fitted the models is never overwritten by a later re-aggregation.

## 10. Limitations and honest caveats
* **Where this ran:** the fits were done in an Anthropic-hosted Linux container (4 vCPU), not on the project owner's Mac and not on Colab; the code runs unchanged on hosted Colab.
* **Scope:** the loss study covers the two model families the competitive pipeline actually uses (linear TF-IDF and GBDT on latent features). The PyTorch CNN branch was not re-run (no CPU PyTorch wheel was reachable here); the numpy losses give the exact gradients to wrap.
* **Selection:** candidates and hyper-parameters were fixed before outer-fold results; C grids came from an inner-only pilot. After seeing seed-102 results I added only (a) the cross-fit decision rule, (b) two GBDT robustness baselines and (c) the CatBoost transfer check — none of which can favour the candidate, and the decisive comparisons were re-run on fresh folds (seed 202).
* **Uncertainty:** fold std ≈ 0.02 and the cluster bootstrap conditions on fitted models; dev (17 fakes) is too small to rank rows. A standalone component's CV gain does not guarantee an ensemble or leaderboard gain — the blend weights/class multipliers of V11 were tuned for SqrtBalanced probabilities and would have to be re-validated on OOF before any Kaggle submission (none was made).

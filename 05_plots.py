"""
05_plots.py
===========
Genera las figuras para el paper (máx. 8 páginas → máx. 8 figuras).

Figuras originales:
  Fig 1 — Curva F1 vs k: eficiencia del SAE probe
  Fig 2 — Jaccard distribución nula vs observado (H1)
  Fig 3 — Tabla de resultados comparativa (H2)
  Fig 4 — Activación media de latentes compartidos

Figuras nuevas:
  Fig 5 — UMAP de latentes SAE coloreado por idioma × clase
  Fig 6 — Heatmap activación (top latentes × 4 condiciones)
  Fig 7 — Scatter coeficientes probe EN vs ES
  Fig 8 — Proyecciones sobre vector supervisado por idioma
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings

# Estilo global limpio para paper
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
})

COLORS = {
    "sae":        "#1D9E75",
    "supervised": "#3B6D11",
    "dense":      "#888780",
    "null":       "#B4B2A9",
    "obs":        "#D85A30",
    "en":         "#185FA5",
    "es":         "#993C1D",
    "toxic":      "#D85A30",
    "benign":     "#1D9E75",
}

# Paleta para UMAP: 4 grupos (idioma × clase)
UMAP_COLORS = {
    ("en", 1): "#185FA5",   # EN tóxico — azul oscuro
    ("en", 0): "#85B7EB",   # EN benign — azul claro
    ("es", 1): "#993C1D",   # ES tóxico — coral oscuro
    ("es", 0): "#F0997B",   # ES benign — coral claro
}
UMAP_MARKERS = {("en", 1): "o", ("en", 0): "o", ("es", 1): "s", ("es", 0): "s"}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURAS ORIGINALES
# ══════════════════════════════════════════════════════════════════════════════

def plot_f1_vs_k(f1_curve: list[dict], output_dir: str = "."):
    """Fig 1 — Curva de eficiencia: F1 macro vs número de latentes k."""
    ks  = [r["k"]        for r in f1_curve]
    f1s = [r["f1_macro"] for r in f1_curve]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ks, f1s, "o-", color=COLORS["sae"], linewidth=2,
            markersize=6, label="SAE probe (top-k latentes)")
    ax.axhline(y=max(f1s), linestyle="--", color=COLORS["sae"], alpha=0.4, linewidth=1)
    ax.set_xlabel("Número de latentes k")
    ax.set_ylabel("F1 macro (EN test)")
    ax.set_title("Fig. 1 — Eficiencia del SAE probe")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, output_dir, "fig1_f1_vs_k.pdf")
    return fig


def plot_jaccard_null(jaccard_results: dict, output_dir: str = "."):
    """Fig 2 — Histograma de distribución nula del Jaccard con valor observado."""
    j_null = jaccard_results["j_null"]
    j_obs  = jaccard_results["j_obs"]
    p_val  = jaccard_results["p_value"]
    k      = jaccard_results["k"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(j_null, bins=40, color=COLORS["null"], edgecolor="white",
            linewidth=0.5, label=f"Distribución nula ({len(j_null)} perm.)")
    ax.axvline(j_obs, color=COLORS["obs"], linewidth=2,
               label=f"J observado = {j_obs:.3f}")
    cutoff = np.percentile(j_null, 95)
    ax.axvspan(cutoff, j_null.max() + 0.05, alpha=0.15, color=COLORS["obs"],
               label=f"p < 0.05 (J > {cutoff:.3f})")
    sig = "✓ p < 0.05" if p_val < 0.05 else "✗ p ≥ 0.05"
    ax.set_xlabel(f"Jaccard (k={k})")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Fig. 2 — Test de permutación H₁  [{sig}, p={p_val:.3f}]")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, output_dir, "fig2_jaccard_null.pdf")
    return fig


def plot_results_table(results: dict, output_dir: str = "."):
    """Fig 3 — Heatmap de F1 macro por método y condición."""
    methods  = ["SAE probe", "Supervisado", "Denso (baseline)"]
    conds    = ["EN mono", "ES mono", "Transfer EN→ES"]
    keys_f1  = [
        ["sae_probe_en", "sae_probe_es", "sae_transfer"],
        ["Supervisado_en", "Supervisado_es", "sv_transfer"],
        ["Denso_en", "Denso_es", None],
    ]
    data = np.full((3, 3), np.nan)
    for i, method_keys in enumerate(keys_f1):
        for j, key in enumerate(method_keys):
            if key and key in results:
                data[i, j] = results[key]["f1_macro"]

    fig, ax = plt.subplots(figsize=(6, 2.8))
    im = ax.imshow(data, cmap="YlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(conds, fontsize=10)
    ax.set_yticks(range(3)); ax.set_yticklabels(methods, fontsize=10)
    for i in range(3):
        for j in range(3):
            if not np.isnan(data[i, j]):
                color = "white" if data[i, j] > 0.75 else "black"
                ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="gray")
    plt.colorbar(im, ax=ax, shrink=0.8, label="F1 macro")
    ax.set_title("Fig. 3 — F1 macro por método y condición")
    plt.tight_layout()
    _save(fig, output_dir, "fig3_results_table.pdf")
    return fig


def plot_shared_latents(shared_stats: list[dict], output_dir: str = "."):
    """Fig 4 — Barras de activación media de latentes compartidos."""
    if not shared_stats:
        print("Sin latentes compartidos — no se genera Fig. 4.")
        return None
    n      = min(10, len(shared_stats))
    stats  = shared_stats[:n]
    labels = [f"L{s['latent_idx']}" for s in stats]
    x, w   = np.arange(n), 0.2
    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), 4))
    ax.bar(x - 1.5*w, [s["en_toxic_mean"]  for s in stats], w,
           color=COLORS["en"], alpha=0.9, label="EN tóxico")
    ax.bar(x - 0.5*w, [s["en_benign_mean"] for s in stats], w,
           color=COLORS["en"], alpha=0.4, label="EN benign")
    ax.bar(x + 0.5*w, [s["es_toxic_mean"]  for s in stats], w,
           color=COLORS["es"], alpha=0.9, label="ES tóxico")
    ax.bar(x + 1.5*w, [s["es_benign_mean"] for s in stats], w,
           color=COLORS["es"], alpha=0.4, label="ES benign")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Activación media")
    ax.set_title("Fig. 4 — Activación de latentes compartidos por clase e idioma")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    _save(fig, output_dir, "fig4_shared_latents.pdf")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURAS NUEVAS
# ══════════════════════════════════════════════════════════════════════════════

# ── Fig 5: UMAP ────────────────────────────────────────────────────────────────

def plot_umap(reps: dict, output_dir: str = ".", n_neighbors: int = 15,
              min_dist: float = 0.1, max_points: int = 500, seed: int = 42):
    """
    Fig 5 — UMAP 2D de los latentes SAE, coloreado por idioma × clase.

    Cada punto = un texto. Forma = idioma (○ EN, □ ES).
    Color oscuro = tóxico, claro = benign.

    Interpretación:
      - Puntos EN-tóxico y ES-tóxico agrupados → features universales (H1 ✓)
      - Puntos separados por idioma → features idioma-específicas (H1 ✗)

    Parámetros
    ----------
    n_neighbors : int   — balance local/global en UMAP (default 15)
    min_dist    : float — compactness de los clusters (default 0.1)
    max_points  : int   — subsample por grupo para velocidad en Colab
    """
    try:
        import umap
    except ImportError:
        print("  UMAP no disponible. Instala con: pip install umap-learn")
        print("  Usando PCA como fallback para Fig 5.")
        return _plot_umap_pca_fallback(reps, output_dir, max_points, seed)

    print("\n── Fig 5: UMAP (puede tardar 1-2 min en Colab gratuito)")

    h_all, y_all, lang_all = _collect_reps(reps, "h", max_points, seed)

    # Entrenar UMAP sobre todos los puntos combinados
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=seed, metric="cosine")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb = reducer.fit_transform(h_all)   # (N, 2)

    fig = _render_2d_scatter(emb, y_all, lang_all,
                              title="Fig. 5 — UMAP de latentes SAE (cosine)",
                              xlabel="UMAP-1", ylabel="UMAP-2")
    _save(fig, output_dir, "fig5_umap.pdf")
    return fig


def _plot_umap_pca_fallback(reps: dict, output_dir: str, max_points: int, seed: int):
    """PCA 2D como fallback si umap-learn no está disponible."""
    from sklearn.decomposition import PCA
    print("  Usando PCA 2D como fallback...")

    h_all, y_all, lang_all = _collect_reps(reps, "h", max_points, seed)

    pca = PCA(n_components=2, random_state=seed)
    emb = pca.fit_transform(h_all)
    var = pca.explained_variance_ratio_

    fig = _render_2d_scatter(
        emb, y_all, lang_all,
        title="Fig. 5 — PCA de latentes SAE (fallback UMAP)",
        xlabel=f"PC1 ({var[0]*100:.1f}% var.)",
        ylabel=f"PC2 ({var[1]*100:.1f}% var.)",
    )
    _save(fig, output_dir, "fig5_umap_pca.pdf")
    return fig


def _collect_reps(reps: dict, key: str, max_points: int,
                  seed: int) -> tuple[np.ndarray, np.ndarray, list]:
    """Combina train+test de EN y ES con subsample."""
    rng = np.random.default_rng(seed)
    arrays, labels, langs = [], [], []

    for lang in ["en", "es"]:
        for split in ["train", "test"]:
            X = reps[lang][split][key]
            y = reps[lang][split]["label"]
            n = min(max_points // 4, len(y))        # max_points / 4 por grupo
            idx = rng.choice(len(y), n, replace=False)
            arrays.append(X[idx])
            labels.append(y[idx])
            langs.extend([lang] * n)

    return np.vstack(arrays), np.concatenate(labels), langs


def _render_2d_scatter(emb: np.ndarray, y: np.ndarray, langs: list,
                        title: str, xlabel: str, ylabel: str):
    """Dibuja el scatter 2D con la paleta idioma × clase."""
    fig, ax = plt.subplots(figsize=(6, 5))

    groups = [("en", 1), ("en", 0), ("es", 1), ("es", 0)]
    labels_map = {
        ("en", 1): "EN tóxico", ("en", 0): "EN benign",
        ("es", 1): "ES tóxico", ("es", 0): "ES benign",
    }
    y_arr    = np.array(y)
    lang_arr = np.array(langs)

    for (lang, cls) in groups:
        mask = (lang_arr == lang) & (y_arr == cls)
        if mask.sum() == 0:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=UMAP_COLORS[(lang, cls)],
                   marker=UMAP_MARKERS[(lang, cls)],
                   s=18, alpha=0.65, linewidths=0,
                   label=labels_map[(lang, cls)])

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9, markerscale=1.5,
              framealpha=0.9, edgecolor="lightgray")
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    return fig


# ── Fig 6: Heatmap activación top-latentes × condición ────────────────────────

def plot_activation_heatmap(reps: dict, top_k: int = 40, output_dir: str = "."):
    """
    Fig 6 — Heatmap (top_k latentes) × (4 condiciones).

    Columnas: EN-tóxico | EN-benign | ES-tóxico | ES-benign
    Filas: los top_k latentes con mayor varianza entre condiciones.

    Un latente universal mostrará columnas EN-tóxico y ES-tóxico similares
    (misma intensidad de color) y distintas de las columnas benign.
    Un latente idioma-específico será intenso solo en EN o solo en ES.

    También calcula la cosine similarity entre los vectores de activación
    media de cada condición y los anota en el título.
    """
    print("\n── Fig 6: Heatmap activación")

    # Calcular activación media por condición
    cond_means = {}
    for lang in ["en", "es"]:
        for split in ["train", "test"]:
            h = reps[lang][split]["h"]
            y = reps[lang][split]["label"]
            for cls, cls_name in [(1, "toxic"), (0, "benign")]:
                key = f"{lang}_{cls_name}"
                mask = (np.array(y) == cls)
                if mask.sum() > 0:
                    if key not in cond_means:
                        cond_means[key] = []
                    cond_means[key].append(h[mask].mean(axis=0))

    # Promediar si hay train+test
    for k in cond_means:
        cond_means[k] = np.mean(cond_means[k], axis=0)   # (d_sae,)

    cond_order = ["en_toxic", "en_benign", "es_toxic", "es_benign"]
    cond_labels = ["EN tóxico", "EN benign", "ES tóxico", "ES benign"]
    matrix = np.stack([cond_means[c] for c in cond_order], axis=1)  # (d_sae, 4)

    # Seleccionar top_k latentes con mayor varianza entre condiciones
    var   = matrix.var(axis=1)
    top_i = np.argsort(var)[::-1][:top_k]
    matrix_top = matrix[top_i]                 # (top_k, 4)

    # Cosine similarity entre EN-tóxico y ES-tóxico
    v_en = cond_means["en_toxic"]
    v_es = cond_means["es_toxic"]
    cos_tox = _cosine(v_en, v_es)

    # Cosine similarity entre EN-benign y ES-benign
    v_en_b = cond_means["en_benign"]
    v_es_b = cond_means["es_benign"]
    cos_ben = _cosine(v_en_b, v_es_b)

    print(f"  Cosine similarity EN-tóxico vs ES-tóxico:  {cos_tox:.4f}")
    print(f"  Cosine similarity EN-benign vs ES-benign:  {cos_ben:.4f}")
    print(f"  (Si cos_tox >> cos_ben → los latentes de toxicidad son más universales)")

    # Plot
    fig, ax = plt.subplots(figsize=(5, max(5, top_k * 0.18)))
    im = ax.imshow(matrix_top, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")

    ax.set_xticks(range(4))
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"L{top_i[i]}" for i in range(top_k)], fontsize=7)
    ax.set_ylabel(f"Top-{top_k} latentes (por varianza)")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Activación media")
    ax.set_title(
        f"Fig. 6 — Activación latentes × condición\n"
        f"cos(EN-tox, ES-tox)={cos_tox:.3f}  |  cos(EN-ben, ES-ben)={cos_ben:.3f}",
        fontsize=10,
    )
    plt.tight_layout()
    _save(fig, output_dir, "fig6_activation_heatmap.pdf")
    return fig, {"cos_toxic": cos_tox, "cos_benign": cos_ben}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity entre dos vectores."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ── Fig 7: Scatter coeficientes probe EN vs ES ────────────────────────────────

def plot_probe_coeff_scatter(reps: dict, C: float = 0.1, output_dir: str = "."):
    """
    Fig 7 — Scatter: coeficiente del probe logístico EN (eje X)
             vs coeficiente del probe logístico ES (eje Y).

    Cada punto = un latente SAE.
    - Diagonal (cuadrante I y III): latentes universales con mismo signo en ambos idiomas.
    - Eje X (y≈0): latentes específicos del inglés.
    - Eje Y (x≈0): latentes específicos del español.

    Los latentes en la diagonal son los candidatos a features cross-lingüe.
    """
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print("\n── Fig 7: Scatter coeficientes probe EN vs ES")

    def _fit_probe(h, y):
        scaler = StandardScaler()
        h_s    = scaler.fit_transform(h)
        clf    = LogisticRegression(l1_ratio=1.0, C=C, solver="saga",
                                    max_iter=2000, random_state=42,
                                    class_weight="balanced")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(h_s, y)
        return clf.coef_[0]

    coef_en = _fit_probe(reps["en"]["train"]["h"], reps["en"]["train"]["label"])
    coef_es = _fit_probe(reps["es"]["train"]["h"], reps["es"]["train"]["label"])

    # Alineación de signos: algunos latentes pueden estar "invertidos"
    # Si un latente tiene coef_en > 0 y coef_es > 0 → mismo concepto.
    # Si coef_en > 0 y coef_es < 0 → posiblemente invertido en un idioma.

    # Identificar latentes no-cero en al menos un idioma
    active = (coef_en != 0) | (coef_es != 0)
    n_active = active.sum()
    print(f"  Latentes activos (≥1 idioma): {n_active}")

    # Calcular cosine similarity entre vectores de coeficientes
    cos_coef = _cosine(coef_en, coef_es)
    print(f"  Cosine similarity coef_EN vs coef_ES: {cos_coef:.4f}")

    # Latentes más extremos en la diagonal (universales candidatos)
    diagonal_score = np.abs(coef_en) + np.abs(coef_es)
    top_diag       = np.argsort(diagonal_score)[::-1][:5]

    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Todos los latentes en gris
    ax.scatter(coef_en, coef_es, c="lightgray", s=6, alpha=0.5,
               linewidths=0, zorder=1, label=f"Todos ({len(coef_en)})")

    # Latentes activos destacados
    ax.scatter(coef_en[active], coef_es[active],
               c=COLORS["sae"], s=20, alpha=0.7, linewidths=0, zorder=2,
               label=f"Activos en ≥1 idioma ({n_active})")

    # Top-5 universales etiquetados
    for i in top_diag:
        if coef_en[i] != 0 or coef_es[i] != 0:
            ax.scatter(coef_en[i], coef_es[i], c=COLORS["obs"],
                       s=50, zorder=3, linewidths=0)
            ax.annotate(f"L{i}", (coef_en[i], coef_es[i]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points",
                        color=COLORS["obs"])

    # Líneas de referencia
    lim = max(np.abs(coef_en).max(), np.abs(coef_es).max()) * 1.1
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.plot([-lim, lim], [-lim, lim], "--", color="gray",
            linewidth=0.8, alpha=0.4, label="diagonal y=x")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    # Anotar cuadrantes
    offset = lim * 0.85
    ax.text( offset,  offset, "Universales\n(ambos +)", fontsize=8,
             ha="right", va="top", color="gray")
    ax.text(-offset, -offset, "Universales\n(ambos –)", fontsize=8,
             ha="left",  va="bottom", color="gray")
    ax.text( offset, -offset, "Específico EN", fontsize=8,
             ha="right", va="bottom", color="gray")
    ax.text(-offset,  offset, "Específico ES", fontsize=8,
             ha="left",  va="top", color="gray")

    ax.set_xlabel("Coeficiente probe EN")
    ax.set_ylabel("Coeficiente probe ES")
    ax.set_title(
        f"Fig. 7 — Coeficientes probe por latente\n"
        f"cos(coef_EN, coef_ES) = {cos_coef:.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, output_dir, "fig7_coeff_scatter.pdf")
    return fig, {"cos_coeff": cos_coef, "n_active": int(n_active)}


# ── Fig 8: Distribuciones de proyección sobre vector supervisado ──────────────

def plot_supervised_projections(reps: dict, classifiers: dict,
                                 output_dir: str = "."):
    """
    Fig 8 — Distribuciones KDE de proyección sobre el vector supervisado
             (difference-in-means) para EN y ES, separadas por clase.

    El vector supervisado fue entrenado en EN. Esta figura muestra:
      - Si las distribuciones ES se solapan o separan igual que EN → buen transfer.
      - Si las distribuciones ES están desplazadas respecto a EN → artefacto léxico:
        el vector aprendió palabras inglesas, no el concepto de toxicidad.

    Esto explica cualitativamente por qué el transfer falla (o no).
    """
    from scipy.stats import gaussian_kde

    print("\n── Fig 8: Proyecciones sobre vector supervisado")

    sv = classifiers["supervised"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

    for ax, (lang, title) in zip(axes, [("en", "EN (train domain)"),
                                         ("es", "ES (transfer domain)")]):
        # Proyectar train+test combinados
        z_parts, y_parts = [], []
        for split in ["train", "test"]:
            z_parts.append(reps[lang][split]["z"])
            y_parts.append(reps[lang][split]["label"])
        z_all = np.vstack(z_parts)
        y_all = np.concatenate(y_parts)

        proj = sv._project(z_all)   # escalar por texto

        # KDE por clase
        for cls, label, color, ls in [
            (1, "Tóxico",  COLORS["toxic"],  "-"),
            (0, "Benign",  COLORS["benign"], "--"),
        ]:
            mask = (y_all == cls)
            if mask.sum() < 5:
                continue
            vals = proj[mask]
            kde  = gaussian_kde(vals, bw_method=0.3)
            xs   = np.linspace(proj.min() - 0.5, proj.max() + 0.5, 300)
            ax.plot(xs, kde(xs), color=color, linestyle=ls,
                    linewidth=2, label=label)
            ax.fill_between(xs, kde(xs), alpha=0.1, color=color)

        # Umbral de decisión
        ax.axvline(sv.threshold, color="black", linewidth=1.2,
                   linestyle=":", label=f"Umbral ({sv.threshold:.2f})")

        ax.set_xlabel("Proyección sobre dirección supervisada")
        ax.set_ylabel("Densidad")
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.suptitle("Fig. 8 — Distribución de proyecciones por idioma y clase",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    _save(fig, output_dir, "fig8_supervised_projections.pdf")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# WRAPPER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_figures(results: dict, config: dict,
                          reps: dict = None, classifiers: dict = None):
    """
    Genera todas las figuras del paper.

    Parámetros
    ----------
    results     : dict  — salida de run_full_analysis()
    config      : dict  — CONFIG global
    reps        : dict  — representaciones (necesario para Fig 5, 6, 7, 8)
    classifiers : dict  — clasificadores entrenados (necesario para Fig 8)
    """
    out = config.get("output_dir", "./resultados")
    os.makedirs(out, exist_ok=True)

    cosine_results = {}

    print("\n" + "="*50)
    print("GENERANDO FIGURAS")
    print("="*50)

    # ── Figuras originales ─────────────────────────────────────────────────────
    if "f1_vs_k" in results:
        plot_f1_vs_k(results["f1_vs_k"], out)

    if "jaccard" in results:
        plot_jaccard_null(results["jaccard"], out)

    if results:
        plot_results_table(results, out)

    if "shared_stats" in results and results["shared_stats"]:
        plot_shared_latents(results["shared_stats"], out)

    # ── Figuras nuevas (requieren reps) ───────────────────────────────────────
    if reps is None:
        print("\nreps=None → saltando figuras 5-8 (UMAP, heatmap, scatter, proyecciones).")
        print("Llama a generate_all_figures(results, config, reps=reps, classifiers=clfs)")
        return cosine_results

    print("\n── Figuras nuevas (5-8)")

    # Fig 5 — UMAP / PCA
    plot_umap(reps, out,
              max_points=config.get("umap_max_points", 500),
              seed=config.get("random_seed", 42))

    # Fig 6 — Heatmap + cosine similarity
    _, cos_res = plot_activation_heatmap(reps, top_k=40, output_dir=out)
    cosine_results.update(cos_res)

    # Fig 7 — Scatter coeficientes
    _, coeff_res = plot_probe_coeff_scatter(
        reps, C=config.get("probe_C", 0.1), output_dir=out
    )
    cosine_results.update(coeff_res)

    # Fig 8 — Proyecciones supervisadas
    if classifiers is not None:
        plot_supervised_projections(reps, classifiers, out)
    else:
        print("classifiers=None → saltando Fig 8.")

    # Guardar métricas de cosine en results
    if cosine_results:
        import json
        path = os.path.join(out, "cosine_similarity.json")
        with open(path, "w") as f:
            json.dump(cosine_results, f, indent=2)
        print(f"\nMétricas cosine guardadas en: {path}")
        print(f"  cos(EN-tox, ES-tox)  = {cosine_results.get('cos_toxic', '—'):.4f}")
        print(f"  cos(EN-ben, ES-ben)  = {cosine_results.get('cos_benign', '—'):.4f}")
        print(f"  cos(coef_EN, coef_ES)= {cosine_results.get('cos_coeff', '—'):.4f}")

    return cosine_results


# ── Utilidad interna ──────────────────────────────────────────────────────────

def _save(fig, output_dir: str, filename: str):
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"  Guardado: {path}")
    plt.show()


# ── Test rápido ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N, d_z, d_h = 400, 512, 256

    # Simular features universales en latentes 10, 42, 77
    toxic_mask = np.zeros(d_h); toxic_mask[[10, 42, 77]] = 2.5
    noise_en   = np.zeros(d_h); noise_en[[5, 20]]        = 1.0   # artefactos EN
    noise_es   = np.zeros(d_h); noise_es[[100, 150]]     = 1.0   # artefactos ES

    def make_data(lang_noise, seed):
        rg = np.random.default_rng(seed)
        y  = rg.integers(0, 2, N)
        z  = rg.standard_normal((N, d_z)) + np.outer(y, np.ones(d_z) * 0.5)
        h  = np.maximum(
            rg.standard_normal((N, d_h)) * 0.2
            + np.outer(y, toxic_mask)
            + np.outer(np.ones(N), lang_noise * 0.3),
            0
        )
        half = N // 2
        return {
            "train": {"h": h[:half],   "z": z[:half],   "label": y[:half]},
            "test":  {"h": h[half:],   "z": z[half:],   "label": y[half:]},
        }

    reps_mock = {"en": make_data(noise_en, 42), "es": make_data(noise_es, 99)}

    # Entrenar supervisado para Fig 8
    import sys; sys.path.insert(0, ".")
    from importlib import import_module
    clf_mod = import_module("03_classifiers")
    sv = clf_mod.SupervisedVector()
    sv.fit(reps_mock["en"]["train"]["z"], reps_mock["en"]["train"]["label"])
    probe = clf_mod.SAEProbe(C=0.1)
    probe.fit(reps_mock["en"]["train"]["h"], reps_mock["en"]["train"]["label"])
    clfs_mock = {"supervised": sv, "sae_probe": probe}

    os.makedirs("./test_output", exist_ok=True)
    _, cos = plot_activation_heatmap(reps_mock, top_k=20, output_dir="./test_output")
    print(f"Cosine toxic: {cos['cos_toxic']:.3f} | benign: {cos['cos_benign']:.3f}")

    _, sc = plot_probe_coeff_scatter(reps_mock, output_dir="./test_output")
    print(f"Cosine coeff: {sc['cos_coeff']:.3f}")

    plot_umap(reps_mock, output_dir="./test_output", max_points=200)
    plot_supervised_projections(reps_mock, clfs_mock, output_dir="./test_output")
    print("\nTest completo OK.")

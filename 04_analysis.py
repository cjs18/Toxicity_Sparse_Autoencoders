"""
04_analysis.py
==============
Todas las métricas del marco de hipótesis:

  H1 — Universalidad cross-lingüe:
    - Jaccard de top-k latentes (EN vs ES)
    - Test de permutación para p-valor
    - Activación media por clase de latentes compartidos

  H2 — SAE probe vs. vector supervisado:
    - F1 en 4 condiciones (EN/ES × train/transfer)
    - Test de McNemar para significancia estadística
    - Curva F1 vs k (eficiencia del probe)
"""

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import f1_score
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# H1 — Jaccard y test de permutación
# ══════════════════════════════════════════════════════════════════════════════

def get_top_k_by_activation(h: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """
    Selecciona los top-k latentes por activación diferencial:
        Δfreq_i = P(h_i > 0 | tóxico) − P(h_i > 0 | benign)

    Este método no depende del probe — mide directamente qué latentes
    el SAE activa más para textos tóxicos.
    """
    y = np.array(y)
    h_toxic  = h[y == 1]
    h_benign = h[y == 0]

    freq_toxic  = (h_toxic  > 0).mean(axis=0)   # (d_sae,)
    freq_benign = (h_benign > 0).mean(axis=0)

    delta = freq_toxic - freq_benign              # positivo → más activo en tóxicos
    top_k = np.argsort(delta)[::-1][:k]          # los k con mayor Δfreq
    return top_k


def jaccard(set_a: np.ndarray, set_b: np.ndarray) -> float:
    """Jaccard similarity entre dos conjuntos de índices."""
    a, b = set(set_a), set(set_b)
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def jaccard_permutation_test(
    h_en: np.ndarray, y_en: np.ndarray,
    h_es: np.ndarray, y_es: np.ndarray,
    k: int,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Test de permutación para H1.

    H0: El Jaccard observado no es mayor que lo esperado por azar.

    Procedimiento:
      1. Calcular J_obs con las etiquetas reales.
      2. Para cada permutación: mezclar etiquetas en EN y ES → calcular J.
      3. p-valor = fracción de J_perm >= J_obs.
    """
    rng = np.random.default_rng(seed)

    # J observado
    top_en = get_top_k_by_activation(h_en, y_en, k)
    top_es = get_top_k_by_activation(h_es, y_es, k)
    j_obs  = jaccard(top_en, top_es)

    # Distribución nula
    j_null = []
    for _ in range(n_permutations):
        y_en_perm = rng.permutation(y_en)
        y_es_perm = rng.permutation(y_es)
        top_en_p  = get_top_k_by_activation(h_en, y_en_perm, k)
        top_es_p  = get_top_k_by_activation(h_es, y_es_perm, k)
        j_null.append(jaccard(top_en_p, top_es_p))

    j_null  = np.array(j_null)
    p_value = (j_null >= j_obs).mean()

    # Latentes compartidos
    shared = np.intersect1d(top_en, top_es)

    print(f"\n── Jaccard (k={k})")
    print(f"  J_obs  = {j_obs:.4f}")
    print(f"  J_null = {j_null.mean():.4f} ± {j_null.std():.4f}")
    print(f"  p-valor = {p_value:.4f} {'✓ Significativo' if p_value < 0.05 else '✗ No significativo'}")
    print(f"  Latentes compartidos ({len(shared)}): {shared[:10]}{'...' if len(shared)>10 else ''}")

    return {
        "j_obs":    j_obs,
        "j_null":   j_null,
        "p_value":  p_value,
        "top_en":   top_en,
        "top_es":   top_es,
        "shared":   shared,
        "k":        k,
    }


def analyze_shared_latents(
    shared_idx: np.ndarray,
    h_en: np.ndarray, y_en: np.ndarray,
    h_es: np.ndarray, y_es: np.ndarray,
    top_n: int = 10,
) -> list[dict]:
    """
    Para cada latente compartido, calcula la activación media por clase
    en ambos idiomas. Permite inspección cualitativa.
    """
    results = []
    for idx in shared_idx[:top_n]:
        # EN
        act_en_tox = h_en[y_en == 1, idx].mean()
        act_en_ben = h_en[y_en == 0, idx].mean()
        # ES
        act_es_tox = h_es[y_es == 1, idx].mean()
        act_es_ben = h_es[y_es == 0, idx].mean()

        results.append({
            "latent_idx":    int(idx),
            "en_toxic_mean": float(act_en_tox),
            "en_benign_mean":float(act_en_ben),
            "en_delta":      float(act_en_tox - act_en_ben),
            "es_toxic_mean": float(act_es_tox),
            "es_benign_mean":float(act_es_ben),
            "es_delta":      float(act_es_tox - act_es_ben),
        })

    results.sort(key=lambda x: x["en_delta"] + x["es_delta"], reverse=True)
    return results


def print_shared_latent_table(shared_stats: list[dict]):
    """Imprime tabla de latentes compartidos."""
    print(f"\n{'Idx':>6} {'EN Δ':>8} {'ES Δ':>8}  {'EN (tox/ben)':>16}  {'ES (tox/ben)':>16}")
    print("─" * 64)
    for r in shared_stats:
        print(f"{r['latent_idx']:>6} {r['en_delta']:>8.4f} {r['es_delta']:>8.4f}  "
              f"{r['en_toxic_mean']:>6.4f}/{r['en_benign_mean']:<6.4f}  "
              f"{r['es_toxic_mean']:>6.4f}/{r['es_benign_mean']:<6.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# H2 — Test de McNemar
# ══════════════════════════════════════════════════════════════════════════════

def mcnemar_test(y_true: np.ndarray,
                 y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray) -> dict:
    """
    Test de McNemar para comparar dos clasificadores sobre el mismo test set.

    Tabla de contingencia:
                   B correcto   B incorrecto
    A correcto       n00            n01
    A incorrecto     n10            n11

    El test solo mira los casos discordantes (n01 y n10).
    H0: los dos clasificadores cometen los mismos errores (n01 = n10).
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    n00 = ((correct_a)  & (correct_b)).sum()   # ambos correctos
    n01 = ((correct_a)  & (~correct_b)).sum()  # solo A correcto
    n10 = ((~correct_a) & (correct_b)).sum()   # solo B correcto
    n11 = ((~correct_a) & (~correct_b)).sum()  # ambos incorrectos

    # McNemar exacto: stat = (|n01 - n10| - 1)^2 / (n01 + n10)
    # Con corrección de continuidad de Edwards
    discordant = n01 + n10
    if discordant == 0:
        p_value = 1.0
        stat    = 0.0
    else:
        stat    = (abs(n01 - n10) - 1) ** 2 / discordant
        from scipy.stats import chi2
        p_value = chi2.sf(stat, df=1)

    print(f"\n── Test de McNemar")
    print(f"  Tabla: n00={n00} n01={n01} n10={n10} n11={n11}")
    print(f"  stat = {stat:.4f}, p-valor = {p_value:.4f}")
    print(f"  {'✓ Diferencia significativa (p<0.05)' if p_value < 0.05 else '✗ Sin diferencia significativa'}")

    return {"stat": stat, "p_value": p_value,
            "n00": n00, "n01": n01, "n10": n10, "n11": n11}


# ══════════════════════════════════════════════════════════════════════════════
# Curva F1 vs k (eficiencia del probe)
# ══════════════════════════════════════════════════════════════════════════════

def f1_vs_k_curve(
    h_train: np.ndarray, y_train: np.ndarray,
    h_test:  np.ndarray, y_test:  np.ndarray,
    k_values: list[int],
    seed: int = 42,
) -> list[dict]:
    """
    Para cada k, entrena un probe logístico usando solo los top-k latentes
    (por activación diferencial sobre train) y evalúa en test.

    Permite responder: ¿cuántos latentes necesita el SAE para saturar el F1?
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    results = []
    for k in k_values:
        # Seleccionar top-k latentes
        top_k = get_top_k_by_activation(h_train, y_train, k)

        h_tr_k = h_train[:, top_k]
        h_te_k = h_test[:, top_k]

        # Probe logístico denso (sin L1 — ya estamos usando k latentes)
        scaler = StandardScaler()
        clf    = LogisticRegression(C=1.0, max_iter=500, random_state=seed,
                                    class_weight="balanced", solver="lbfgs")
        clf.fit(scaler.fit_transform(h_tr_k), y_train)
        y_pred = clf.predict(scaler.transform(h_te_k))
        f1     = f1_score(y_test, y_pred, average="macro")

        results.append({"k": k, "f1_macro": f1})
        print(f"  k={k:>4}  F1 macro = {f1:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Análisis completo
# ══════════════════════════════════════════════════════════════════════════════

def run_full_analysis(reps: dict, classifiers: dict, config: dict) -> dict:
    """
    Ejecuta todo el análisis de hipótesis y devuelve un dict con los resultados.
    """
    probe = classifiers["sae_probe"]
    sv    = classifiers["supervised"]
    dense = classifiers["dense"]

    results = {}

    # ── 1. Performance monolingüe ──────────────────────────────────────────────
    print("\n" + "="*50)
    print("1. PERFORMANCE MONOLINGÜE")
    print("="*50)

    for lang, arm_label, clf, rep_key in [
        ("en", "SAE probe",   probe, "h"),
        ("en", "Supervisado", sv,    "z"),
        ("en", "Denso",       dense, "z"),
        ("es", "SAE probe",   probe, "h"),
        ("es", "Supervisado", sv,    "z"),
        ("es", "Denso",       dense, "z"),
    ]:
        # Nota: para ES, el probe no se reentrenó — evaluamos directamente
        split = reps[lang]["test"]
        X     = split[rep_key]
        y     = split["label"]
        key   = f"{arm_label}_{lang}"
        print(f"\n  {arm_label} — {lang.upper()} test:")
        results[key] = clf.evaluate(X, y)

    # ── 2. Transfer EN → ES (H2 principal) ────────────────────────────────────
    print("\n" + "="*50)
    print("2. TRANSFER EN → ES")
    print("="*50)

    es_test = reps["es"]["test"]

    print("\n  SAE probe (entrenado EN, evaluado ES):")
    results["sae_transfer"] = probe.evaluate(es_test["h"], es_test["label"])

    print("\n  Supervisado (entrenado EN, evaluado ES):")
    results["sv_transfer"] = sv.evaluate(es_test["z"], es_test["label"])

    # ── 3. Test de McNemar (H2) ────────────────────────────────────────────────
    print("\n" + "="*50)
    print("3. TEST DE MCNEMAR (transfer EN→ES)")
    print("="*50)

    y_pred_probe = probe.predict(es_test["h"])
    y_pred_sv    = sv.predict(es_test["z"])
    y_true_es    = es_test["label"]

    results["mcnemar"] = mcnemar_test(y_true_es, y_pred_probe, y_pred_sv)

    # ── 4. Jaccard + permutación (H1) ─────────────────────────────────────────
    print("\n" + "="*50)
    print("4. JACCARD Y TEST DE PERMUTACIÓN (H1)")
    print("="*50)

    k = config["top_k_main"]
    results["jaccard"] = jaccard_permutation_test(
        reps["en"]["train"]["h"], reps["en"]["train"]["label"],
        reps["es"]["train"]["h"], reps["es"]["train"]["label"],
        k=k,
        n_permutations=config["n_permutations"],
    )

    # ── 5. Análisis de latentes compartidos ───────────────────────────────────
    print("\n" + "="*50)
    print("5. LATENTES COMPARTIDOS — activación por clase")
    print("="*50)

    shared = results["jaccard"]["shared"]
    if len(shared) > 0:
        shared_stats = analyze_shared_latents(
            shared,
            reps["en"]["train"]["h"], reps["en"]["train"]["label"],
            reps["es"]["train"]["h"], reps["es"]["train"]["label"],
        )
        print_shared_latent_table(shared_stats)
        results["shared_stats"] = shared_stats
    else:
        print("  Sin latentes compartidos — H0 no rechazada.")
        results["shared_stats"] = []

    # ── 6. Curva F1 vs k ──────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("6. CURVA F1 vs K (eficiencia del probe)")
    print("="*50)

    results["f1_vs_k"] = f1_vs_k_curve(
        reps["en"]["train"]["h"], reps["en"]["train"]["label"],
        reps["en"]["test"]["h"],  reps["en"]["test"]["label"],
        k_values=config["top_k_values"],
    )

    return results


# ── Test rápido ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N, d_h = 300, 512
    toxic_latents = [10, 42, 77, 200, 350]

    toxic_mask = np.zeros(d_h)
    toxic_mask[toxic_latents] = 1.0

    y  = rng.integers(0, 2, N)
    h  = np.maximum(rng.standard_normal((N, d_h)) * 0.1
                    + np.outer(y, toxic_mask * 2.0), 0)

    # Simular ES con mismos latentes + ruido
    y_es = rng.integers(0, 2, N)
    h_es = np.maximum(rng.standard_normal((N, d_h)) * 0.2
                      + np.outer(y_es, toxic_mask * 1.5), 0)

    print("── Test Jaccard")
    res = jaccard_permutation_test(h, y, h_es, y_es, k=10, n_permutations=200)

    print("\n── Test McNemar")
    y_pred_a = rng.integers(0, 2, 60)
    y_pred_b = rng.integers(0, 2, 60)
    y_true   = rng.integers(0, 2, 60)
    mcnemar_test(y_true, y_pred_a, y_pred_b)

    print("\n── Curva F1 vs k")
    h_tr, h_te = h[:240], h[240:]
    y_tr, y_te = y[:240], y[240:]
    f1_vs_k_curve(h_tr, y_tr, h_te, y_te, k_values=[5, 10, 25, 50])

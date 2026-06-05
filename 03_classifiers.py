"""
03_classifiers.py
=================
Implementa los dos brazos del experimento:

  Brazo A — SAE probe:
    Clasificador logístico Lasso entrenado sobre los latentes h del SAE.
    La penalización L1 fuerza sparsity: solo usa un subconjunto de latentes.

  Brazo B — Vector supervisado (difference-in-means):
    Calcula la dirección media entre clases en el espacio z del LLM.
    Proyecta todos los textos sobre esa dirección → escalar → umbral.

  Baseline — Probe denso:
    Clasificador logístico Ridge sobre z completo.
    Mide cuánta información tiene el LLM antes de pasar por el SAE.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════════════════════
# BRAZO A — SAE Probe (logístico Lasso sobre latentes sparse)
# ══════════════════════════════════════════════════════════════════════════════

class SAEProbe:
    """
    Clasificador logístico con penalización L1 sobre latentes SAE.

    La penalización L1 (Lasso) fuerza que solo un subconjunto de los
    d_sae latentes tenga coeficiente distinto de cero — análogo a
    'top-k latentes' pero aprendido automáticamente.
    """

    def __init__(self, C: float = 0.1, max_iter: int = 1000, seed: int = 42):
        """
        C: inverso de la regularización L1. Menor C → más sparse.
           Barrer C ∈ {0.01, 0.05, 0.1, 0.5} para validar.
        """
        self.C        = C
        self.max_iter = max_iter
        self.seed     = seed
        self.clf      = None
        self.scaler   = StandardScaler()

    def fit(self, h_train: np.ndarray, y_train: np.ndarray):
        """Entrena el probe sobre los latentes SAE."""
        # Escalar (aunque h ya es sparse, normalizar mejora la convergencia)
        h_scaled = self.scaler.fit_transform(h_train)
        self.clf = LogisticRegression(
            l1_ratio=1.0,             # L1 puro (equivale a penalty="l1")
            C=self.C,
            solver="saga",            # saga soporta l1_ratio en ElasticNet
            max_iter=self.max_iter,
            random_state=self.seed,
            class_weight="balanced",  # importante: datos pueden estar desbalanceados
        )
        self.clf.fit(h_scaled, y_train)
        return self

    def predict(self, h: np.ndarray) -> np.ndarray:
        h_scaled = self.scaler.transform(h)
        return self.clf.predict(h_scaled)

    def predict_proba(self, h: np.ndarray) -> np.ndarray:
        h_scaled = self.scaler.transform(h)
        return self.clf.predict_proba(h_scaled)

    def get_active_latents(self) -> np.ndarray:
        """
        Devuelve los índices de los latentes con coeficiente != 0.
        Estos son los latentes que el probe considera relevantes para toxicidad.
        """
        coefs = self.clf.coef_[0]
        return np.where(coefs != 0)[0]

    def get_top_k_latents(self, k: int) -> np.ndarray:
        """
        Devuelve los k latentes con mayor |coeficiente|,
        ordenados de mayor a menor importancia.
        """
        coefs    = self.clf.coef_[0]
        top_idx  = np.argsort(np.abs(coefs))[::-1][:k]
        return top_idx

    def n_active_latents(self) -> int:
        return len(self.get_active_latents())

    def evaluate(self, h_test: np.ndarray, y_test: np.ndarray,
                 verbose: bool = True) -> dict:
        y_pred = self.predict(h_test)
        f1     = f1_score(y_test, y_pred, average="macro")
        f1_tox = f1_score(y_test, y_pred, pos_label=1, average="binary")
        if verbose:
            print(f"  F1 macro:   {f1:.4f}")
            print(f"  F1 tóxico:  {f1_tox:.4f}")
            print(f"  Latentes activos: {self.n_active_latents()}")
        return {"f1_macro": f1, "f1_toxic": f1_tox,
                "n_active": self.n_active_latents()}


# ══════════════════════════════════════════════════════════════════════════════
# BRAZO B — Vector supervisado (difference-in-means)
# ══════════════════════════════════════════════════════════════════════════════

class SupervisedVector:
    """
    Difference-in-means: dirección media entre clases en el espacio LLM.

    direction = mean(z | tóxico) − mean(z | benign)

    Clasificación: proyectar z sobre direction → escalar → umbral óptimo.
    Es el baseline supervisado de referencia del survey (Apéndice F).
    """

    def __init__(self):
        self.direction = None   # vector Δz
        self.threshold = None   # umbral de decisión
        self.scaler    = StandardScaler()

    def fit(self, z_train: np.ndarray, y_train: np.ndarray):
        """Calcula la dirección media y el umbral óptimo sobre train."""
        y_train = np.array(y_train)
        z_scaled = self.scaler.fit_transform(z_train)

        z_toxic  = z_scaled[y_train == 1]
        z_benign = z_scaled[y_train == 0]

        # Dirección: diferencia de medias, normalizada a longitud 1
        self.direction = z_toxic.mean(axis=0) - z_benign.mean(axis=0)
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction /= norm

        # Proyecciones sobre la dirección
        projections = z_scaled @ self.direction

        # Umbral óptimo: maximizar F1 sobre train (barrido simple)
        self.threshold = self._find_threshold(projections, y_train)
        return self

    def _find_threshold(self, projections: np.ndarray,
                        y_true: np.ndarray) -> float:
        """Busca el umbral que maximiza F1 macro sobre las proyecciones."""
        best_f1, best_thr = 0.0, 0.0
        for thr in np.linspace(projections.min(), projections.max(), 100):
            preds = (projections >= thr).astype(int)
            f1    = f1_score(y_true, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        return best_thr

    def _project(self, z: np.ndarray) -> np.ndarray:
        z_scaled = self.scaler.transform(z)
        return z_scaled @ self.direction

    def predict(self, z: np.ndarray) -> np.ndarray:
        return (self._project(z) >= self.threshold).astype(int)

    def predict_proba(self, z: np.ndarray) -> np.ndarray:
        """Pseudo-probabilidad via sigmoide sobre la proyección."""
        proj = self._project(z)
        prob = 1 / (1 + np.exp(-(proj - self.threshold)))
        return np.column_stack([1 - prob, prob])

    def evaluate(self, z_test: np.ndarray, y_test: np.ndarray,
                 verbose: bool = True) -> dict:
        y_pred = self.predict(z_test)
        f1     = f1_score(y_test, y_pred, average="macro")
        f1_tox = f1_score(y_test, y_pred, pos_label=1, average="binary")
        if verbose:
            print(f"  F1 macro:   {f1:.4f}")
            print(f"  F1 tóxico:  {f1_tox:.4f}")
        return {"f1_macro": f1, "f1_toxic": f1_tox}


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE — Probe denso sobre activaciones brutas
# ══════════════════════════════════════════════════════════════════════════════

class DenseProbe:
    """
    Logístico Ridge sobre z completo. Mide el techo de performance
    de las representaciones del LLM antes del SAE.
    """

    def __init__(self, C: float = 1.0, seed: int = 42):
        self.scaler = StandardScaler()
        self.clf    = LogisticRegression(
            l1_ratio=0.0, C=C, solver="saga",
            max_iter=1000, random_state=seed,
            class_weight="balanced",
        )

    def fit(self, z_train: np.ndarray, y_train: np.ndarray):
        z_scaled = self.scaler.fit_transform(z_train)
        self.clf.fit(z_scaled, y_train)
        return self

    def predict(self, z: np.ndarray) -> np.ndarray:
        return self.clf.predict(self.scaler.transform(z))

    def evaluate(self, z_test: np.ndarray, y_test: np.ndarray,
                 verbose: bool = True) -> dict:
        y_pred = self.predict(z_test)
        f1     = f1_score(y_test, y_pred, average="macro")
        f1_tox = f1_score(y_test, y_pred, pos_label=1, average="binary")
        if verbose:
            print(f"  F1 macro:   {f1:.4f}")
            print(f"  F1 tóxico:  {f1_tox:.4f}")
        return {"f1_macro": f1, "f1_toxic": f1_tox}


# ══════════════════════════════════════════════════════════════════════════════
# Entrenamiento del experimento completo
# ══════════════════════════════════════════════════════════════════════════════

def train_all_classifiers(reps: dict, config: dict) -> dict:
    """
    Entrena los tres clasificadores sobre EN y los almacena.

    Devuelve:
        {
          "sae_probe":  SAEProbe  (entrenado en EN train),
          "supervised": SupervisedVector (entrenado en EN train),
          "dense":      DenseProbe (entrenado en EN train),
        }
    """
    print("\n" + "="*50)
    print("ENTRENANDO CLASIFICADORES (sobre EN train)")
    print("="*50)

    en_train = reps["en"]["train"]

    # Brazo A
    print("\n── Brazo A: SAE Probe (Lasso)")
    probe = SAEProbe(C=config.get("probe_C", 0.1))
    probe.fit(en_train["h"], en_train["label"])
    probe.evaluate(reps["en"]["test"]["h"], reps["en"]["test"]["label"])

    # Brazo B
    print("\n── Brazo B: Vector supervisado (diff-in-means)")
    sv = SupervisedVector()
    sv.fit(en_train["z"], en_train["label"])
    sv.evaluate(reps["en"]["test"]["z"], reps["en"]["test"]["label"])

    # Baseline
    print("\n── Baseline: Probe denso sobre z")
    dense = DenseProbe()
    dense.fit(en_train["z"], en_train["label"])
    dense.evaluate(reps["en"]["test"]["z"], reps["en"]["test"]["label"])

    return {"sae_probe": probe, "supervised": sv, "dense": dense}


# ── Test rápido ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test con datos sintéticos
    rng = np.random.default_rng(42)
    N, d_z, d_h = 200, 512, 4096

    # Simular que la toxicidad eleva ciertos latentes
    toxic_mask = np.zeros(d_h)
    toxic_mask[[10, 42, 77, 200, 350]] = 1.0

    y     = rng.integers(0, 2, N)
    z     = rng.standard_normal((N, d_z)) + np.outer(y, np.ones(d_z) * 0.5)
    h     = np.maximum(rng.standard_normal((N, d_h)) * 0.1
                       + np.outer(y, toxic_mask), 0)

    # Split
    from sklearn.model_selection import train_test_split
    z_tr, z_te, h_tr, h_te, y_tr, y_te = train_test_split(
        z, h, y, test_size=0.2, random_state=42
    )

    print("── SAE Probe")
    probe = SAEProbe(C=0.1)
    probe.fit(h_tr, y_tr)
    res = probe.evaluate(h_te, y_te)
    print(f"  Latentes activos: {probe.n_active_latents()}")
    print(f"  Top-10: {probe.get_top_k_latents(10)}")

    print("\n── Supervisado")
    sv = SupervisedVector()
    sv.fit(z_tr, y_tr)
    sv.evaluate(z_te, y_te)

    print("\n── Denso")
    dense = DenseProbe()
    dense.fit(z_tr, y_tr)
    dense.evaluate(z_te, y_te)

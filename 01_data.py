"""
01_data.py
==========
Carga los datasets de hate speech en inglés y español,
los balancea, limpia y divide en train/test.

Salida: dicts con splits listos para extraer activaciones.
"""

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import numpy as np


# ── Función principal ──────────────────────────────────────────────────────────

def load_and_prepare(config: dict) -> dict[str, dict]:
    """
    Devuelve:
        {
          "en": {"train": {"text": [...], "label": [...]},
                 "test":  {"text": [...], "label": [...]}},
          "es": { mismo formato }
        }
    Las etiquetas son binarias: 0 = no tóxico, 1 = tóxico.
    """
    data = {}
    for lang, dataset_id in [("en", config["dataset_en"]),
                              ("es", config["dataset_es"])]:
        print(f"\n── Cargando {lang.upper()}: {dataset_id}")
        data[lang] = _load_single(dataset_id, config, lang)

    return data


def _load_single(dataset_id: str, config: dict, lang: str) -> dict:
    """Carga un dataset, lo binariza, balancea y divide."""

    # ── 1. Carga ───────────────────────────────────────────────────────────────
    try:
        ds = load_dataset(dataset_id, split="train")
    except Exception as e:
        print(f"  Error cargando {dataset_id}: {e}")
        print("  Usando dataset alternativo: ucberkeley-dlab/measuring-hate-speech")
        ds = _fallback_dataset(lang, config)
        return ds

    print(f"  Columnas disponibles: {ds.column_names}")
    print(f"  Total ejemplos:       {len(ds)}")

    # ── 2. Detectar columnas de texto y etiqueta ───────────────────────────────
    text_col  = _detect_text_column(ds.column_names)
    label_col = _detect_label_column(ds.column_names)
    print(f"  Columna texto:        '{text_col}'")
    print(f"  Columna etiqueta:     '{label_col}'")

    texts  = ds[text_col]
    labels = ds[label_col]

    # ── 3. Binarizar etiquetas ─────────────────────────────────────────────────
    # IMSyPP usa: 0=acceptable, 1=inappropriate, 2=offensive, 3=violent
    # Binarizamos: 0 = no tóxico (clase 0), 1 = tóxico (clases 1-3)
    labels_bin = [0 if l == 0 else 1 for l in labels]

    # ── 4. Limpiar textos ──────────────────────────────────────────────────────
    texts_clean = [_clean_text(t) for t in texts]

    # ── 5. Balancear clases ────────────────────────────────────────────────────
    texts_bal, labels_bal = _balance(texts_clean, labels_bin, config["max_samples"],
                                     config["random_seed"])
    print(f"  Tras balanceo:        {len(texts_bal)} ejemplos")
    print(f"  Distribución:         {sum(labels_bal)} tóxicos / "
          f"{len(labels_bal) - sum(labels_bal)} benignos")

    # ── 6. Train/test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        texts_bal, labels_bal,
        test_size=config["test_size"],
        random_state=config["random_seed"],
        stratify=labels_bal,
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    return {
        "train": {"text": X_train, "label": y_train},
        "test":  {"text": X_test,  "label": y_test},
    }


# ── Utilidades ─────────────────────────────────────────────────────────────────

def _detect_text_column(columns: list[str]) -> str:
    """Intenta encontrar la columna de texto por nombre."""
    candidates = ["text", "tweet", "sentence", "comment", "content", "post"]
    for c in candidates:
        if c in columns:
            return c
    # Si no, devuelve la primera columna de tipo string
    return columns[0]


def _detect_label_column(columns: list[str]) -> str:
    candidates = ["label", "labels", "class", "hate", "toxic", "category",
                  "gold_label", "annotation"]
    for c in candidates:
        if c in columns:
            return c
    return columns[-1]


def _clean_text(text: str) -> str:
    """Limpieza mínima: strip, eliminar saltos de línea múltiples."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def _balance(texts: list, labels: list, max_samples: int,
             seed: int) -> tuple[list, list]:
    """Undersample para balancear clases y limitar a max_samples total."""
    rng = np.random.default_rng(seed)

    idx_pos = [i for i, l in enumerate(labels) if l == 1]
    idx_neg = [i for i, l in enumerate(labels) if l == 0]

    # Número de ejemplos por clase: mitad de max_samples, limitado por la clase menor
    n_per_class = min(max_samples // 2, len(idx_pos), len(idx_neg))

    idx_pos_s = rng.choice(idx_pos, n_per_class, replace=False).tolist()
    idx_neg_s = rng.choice(idx_neg, n_per_class, replace=False).tolist()

    idx_all = idx_pos_s + idx_neg_s
    rng.shuffle(idx_all)

    return [texts[i] for i in idx_all], [labels[i] for i in idx_all]


def _fallback_dataset(lang: str, config: dict) -> dict:
    """
    Dataset alternativo si IMSyPP no está disponible.
    Usa el dataset MLMA (multilingual) o genera datos de prueba mínimos.
    """
    try:
        # MLMA: Multilingual and Multi-Aspect Hate Speech Analysis
        ds = load_dataset("nedjmaou/MLMA_hate_speech", split="train")
        # Filtrar por idioma
        lang_map = {"en": "english", "es": "spanish"}
        ds = ds.filter(lambda x: x.get("language") == lang_map.get(lang, "english"))

        texts  = ds["tweet"]
        labels = [1 if x == "hateful" else 0 for x in ds["sentiment"]]
        texts_clean = [_clean_text(t) for t in texts]
        texts_bal, labels_bal = _balance(texts_clean, labels,
                                         config["max_samples"], config["random_seed"])
        X_train, X_test, y_train, y_test = train_test_split(
            texts_bal, labels_bal,
            test_size=config["test_size"],
            random_state=config["random_seed"],
            stratify=labels_bal,
        )
        print(f"  Fallback OK: {len(X_train)} train, {len(X_test)} test")
        return {
            "train": {"text": X_train, "label": y_train},
            "test":  {"text": X_test,  "label": y_test},
        }
    except Exception as e:
        print(f"  Fallback también falló: {e}")
        print("  Generando datos sintéticos mínimos para testing del pipeline.")
        return _synthetic_data(config)


def _synthetic_data(config: dict) -> dict:
    """Datos sintéticos para verificar que el pipeline funciona."""
    import random
    random.seed(config["random_seed"])

    toxic_en    = ["I hate you", "You are disgusting", "Kill yourself",
                   "Terrible person", "You should die"]
    non_toxic   = ["Good morning", "Nice day today", "I like cats",
                   "Interesting idea", "Thank you"]

    texts  = (toxic_en + non_toxic) * 40
    labels = ([1] * 5 + [0] * 5) * 40

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    return {
        "train": {"text": X_train, "label": y_train},
        "test":  {"text": X_test,  "label": y_test},
    }


# ── Test rápido ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from importlib import import_module
    setup = import_module("00_setup")
    config = setup.CONFIG

    data = load_and_prepare(config)

    for lang in ["en", "es"]:
        print(f"\n=== {lang.upper()} ===")
        for split in ["train", "test"]:
            n = len(data[lang][split]["text"])
            n_tox = sum(data[lang][split]["label"])
            print(f"  {split}: {n} ejemplos | {n_tox} tóxicos ({100*n_tox/n:.0f}%)")
        print(f"  Ejemplo texto:  '{data[lang]['train']['text'][0][:60]}...'")
        print(f"  Ejemplo label:  {data[lang]['train']['label'][0]}")

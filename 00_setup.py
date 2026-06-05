"""
00_setup.py
===========
Ejecutar este bloque primero en Google Colab.
Instala dependencias y verifica el entorno.
"""

# ── Instalación ────────────────────────────────────────────────────────────────
# Ejecutar en Colab como celda de shell:
#   !pip install torch transformers datasets scikit-learn sae-lens huggingface_hub -q

import sys
import importlib

REQUIRED = {
    "torch":            "torch",
    "transformers":     "transformers",
    "datasets":         "datasets",
    "sklearn":          "scikit-learn",
    "sae_lens":         "sae-lens",
    "huggingface_hub":  "huggingface_hub",
    "numpy":            "numpy",
    "matplotlib":       "matplotlib",
    "scipy":            "scipy",
}

missing = []
for module, package in REQUIRED.items():
    try:
        importlib.import_module(module)
    except ImportError:
        missing.append(package)

if missing:
    print(f"Faltan paquetes: {missing}")
    print(f"Ejecuta: pip install {' '.join(missing)} -q")
else:
    print("✓ Todas las dependencias disponibles.")

# ── Constantes globales del proyecto ──────────────────────────────────────────
import torch

CONFIG = {
    # Modelo y SAE
    "model_name":       "EleutherAI/pythia-70m",
    "sae_release":      "pythia-70m-deduped-res-sm",   # release en SAE Lens
    "sae_id":           "blocks.4.hook_resid_post",     # capa 4 (0-indexed), residual stream
    "layer":            4,

    # Datos
    "dataset_en":       "IMSyPP/hate_speech_en",        # inglés, HuggingFace
    "dataset_es":       "IMSyPP/hate_speech_es",        # español, HuggingFace
    "max_length":       128,                            # tokens máximos por texto
    "max_samples":      2000,                           # por idioma (para Colab gratuito)
    "test_size":        0.2,
    "random_seed":      42,

    # Probe
    "top_k_values":     [5, 10, 25, 50, 100],          # barrido de k para curva F1 vs k
    "top_k_main":       50,                             # k principal para análisis Jaccard
    "n_permutations":   1000,                           # para test de permutación H1

    # Hardware
    "device":           "cuda" if torch.cuda.is_available() else "cpu",

    # Paths de salida
    "output_dir":       "./resultados",
}

print(f"\nDispositivo: {CONFIG['device']}")
print(f"Modelo:      {CONFIG['model_name']}")
print(f"SAE:         {CONFIG['sae_release']} / {CONFIG['sae_id']}")
print(f"Datos EN:    {CONFIG['dataset_en']}")
print(f"Datos ES:    {CONFIG['dataset_es']}")

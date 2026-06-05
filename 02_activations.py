"""
02_activations.py
=================
Pasa los textos por Pythia-70M y el SAE preentrenado,
devolviendo dos matrices de representaciones por split:

  - z  ∈ ℝ^(N × d_model)   activaciones brutas del LLM (para brazo B)
  - h  ∈ ℝ^(N × d_sae)     latentes sparse del SAE       (para brazo A)

Cada fila corresponde a la representación del token [CLS] / promedio
del residual stream de la capa configurada.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ── Carga del modelo LLM ───────────────────────────────────────────────────────

def load_model(config: dict):
    """Carga Pythia-70M y su tokenizer."""
    print(f"Cargando modelo: {config['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    # Pythia no tiene pad token por defecto — usamos EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        output_hidden_states=True,   # necesario para extraer activaciones internas
    )
    model.eval()
    model.to(config["device"])

    d_model = model.config.hidden_size
    print(f"  Modelo cargado. d_model = {d_model}, capas = {model.config.num_hidden_layers}")
    return model, tokenizer


# ── Carga del SAE ──────────────────────────────────────────────────────────────

def load_sae(config: dict):
    """
    Carga el SAE preentrenado desde SAE Lens (HuggingFace).
    Devuelve un objeto SAE con métodos encode() y decode().
    """
    try:
        from sae_lens import SAE
        print(f"\nCargando SAE: {config['sae_release']} / {config['sae_id']}")
        sae, cfg_dict, _ = SAE.from_pretrained(
            release=config["sae_release"],
            sae_id=config["sae_id"],
            device=config["device"],
        )
        sae.eval()
        d_sae = sae.cfg.d_sae
        print(f"  SAE cargado. d_sae = {d_sae}")
        return sae

    except ImportError:
        print("  sae_lens no disponible — usando SAE mock para desarrollo.")
        print("  Instala con: pip install sae-lens")
        return _MockSAE(config)

    except Exception as e:
        print(f"  Error cargando SAE: {e}")
        print("  Usando SAE mock para continuar el desarrollo del pipeline.")
        return _MockSAE(config)


class _MockSAE:
    """SAE simulado: PCA sparse para testing sin GPU/dependencias."""
    def __init__(self, config):
        self.device   = config["device"]
        self.d_sae    = 512      # dimensión simulada
        self.W_enc    = None     # se inicializa al primer encode()
        self._fitted  = False

    def fit(self, z: np.ndarray):
        """Ajusta PCA sobre las activaciones para simular el SAE."""
        from sklearn.decomposition import PCA
        n_components = min(self.d_sae, z.shape[0], z.shape[1])
        self.pca = PCA(n_components=n_components)
        self.pca.fit(z)
        self._fitted  = True
        self.d_sae    = n_components

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Proyecta activaciones al espacio del SAE (sparse via ReLU)."""
        if not self._fitted:
            # fit lazy en el primer batch
            self.fit(z.cpu().float().numpy())
        z_np  = z.cpu().float().numpy()
        h_np  = self.pca.transform(z_np)
        h_np  = np.maximum(h_np, 0)          # ReLU → sparse
        return torch.tensor(h_np, dtype=torch.float32)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


# ── Extracción de activaciones ─────────────────────────────────────────────────

def extract_representations(
    texts: list[str],
    model,
    tokenizer,
    sae,
    config: dict,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """
    Procesa una lista de textos y devuelve:
        {
          "z": np.ndarray (N, d_model)   — activaciones LLM brutas
          "h": np.ndarray (N, d_sae)     — latentes SAE
        }
    Usa el promedio sobre tokens no-padding como representación del texto.
    """
    all_z = []
    all_h = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extrayendo"):
        batch_texts = texts[i : i + batch_size]

        # ── Tokenización ───────────────────────────────────────────────────────
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["max_length"],
        ).to(config["device"])

        # ── Forward pass ───────────────────────────────────────────────────────
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states: tuple de (n_layers + 1) tensores (batch, seq_len, d_model)
        # Usamos la capa configurada (post-residual stream)
        layer_idx = config["layer"]
        hidden = outputs.hidden_states[layer_idx]   # (batch, seq, d_model)

        # ── Promedio sobre tokens no-padding ───────────────────────────────────
        mask     = inputs["attention_mask"].unsqueeze(-1).float()  # (batch, seq, 1)
        z_batch  = (hidden * mask).sum(dim=1) / mask.sum(dim=1)    # (batch, d_model)

        # ── Paso por el SAE ────────────────────────────────────────────────────
        h_batch = sae.encode(z_batch)   # (batch, d_sae)

        all_z.append(z_batch.cpu().float().numpy())
        all_h.append(h_batch.cpu().float().numpy())

    return {
        "z": np.vstack(all_z),
        "h": np.vstack(all_h),
    }


def extract_all(data: dict, model, tokenizer, sae, config: dict) -> dict:
    """
    Wrapper: extrae representaciones para todos los splits e idiomas.

    Devuelve:
        {
          "en": {
            "train": {"z": ..., "h": ..., "label": [...]},
            "test":  {"z": ..., "h": ..., "label": [...]}
          },
          "es": { ... }
        }
    """
    reps = {}
    for lang in ["en", "es"]:
        print(f"\n{'='*50}")
        print(f"Idioma: {lang.upper()}")
        reps[lang] = {}
        for split in ["train", "test"]:
            print(f"\n  Split: {split}")
            texts  = data[lang][split]["text"]
            labels = data[lang][split]["label"]
            rep    = extract_representations(texts, model, tokenizer, sae, config)
            reps[lang][split] = {
                "z":     rep["z"],
                "h":     rep["h"],
                "label": np.array(labels),
            }
            print(f"  z shape: {rep['z'].shape} | h shape: {rep['h'].shape}")
            sparsity = (rep["h"] == 0).mean()
            print(f"  Sparsity SAE (fracción ceros): {sparsity:.3f}")

    return reps


# ── Test rápido ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from importlib import import_module

    setup = import_module("00_setup")
    data_mod = import_module("01_data")

    config = setup.CONFIG
    # Test mínimo con pocos datos
    config["max_samples"] = 20

    data = data_mod.load_and_prepare(config)
    model, tokenizer = load_model(config)
    sae = load_sae(config)

    reps = extract_all(data, model, tokenizer, sae, config)

    for lang in ["en", "es"]:
        for split in ["train", "test"]:
            r = reps[lang][split]
            print(f"{lang}/{split} — z: {r['z'].shape}, h: {r['h'].shape}, "
                  f"labels: {r['label'].shape}")

"""Build a CALVIN-flavored manifest from existing aux ckpts.

Subset of LIBERO manifest: only headline backbones × {frozen, +aux}.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    entries = []

    # ---- spec for headline backbones ----
    families = [
        # (family, name_prefix, aux_subdir, frozen_kwargs)
        ("vjepa2",   "vjepa2",        "vjepa2_aux_calvin",
         {"arch": "vit_large", "img_size": 256, "n_params_million": 304.0, "model_size": "vit_large"}),
        ("vjepa2_1", "vjepa2_1",      "vjepa2_1_aux_calvin",
         {"arch": "vit_base",  "img_size": 384, "n_params_million": 86.0,  "model_size": "vit_base",
          "ckpt_key": "ema_encoder"}),
        ("webdino",  "webdino",       "webdino_aux_calvin",
         {"img_size": 224, "n_params_million": 303.7, "model_size": "vit_large",
          "hf_repo": "facebook/webssl-dino300m-full2b-224"}),
        ("siglip2",  "siglip2",       "siglip2_aux_calvin",
         {"img_size": 256, "n_params_million": 303.5, "model_size": "vit_large",
          "hf_repo": "google/siglip2-large-patch16-256"}),
        ("videomae", "videomae",      "videomae_aux_calvin",
         {"img_size": 224, "n_params_million": 303.9, "model_size": "vit_large",
          "hf_repo": "MCG-NJU/videomae-large"}),
        ("lapa",     "lapa",          "lapa_aux_calvin",
         {"img_size": 256, "n_params_million": 344.0, "model_size": "344M"}),
        ("sd3vae",   "sdvae",         "sd3vae_aux_calvin",
         {"img_size": 128, "n_params_million": 83.7,  "model_size": "sdxl_vae",
          "hf_repo": "stabilityai/sdxl-vae"}),
        ("cosmos_tokenizer", "cosmos_tokenizer", "cosmos_tokenizer_aux_calvin",
         {"img_size": 128, "n_params_million": 9.0,   "model_size": "cosmos1_8x8x8",
          "hf_repo": "nvidia/Cosmos-Tokenizer-CV8x8x8"}),
    ]

    for family, name_pre, aux_subdir, kw in families:
        # frozen row (no ckpt — falls back to default HF/extractor weights)
        entries.append({
            "base_model": family, "variant": f"{name_pre}_frozen_calvin",
            "family": family,
            "objective": f"frozen ({name_pre})", "seq_len": 32,
            "n_params_million": kw.get("n_params_million"),
            "model_size": kw.get("model_size"),
            "n_cams": 1, "action_dim": 0,
            "pretrained": "official pretrain", "ckpt": None,
            **{k: v for k, v in kw.items() if k in ("arch", "img_size", "ckpt_key", "hf_repo")},
            "num_frames": 32,
        })
        # aux row
        ck = REPO_ROOT / "runs" / "comparison" / aux_subdir / "ckpt_last.pt"
        if ck.exists():
            entries.append({
                "base_model": family, "variant": f"{name_pre}_aux_calvin",
                "family": family,
                "objective": f"{family} + InvDynAuxHead λ=1.0 on CALVIN",
                "seq_len": 32,
                "n_params_million": kw.get("n_params_million"),
                "model_size": kw.get("model_size"),
                "n_cams": 1, "action_dim": 0,
                "pretrained": f"official + CALVIN aux finetune",
                "ckpt": str(ck.relative_to(REPO_ROOT)),
                **{k: v for k, v in kw.items() if k in ("arch", "img_size", "ckpt_key", "hf_repo")},
                "num_frames": 32,
            })

    out = REPO_ROOT / "runs" / "full_results" / "manifest_calvin.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(entries, sort_keys=False, allow_unicode=True))
    print(f"wrote {out} with {len(entries)} entries:")
    for e in entries:
        print(f"  {e['variant']:35s} ckpt={e['ckpt']}")


if __name__ == "__main__":
    main()

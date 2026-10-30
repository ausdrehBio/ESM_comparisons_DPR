#!/usr/bin/env python3
"""Download one model's weights into the project cache and prove they load (on CPU).

Runs inside that model's own env (fair-esm for esm2_650m, esm for esmc_600m). Cache
locations come from TORCH_HOME / HF_HOME, set by run_server.sh. Prints a JSON summary on
stdout, which run_server.sh keeps as the step's completion marker.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

ESMC_REPO = "biohub/ESMC-600M"


def fetch_esm2():
    import esm
    import torch

    # fair-esm downloads into torch.hub's checkpoint dir on first load.
    model, _ = esm.pretrained.esm2_t33_650M_UR50D()
    return {
        "package": f"fair-esm {esm.__version__}",
        "source": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
        "cache": os.path.join(torch.hub.get_dir(), "checkpoints"),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def fetch_esmc():
    import esm
    import torch
    from esm.models.esmc import EsmcModel
    from huggingface_hub import snapshot_download

    # Same file patterns esm's own loader requests, so the offline lookup in embed.py hits this snapshot.
    path = snapshot_download(repo_id=ESMC_REPO, allow_patterns=["*.json", "*.safetensors"])
    model = EsmcModel.from_pretrained(ESMC_REPO, device="cpu", dtype=torch.float32, local_files_only=True)
    return {
        "package": f"esm {esm.__version__}",
        "source": f"https://huggingface.co/{ESMC_REPO}",
        "revision": os.path.basename(path),
        "cache": path,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


FETCHERS = {"esm2_650m": fetch_esm2, "esmc_600m": fetch_esmc}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", choices=sorted(FETCHERS))
    args = ap.parse_args()

    info = FETCHERS[args.model]()
    print(f"download_models: {args.model}: {info['parameters']:,} parameters loaded from {info['cache']}", file=sys.stderr)
    print(json.dumps({"model": args.model, "completed": datetime.now(timezone.utc).isoformat(), **info}))


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# ESM-C 600M vs ESM-2 650M embedding benchmark: one-command server entry point.
#
# Chains environment install -> reference build -> model download -> Nextflow pipeline.
# Every step is guarded on its outputs, so re-running is safe and only redoes what is
# missing or out of date.
#
# Usage:
#   bash run_server.sh                  full run (needs a visible NVIDIA GPU)
#   bash run_server.sh --dry-run        preflight only: report what each step would do, change nothing
#   bash run_server.sh --download-only  everything except the pipeline (e.g. on a login node with internet)
#   bash run_server.sh [nextflow args]  anything else goes to `nextflow run main.nf`, e.g.
#                                       --precision bf16 --max_tokens 4096 --outdir results_bf16
#
# Environment overrides:
#   INPUT_CSV              input CSV with sequence_id and sequence columns, others are ignored (default:
#                          esm_sequences.csv next to this script; a relative path is taken from the
#                          directory you run the script in)
#   TORCH_INDEX_URL        pip index for torch, e.g. https://download.pytorch.org/whl/cu126 for older drivers
#   MICROMAMBA_VERSION     micromamba release to bootstrap when no micromamba/conda exists (default: latest)
#   ESM_BENCH_ALLOW_CPU=1  run on CPU instead of failing when no GPU is visible
#   HF_TOKEN               Hugging Face token; only needed if the ESM-C weights become gated

set -Eeuo pipefail

CALLER_DIR="$PWD"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
trap 'die "command failed at line $LINENO: $BASH_COMMAND"' ERR

sha256() {  # sha256 of stdin
  if type -P sha256sum >/dev/null; then sha256sum | cut -d' ' -f1; else shasum -a 256 | cut -d' ' -f1; fi
}

fetch() {  # download a URL to stdout
  if type -P curl >/dev/null; then curl -fsSL "$1"; else wget -qO- "$1"; fi
}

# For out="$(capture cmd ...)" || die "...": stdout+stderr of cmd, keeping its exit status.
# Without the trap reset the inherited ERR trap fires inside the subshell first and logs a
# misleading "command failed" line before the caller's own message.
capture() { trap - ERR; "$@" 2>&1; }

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
CORE_ENV=esm_bench_core     # nextflow, java, analysis
ESM2_ENV=esm_bench_esm2     # fair-esm (ESM-2)
ESMC_ENV=esm_bench_esmc     # esm (ESM-C); both packages install the module `esm`, hence separate envs
MODEL_PYTHON="python=3.12"
TORCH_SPEC="torch==2.11.*"  # same torch in both model envs; esm 3.4.x requires >=2.11,<2.12
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

INPUT_CSV="${INPUT_CSV:-$SCRIPT_DIR/esm_sequences.csv}"
if [[ $INPUT_CSV != /* ]]; then INPUT_CSV="$CALLER_DIR/$INPUT_CSV"; fi
REF_DIR="$SCRIPT_DIR/data/ref"
MODEL_DIR="$SCRIPT_DIR/data/models"
LOCAL_MAMBA_ROOT="$SCRIPT_DIR/.micromamba"

# Project-local caches keep multi-GB downloads out of (often quota-limited) home directories.
export TORCH_HOME="$SCRIPT_DIR/.cache/torch"
export HF_HOME="$SCRIPT_DIR/.cache/huggingface"
export PIP_CACHE_DIR="$SCRIPT_DIR/.cache/pip"
export NXF_HOME="$SCRIPT_DIR/.nxf_home"
# Keep env Pythons isolated from user-level packages and pip settings common on shared servers.
export PYTHONNOUSERSITE=1 PIP_USER=0 PIP_REQUIRE_VIRTUALENV=0 PIP_DISABLE_PIP_VERSION_CHECK=1
unset PYTHONPATH PYTHONHOME
# fair-esm checkpoints pickle an argparse.Namespace, which torch>=2.6 refuses under weights_only.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
DRY_RUN=0
DOWNLOAD_ONLY=0
PASSTHRU=()
for arg in "$@"; do
  case "$arg" in
    --dry-run)       DRY_RUN=1 ;;
    --download-only) DOWNLOAD_ONLY=1 ;;
    -h|--help)       sed -n '2,/^$/s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)               PASSTHRU+=("$arg") ;;
  esac
done
if [[ $DRY_RUN == 1 && $DOWNLOAD_ONLY == 1 ]]; then
  die "--dry-run and --download-only are mutually exclusive"
fi
if [[ $DRY_RUN == 1 ]]; then log "dry run: nothing will be installed, downloaded or written"; fi

# ---------------------------------------------------------------------------
# Path sanity: fail before anything expensive if the clone is incomplete
# ---------------------------------------------------------------------------
log "preflight: checking repository files in $SCRIPT_DIR"
for f in main.nf nextflow.config envs/core.yml envs/esm2-requirements.txt envs/esmc-requirements.txt \
         bin/prepare_input.py bin/download_models.py bin/embed.py bin/compare_embeddings.py; do
  [[ -f "$SCRIPT_DIR/$f" ]] || die "missing $f in $SCRIPT_DIR (incomplete clone?)"
done
[[ -s "$INPUT_CSV" ]] || die "input CSV missing or empty: $INPUT_CSV (override with INPUT_CSV=...)"

# ---------------------------------------------------------------------------
# Package manager
# ---------------------------------------------------------------------------
PM=""
PM_KIND=""
find_pm() {
  if [[ -x $LOCAL_MAMBA_ROOT/bin/micromamba ]]; then
    PM="$LOCAL_MAMBA_ROOT/bin/micromamba"; PM_KIND=micromamba
    export MAMBA_ROOT_PREFIX="$LOCAL_MAMBA_ROOT"
  elif type -P micromamba >/dev/null; then
    PM="$(type -P micromamba)"; PM_KIND=micromamba
  elif type -P conda >/dev/null; then
    PM="$(type -P conda)"; PM_KIND=conda
  elif type -P mamba >/dev/null; then
    PM="$(type -P mamba)"; PM_KIND=micromamba  # mamba without conda is mamba 2.x, which shares micromamba's CLI
  fi
}

install_micromamba() {
  local platform url
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)               platform=linux-64 ;;
    Linux-aarch64|Linux-arm64)  platform=linux-aarch64 ;;
    Linux-ppc64le)              platform=linux-ppc64le ;;
    Darwin-arm64)               platform=osx-arm64 ;;
    Darwin-x86_64)              platform=osx-64 ;;
    *) die "no micromamba build for $(uname -s)-$(uname -m); install conda or micromamba manually" ;;
  esac
  url="https://micro.mamba.pm/api/micromamba/$platform/${MICROMAMBA_VERSION:-latest}"
  mkdir -p "$LOCAL_MAMBA_ROOT"
  log "installing micromamba ($platform, ${MICROMAMBA_VERSION:-latest}) into $LOCAL_MAMBA_ROOT"
  fetch "$url" | tar -xj -C "$LOCAL_MAMBA_ROOT" bin/micromamba || die "could not download/unpack micromamba from $url"
  "$LOCAL_MAMBA_ROOT/bin/micromamba" --version >/dev/null || die "bootstrapped micromamba does not run"
}

log "[1/5] package manager"
find_pm
if [[ -n $PM ]]; then
  log "using $PM_KIND at $PM"
  "$PM" env list >/dev/null || die "'$PM env list' failed; is this $PM_KIND installation usable?"
else
  type -P curl >/dev/null || type -P wget >/dev/null || die "neither micromamba/conda nor curl/wget found; cannot bootstrap"
  type -P bzip2 >/dev/null || die "bzip2 is required to unpack micromamba"
  if [[ $DRY_RUN == 1 ]]; then
    log "no micromamba/conda found: would install micromamba into $LOCAL_MAMBA_ROOT"
  else
    install_micromamba
    find_pm
  fi
fi

# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------
env_prefix() {  # prefix of a named env; prints nothing if absent
  { "$PM" env list 2>/dev/null || true; } | awk -v n="$1" '$1 == n && !found { print $NF; found = 1 }'
}

env_spec_hash() {  # $1 = core|esm2|esmc; changes whenever the env's spec does
  if [[ $1 == core ]]; then
    sha256 < envs/core.yml
  else
    { cat "envs/$1-requirements.txt"; printf '%s\n' "$MODEL_PYTHON" "$TORCH_SPEC" "$TORCH_INDEX_URL"; } | sha256
  fi
}

env_state() {  # $1 = env name, $2 = kind -> ok | stale | absent
  local prefix stamp
  if [[ -z $PM ]]; then echo absent; return; fi
  prefix="$(env_prefix "$1")"
  stamp="$prefix/.esm_bench.stamp"
  if [[ -z $prefix ]]; then
    echo absent
  elif [[ -f $stamp && "$(cat "$stamp")" == "$(env_spec_hash "$2")" ]]; then
    echo ok
  else
    echo stale
  fi
}

ensure_env() {  # $1 = env name, $2 = kind (core|esm2|esmc)
  local name=$1 kind=$2 state prefix
  state="$(env_state "$name" "$kind")"
  if [[ $state == ok ]]; then
    log "env $name: up to date, skip"
    return 0
  fi
  if [[ $DRY_RUN == 1 ]]; then
    if [[ $state == stale ]]; then
      log "env $name: spec changed or earlier install incomplete, would rebuild"
    else
      log "env $name: would create"
    fi
    return 0
  fi
  if [[ $state == stale ]]; then
    log "env $name: spec changed or earlier install incomplete, removing to rebuild"
    "$PM" env remove -y -n "$name" || die "could not remove env $name"
  fi

  log "env $name: creating"
  if [[ $kind == core && $PM_KIND == micromamba ]]; then
    "$PM" create -y -n "$name" -f envs/core.yml || die "creating env $name from envs/core.yml failed"
  elif [[ $kind == core ]]; then
    "$PM" env create -n "$name" -f envs/core.yml || die "creating env $name from envs/core.yml failed"
  else
    "$PM" create -y -n "$name" -c conda-forge --override-channels "$MODEL_PYTHON" pip \
      || die "creating env $name failed"
  fi
  prefix="$(env_prefix "$name")"
  [[ -n $prefix ]] || die "env $name was created but '$PM env list' does not show it"

  if [[ $kind != core ]]; then
    local pip=("$prefix/bin/python" -m pip install)
    log "env $name: pip install $TORCH_SPEC${TORCH_INDEX_URL:+ from $TORCH_INDEX_URL}"
    if [[ -n $TORCH_INDEX_URL ]]; then
      "${pip[@]}" "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL" || die "torch install failed in $name"
    else
      "${pip[@]}" "$TORCH_SPEC" || die "torch install failed in $name"
    fi
    log "env $name: pip install -r envs/$kind-requirements.txt"
    "${pip[@]}" -r "envs/$kind-requirements.txt" || die "pip install -r envs/$kind-requirements.txt failed in $name"
  fi
  # Written last: the stamp exists only after a fully successful install.
  env_spec_hash "$kind" > "$prefix/.esm_bench.stamp"
  log "env $name: ready at $prefix"
}

log "[2/5] environments"
ensure_env "$CORE_ENV" core
ensure_env "$ESM2_ENV" esm2
ensure_env "$ESMC_ENV" esmc

ENVS_READY=1
for pair in "$CORE_ENV:core" "$ESM2_ENV:esm2" "$ESMC_ENV:esmc"; do
  if [[ "$(env_state "${pair%%:*}" "${pair##*:}")" != ok ]]; then ENVS_READY=0; fi
done

CORE_PREFIX=""
ESM2_PREFIX=""
ESMC_PREFIX=""
if [[ $ENVS_READY == 1 ]]; then
  CORE_PREFIX="$(env_prefix "$CORE_ENV")"
  ESM2_PREFIX="$(env_prefix "$ESM2_ENV")"
  ESMC_PREFIX="$(env_prefix "$ESMC_ENV")"
elif [[ $DRY_RUN == 0 ]]; then
  die "environments not ready after install; see messages above"
fi

# Activate by prefix, never by name: shared servers often have several conda installs.
activate_core() {
  export CONDA_PREFIX="$CORE_PREFIX"
  export PATH="$CORE_PREFIX/bin:$PATH"
  export JAVA_HOME="$CONDA_PREFIX/lib/jvm"  # conda's OpenJDK lives here; never fall back to a system Java
  hash -r
}

check_tools() {
  local tool out java_out
  for tool in python nextflow java; do
    [[ -x $CORE_PREFIX/bin/$tool ]] || die "env $CORE_ENV: $tool missing from $CORE_PREFIX/bin"
    [[ "$(type -P "$tool")" == "$CORE_PREFIX/bin/$tool" ]] \
      || die "$tool resolves to $(type -P "$tool"), not $CORE_PREFIX/bin/$tool"
  done
  [[ -x $JAVA_HOME/bin/java ]] || die "JAVA_HOME=$JAVA_HOME has no bin/java"
  java_out="$(capture "$JAVA_HOME/bin/java" -version)" || die "java at $JAVA_HOME does not start: $java_out"
  out="$(capture nextflow -version)" || die "nextflow does not start (JAVA_HOME=$JAVA_HOME): $out"
  log "tools: nextflow $(awk '$1 == "version" { print $2; exit }' <<<"$out"), $(head -n1 <<<"$java_out")"

  out="$(capture "$ESM2_PREFIX/bin/python" -c '
import torch, esm, esm.pretrained
esm.pretrained.esm2_t33_650M_UR50D  # attribute only exists in fair-esm, not in EvolutionaryScale esm
print(f"torch {torch.__version__}, fair-esm {esm.__version__}")')" \
    || die "env $ESM2_ENV: torch/fair-esm import check failed: $out"
  log "env $ESM2_ENV: $(tail -n1 <<<"$out")"

  out="$(capture "$ESMC_PREFIX/bin/python" -c '
import torch, esm
from esm.models.esmc import EsmcModel, EsmcTokenizer
print(f"torch {torch.__version__}, esm {esm.__version__}")')" \
    || die "env $ESMC_ENV: torch/esm import check failed: $out"
  log "env $ESMC_ENV: $(tail -n1 <<<"$out")"
}

if [[ $ENVS_READY == 1 ]]; then
  activate_core
  log "tool check"
  check_tools
else
  log "tool check: skipped until the environments exist"
fi

# ---------------------------------------------------------------------------
# Reference build: CSV -> validated FASTA + manifest
# ---------------------------------------------------------------------------
build_reference() {
  local want stamp="$REF_DIR/input.sha256"
  want="$(sha256 < "$INPUT_CSV")"
  if [[ -s $REF_DIR/sequences.fasta && -s $REF_DIR/manifest.tsv && -f $stamp && "$(cat "$stamp")" == "$want" ]]; then
    log "reference: $REF_DIR is up to date for $INPUT_CSV, skip"
    return 0
  fi
  if [[ $ENVS_READY == 0 ]]; then
    log "reference: would build $REF_DIR from $INPUT_CSV"
    return 0
  fi
  if [[ $DRY_RUN == 1 ]]; then
    log "reference: would build $REF_DIR; validating $INPUT_CSV now (no files written)"
    "$CORE_PREFIX/bin/python" bin/prepare_input.py --csv "$INPUT_CSV" --check || die "input validation failed: $INPUT_CSV"
    return 0
  fi
  log "reference: building $REF_DIR from $INPUT_CSV"
  "$CORE_PREFIX/bin/python" bin/prepare_input.py --csv "$INPUT_CSV" --outdir "$REF_DIR" \
    || die "reference build failed for $INPUT_CSV"
  printf '%s\n' "$want" > "$stamp"
}

log "[3/5] reference build"
build_reference

# ---------------------------------------------------------------------------
# Model download (the "read download" of this project): weights into the project caches
# ---------------------------------------------------------------------------
download_model() {  # $1 = model key, $2 = env prefix
  local model=$1 prefix=$2 marker="$MODEL_DIR/$1.complete" weights
  case $model in
    esm2_650m) weights="$TORCH_HOME/hub/checkpoints/esm2_t33_650M_UR50D.pt" ;;
    esmc_600m) weights="$HF_HOME/hub/models--biohub--ESMC-600M" ;;
  esac
  if [[ -s $marker && -e $weights ]]; then
    log "weights $model: cached, skip"
    return 0
  fi
  if [[ $DRY_RUN == 1 ]]; then
    log "weights $model: would download to $(dirname "$weights")"
    return 0
  fi
  mkdir -p "$MODEL_DIR" "$TORCH_HOME" "$HF_HOME"
  log "weights $model: downloading and test-loading on CPU"
  "$prefix/bin/python" bin/download_models.py "$model" > "$marker.tmp" \
    || die "weights $model: download or test-load failed (network access? HF_TOKEN?)"
  mv "$marker.tmp" "$marker"
  log "weights $model: done"
}

log "[4/5] model download"
download_model esm2_650m "$ESM2_PREFIX"
download_model esmc_600m "$ESMC_PREFIX"

if [[ $DOWNLOAD_ONLY == 1 ]]; then
  log "--download-only: setup complete. Run without the flag on a GPU node to start the pipeline."
  exit 0
fi

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
check_cuda() {  # $1 = env name, $2 = env prefix
  local out driver
  if out="$(capture "$2/bin/python" -c 'import torch; print(torch.version.cuda); assert torch.cuda.is_available()')"; then
    return 0
  fi
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
  die "env $1: torch (built for CUDA $(head -n1 <<<"$out")) cannot use the GPU (driver ${driver:-unknown}). Set TORCH_INDEX_URL to a matching wheel index, e.g. https://download.pytorch.org/whl/cu126, and re-run; the env rebuilds automatically."
}

DEVICE=cuda
log "[5/5] pipeline"
# nvidia-smi -L can exit 0 while printing "No devices found." (driver installed, no GPU allocated
# to this job), so count the "GPU N: ..." lines instead of trusting the exit status.
GPU_LIST="" SMI_STATUS="nvidia-smi not found"
if type -P nvidia-smi >/dev/null; then
  SMI_OUT="$(nvidia-smi -L 2>&1 || true)"
  GPU_LIST="$(awk '/^GPU [0-9]+:/' <<<"$SMI_OUT")"
  SMI_STATUS="nvidia-smi -L: $(head -n1 <<<"$SMI_OUT")"
fi
if [[ -n $GPU_LIST ]]; then
  log "GPU: $(head -n1 <<<"$GPU_LIST") ($(wc -l <<<"$GPU_LIST" | tr -d ' ') visible)"
  if [[ $ENVS_READY == 1 ]]; then
    check_cuda "$ESM2_ENV" "$ESM2_PREFIX"
    check_cuda "$ESMC_ENV" "$ESMC_PREFIX"
  fi
elif [[ ${ESM_BENCH_ALLOW_CPU:-0} == 1 ]]; then
  DEVICE=cpu
  log "WARNING: no GPU visible; running on CPU because ESM_BENCH_ALLOW_CPU=1"
elif [[ $DRY_RUN == 1 ]]; then
  log "GPU: none visible here ($SMI_STATUS); a real run would stop at this point (expected on a login node)"
else
  die "no NVIDIA GPU visible to this job ($SMI_STATUS). On SLURM this usually means the job did not request one: add e.g. '#SBATCH --gres=gpu:1' (or '--gpus=1') and use a GPU partition. To run on CPU instead, set ESM_BENCH_ALLOW_CPU=1."
fi

RESUME=(-resume)
for arg in ${PASSTHRU[@]+"${PASSTHRU[@]}"}; do
  if [[ $arg == -resume ]]; then RESUME=(); fi
done
NF_CMD=(nextflow run "$SCRIPT_DIR/main.nf" ${RESUME[@]+"${RESUME[@]}"}
        --input "$REF_DIR/sequences.fasta" --manifest "$REF_DIR/manifest.tsv"
        --core_prefix "$CORE_PREFIX" --esm2_prefix "$ESM2_PREFIX" --esmc_prefix "$ESMC_PREFIX"
        --device "$DEVICE" ${PASSTHRU[@]+"${PASSTHRU[@]}"})

if [[ $DRY_RUN == 1 ]]; then
  log "would exec: ${NF_CMD[*]}"
  log "dry run complete, nothing was changed"
  exit 0
fi

# Compute nodes often have no internet; everything needed is cached by now.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NXF_OFFLINE=true NXF_ANSI_LOG=false
log "handing off to Nextflow: ${NF_CMD[*]}"
trap - ERR
exec "${NF_CMD[@]}"

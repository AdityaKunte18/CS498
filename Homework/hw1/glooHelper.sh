#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 -n <nnodes> -P <nproc_per_node> -r <node_rank> -e <epochs> -p <problem #> -s <script.py> [-- <script args...>]

Options:
  -n    Number of nodes (nnodes)
  -P    Number of processes per node (nproc-per-node)
  -r    Node rank (0..nnodes-1)
  -e    Epochs (0,..,3)
  -q    question (2 or 3)    
  -s    Training script path (e.g., ./run.py)

Example:
  $0 -n 3 -P 1 -r 0 -e 3 -q 2 -s ./run.py
EOF
  exit 1
}

NNODES="" NPROC="" NODE_RANK="" SCRIPT="" EPOCHS=1 QUESTION=2
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) NNODES="$2"; shift 2 ;;
    -P) NPROC="$2"; shift 2 ;;
    -r) NODE_RANK="$2"; shift 2 ;;
    -e) EPOCHS="$2"; shift 2 ;;
    -q) QUESTION="$2"; shift 2 ;;
    -s) SCRIPT="$2"; shift 2 ;;
    --) shift; break ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done
SCRIPT_ARGS=("$@")

[[ -z "${NNODES}" || -z "${NPROC}" || -z "${NODE_RANK}" || -z "${MASTER_ADDR}" || -z "${QUESTION}" || -z "${SCRIPT}" ]] && usage
[[ ! -f "${SCRIPT}" ]] && { echo "ERROR: script not found: ${SCRIPT}"; exit 2; }

# --- netid.txt next to this wrapper determines MASTER_PORT ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HASH_SRC="${SCRIPT_DIR}/netid.txt"
[[ ! -f "${HASH_SRC}" ]] && {
  echo "ERROR: ${HASH_SRC} not found. Create it (e.g., with your NetID) to derive a stable port."
  exit 3
}

sha256_hex() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "ERROR: need sha256sum or shasum installed." >&2
    exit 4
  fi
}

hash_to_port() {
  local hex="$1"
  local num=$(( 0x${hex:0:8} ))
  echo $(( (num % 40000) + 20000 ))   # 20000..59999
}

HASH_HEX="$(sha256_hex "${HASH_SRC}")"
MASTER_PORT="$(hash_to_port "${HASH_HEX}")"

# --- If chosen port is busy, bump upward until free (cap attempts) ---
port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn | awk '{print $4}' | grep -qE "[:.]${port}$"
  elif command -v netstat >/dev/null 2>&1; then
    netstat -ltn | awk '{print $4}' | grep -qE "[:.]${port}$"
  else
    return 1
  fi
}
TRY_PORT="${MASTER_PORT}"
for _ in $(seq 1 50); do
  if ! port_in_use "${TRY_PORT}"; then MASTER_PORT="${TRY_PORT}"; break; fi
  TRY_PORT=$((TRY_PORT+1))
done
[[ "${MASTER_PORT}" -lt 20000 || "${MASTER_PORT}" -gt 65535 ]] && { echo "ERROR: No free port found near ${MASTER_PORT}."; exit 5; }

echo "Using:"
echo "  MASTER_ADDR  = ${MASTER_ADDR}"
echo "  MASTER_PORT  = ${MASTER_PORT}   (from netid.txt)"
echo "  NNODES       = ${NNODES}"
echo "  NPROC/Node   = ${NPROC}"
echo "  NODE_RANK    = ${NODE_RANK}"
echo "  SCRIPT       = ${SCRIPT}"
echo "  QUESTION     = ${QUESTION}"
echo "  EPOCHS       = ${EPOCHS}"


exec torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC}" \
  --node_rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  "${SCRIPT}" \
  --epochs "${EPOCHS}" \
  --question "${QUESTION}" \
  "${SCRIPT_ARGS[@]}"

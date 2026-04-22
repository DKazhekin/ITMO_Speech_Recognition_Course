#!/usr/bin/env bash
# publish_release.sh — Publish a GP1 release directory to GitHub Releases.
#
# Usage:
#   ./scripts/publish_release.sh <baseline> <tag> [--dry-run] [--draft]
#
# Args:
#   baseline  Model baseline name (e.g. quartznet). Matches releases/<baseline>/
#   tag       Release tag (e.g. v0.1.0). Matches releases/<baseline>/<tag>/
#
# Options:
#   --dry-run   Print the gh release create command without executing it.
#   --draft     Publish as a draft release (passthrough to gh release create).
#
# Requirements:
#   - gh CLI must be installed and authenticated (gh auth login)
#   - jq must be installed (or falls back to grep/sed for metadata extraction)
#   - Release directory must exist and contain model.pt, config.yaml, release.json
#
# Docs: https://cli.github.com/manual/gh_release_create
set -euo pipefail

# ---------------------------------------------------------------------------
# Usage / help
# ---------------------------------------------------------------------------

_usage() {
    grep '^#' "$0" | sed 's/^# \?//'
    exit 0
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    _usage
fi

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

if [[ $# -lt 2 ]]; then
    echo "ERROR: Missing required arguments." >&2
    echo "Usage: $0 <baseline> <tag> [--dry-run] [--draft]" >&2
    exit 1
fi

BASELINE="$1"
TAG="$2"
shift 2

DRY_RUN=0
DRAFT_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --draft)
            DRAFT_FLAG="--draft"
            shift
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Verify release directory
# ---------------------------------------------------------------------------

RELEASE_DIR="releases/${BASELINE}/${TAG}"

if [[ ! -d "${RELEASE_DIR}" ]]; then
    echo "ERROR: Release directory not found: ${RELEASE_DIR}" >&2
    echo "Run: python scripts/export.py --baseline ${BASELINE} --tag ${TAG} ..." >&2
    exit 1
fi

for required_file in model.pt config.yaml release.json; do
    if [[ ! -f "${RELEASE_DIR}/${required_file}" ]]; then
        echo "ERROR: Required file missing from release dir: ${RELEASE_DIR}/${required_file}" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Extract metadata from release.json
# ---------------------------------------------------------------------------

RELEASE_JSON="${RELEASE_DIR}/release.json"

if command -v jq &>/dev/null; then
    BEST_CER=$(jq -r '.best_val_cer' "${RELEASE_JSON}")
    PARAMS_COUNT=$(jq -r '.params_count' "${RELEASE_JSON}")
    GIT_COMMIT_META=$(jq -r '.git_commit' "${RELEASE_JSON}")
else
    echo "WARN: jq not found — falling back to grep/sed for metadata extraction" >&2
    BEST_CER=$(grep -o '"best_val_cer"[[:space:]]*:[[:space:]]*[0-9.e+-]*' "${RELEASE_JSON}" \
        | sed 's/.*:[[:space:]]*//')
    PARAMS_COUNT=$(grep -o '"params_count"[[:space:]]*:[[:space:]]*[0-9]*' "${RELEASE_JSON}" \
        | sed 's/.*:[[:space:]]*//')
    GIT_COMMIT_META=$(grep -o '"git_commit"[[:space:]]*:[[:space:]]*"[^"]*"' "${RELEASE_JSON}" \
        | sed 's/.*":\s*"\(.*\)"/\1/')
fi

# ---------------------------------------------------------------------------
# Build release title and notes path
# ---------------------------------------------------------------------------

GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
RELEASE_TITLE="GP1 ${BASELINE} ${TAG} — CER ${BEST_CER} (${PARAMS_COUNT} params)"
NOTES_FILE="${RELEASE_DIR}/README.md"

if [[ ! -f "${NOTES_FILE}" ]]; then
    echo "WARN: README.md not found in ${RELEASE_DIR}; using empty notes" >&2
    NOTES_FILE=""
fi

# ---------------------------------------------------------------------------
# Build asset file list (all files in the release dir)
# ---------------------------------------------------------------------------

# Build asset list compatible with bash 3 (macOS ships bash 3; mapfile requires bash 4+)
ASSET_FILES=()
while IFS= read -r _f; do
    ASSET_FILES+=("$_f")
done < <(find "${RELEASE_DIR}" -maxdepth 1 -type f | sort)

# ---------------------------------------------------------------------------
# Assemble gh release create command
# ---------------------------------------------------------------------------

GH_CMD=(
    gh release create "${TAG}"
    "${ASSET_FILES[@]}"
    --title "${RELEASE_TITLE}"
)

if [[ -n "${NOTES_FILE}" ]]; then
    GH_CMD+=(--notes-file "${NOTES_FILE}")
fi

if [[ -n "${DRAFT_FLAG}" ]]; then
    GH_CMD+=("${DRAFT_FLAG}")
fi

# ---------------------------------------------------------------------------
# Execute or dry-run
# ---------------------------------------------------------------------------

echo "Release dir : ${RELEASE_DIR}"
echo "Tag         : ${TAG}"
echo "Baseline    : ${BASELINE}"
echo "Best CER    : ${BEST_CER}"
echo "Params      : ${PARAMS_COUNT}"
echo "Git commit  : ${GIT_COMMIT} (meta: ${GIT_COMMIT_META})"
echo "Assets      : ${#ASSET_FILES[@]} files"
echo ""

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[DRY RUN] Would execute:"
    echo "  ${GH_CMD[*]}"
    echo ""
    echo "No GitHub API call made."
    exit 0
fi

echo "Running: ${GH_CMD[*]}"
"${GH_CMD[@]}"
echo "Release published: ${TAG}"

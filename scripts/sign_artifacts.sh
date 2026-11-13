#!/usr/bin/env bash
# THME Release Artifact Cryptographic Signing Script (Milestone v0.2.0.6)
# Signs all distribution tarballs and checksum manifests using Minisign & GPG.
set -euo pipefail

DIST_DIR="${1:-dist}"

if [ ! -d "${DIST_DIR}" ]; then
    echo "Directory ${DIST_DIR} does not exist. Skipping signing."
    exit 0
fi

echo "Signing release artifacts in ${DIST_DIR}..."

# Minisign signature
if command -v minisign >/dev/null 2>&1 && [ -n "${MINISIGN_KEY:-}" ]; then
    umask 077
    printf '%s\n' "${MINISIGN_KEY}" > /tmp/minisign.key
    for f in "${DIST_DIR}"/*.tar.gz; do
        if [ -f "$f" ]; then
            echo "  [minisign] Signing $(basename "$f")..."
            minisign -Sm "$f" -s /tmp/minisign.key -t "THME Epiphanes Release Asset"
        fi
    done
    rm -f /tmp/minisign.key
else
    echo "  [minisign] Skipping (minisign CLI not installed or MINISIGN_KEY not provided)"
fi

# GPG detached signature
if command -v gpg >/dev/null 2>&1 && [ -n "${GPG_KEY:-}" ]; then
    for f in "${DIST_DIR}"/*.tar.gz; do
        if [ -f "$f" ]; then
            echo "  [gpg] Signing $(basename "$f")..."
            gpg --batch --yes --detach-sign --armor "$f"
        fi
    done
else
    echo "  [gpg] Skipping (gpg CLI not installed or GPG_KEY not provided)"
fi

echo "Signing complete."

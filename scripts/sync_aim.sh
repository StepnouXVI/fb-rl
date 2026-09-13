#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VPS_HOST="${AIM_VPS_HOST:-vps}"
VPS_DEST_DIR="${AIM_VPS_DEST_DIR:-/home/stm/services/aim_data}"
CONTAINER_NAME="${AIM_CONTAINER_NAME:-aim-server}"

if [ ! -d "results/aim" ]; then
    echo "No local Aim repository found at results/aim! Nothing to sync."
    exit 0
fi

SYNC_ID="sync_$(date +%s)"
VPS_INCOMING="${VPS_DEST_DIR}/incoming_${SYNC_ID}"

echo "================================================================="
echo "  [1/2] Syncing local results/aim to ${VPS_HOST}:${VPS_INCOMING}"
echo "================================================================="
rsync -avz results/aim/ "${VPS_HOST}:${VPS_INCOMING}/"

echo "================================================================="
echo "  [2/2] Ingesting runs into central Aim server container"
echo "================================================================="
ssh "${VPS_HOST}" "docker exec ${CONTAINER_NAME} aim runs --repo /data/incoming_${SYNC_ID} cp --destination /data && rm -rf ${VPS_INCOMING}"

echo "================================================================="
echo "  Sync completed successfully! Dashboard: https://aim.stepnou-xvi.ru"
echo "================================================================="

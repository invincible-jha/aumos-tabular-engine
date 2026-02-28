#!/usr/bin/env bash
# AumOS Tabular Engine — Demo Script
# Submits a generation job, polls for completion, and prints the result URI.
#
# Usage: ./demo-data/demo-run.sh

set -euo pipefail

BASE_URL="${AUMOS_BASE_URL:-http://localhost:8006}"

echo "Submitting synthetic data generation job..."

RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/generation/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "source_uri": "file://demo-data/customers.csv",
    "model": "ctgan",
    "num_rows": 1000,
    "output_format": "parquet"
  }')

JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "Job created: $JOB_ID"

echo "Polling for completion..."
for i in $(seq 1 60); do
  STATUS_RESPONSE=$(curl -s "${BASE_URL}/api/v1/generation/jobs/${JOB_ID}")
  STATUS=$(echo "$STATUS_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")

  echo "  Status: $STATUS (attempt $i/60)"

  if [[ "$STATUS" == "complete" ]]; then
    OUTPUT_URI=$(echo "$STATUS_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('output_uri', 'N/A'))")
    FIDELITY=$(echo "$STATUS_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('fidelity_score', 'N/A'))")
    echo ""
    echo "SUCCESS!"
    echo "  Output URI: $OUTPUT_URI"
    echo "  Fidelity Score: $FIDELITY"
    exit 0
  elif [[ "$STATUS" == "failed" ]]; then
    ERROR=$(echo "$STATUS_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error_message', 'unknown error'))")
    echo "FAILED: $ERROR"
    exit 1
  fi

  sleep 5
done

echo "Timed out after 5 minutes"
exit 1

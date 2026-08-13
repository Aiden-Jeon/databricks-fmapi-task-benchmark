#!/usr/bin/env bash
# kmle-bench one-time workspace bootstrap. Idempotent. Reads config.json.
# Prereqs: databricks CLI authenticated to the target profile; packs built
# locally via `python harness/prepare.py` (see README §Data).
set -euo pipefail
cd "$(dirname "$0")"

CFG=config.json
P=$(jq -r .profile $CFG); CAT=$(jq -r .catalog $CFG)
SP=$(jq -r .schemas.packs $CFG); SPR=$(jq -r .schemas.private $CFG); SR=$(jq -r .schemas.results $CFG)
VP=$(jq -r .volumes.packs $CFG); VPR=$(jq -r .volumes.private $CFG); VR=$(jq -r .volumes.results $CFG)
SCOPE=$(jq -r .secret_scope $CFG); KEY=$(jq -r .secret_key $CFG)
WDIR=$(jq -r .workspace_runner_dir $CFG)

echo "== auth check =="
databricks current-user me -p "$P" >/dev/null && echo "auth OK ($P)"

echo "== schemas + volumes =="
for s in "$SP" "$SPR" "$SR"; do databricks schemas create "$s" "$CAT" -p "$P" 2>/dev/null || true; done
databricks volumes create "$CAT" "$SP" "$VP" MANAGED -p "$P" 2>/dev/null || true
databricks volumes create "$CAT" "$SPR" "$VPR" MANAGED -p "$P" 2>/dev/null || true
databricks volumes create "$CAT" "$SR" "$VR" MANAGED -p "$P" 2>/dev/null || true
echo "volumes ready under $CAT"

echo "== secret scope ($SCOPE/$KEY) =="
databricks secrets create-scope "$SCOPE" -p "$P" 2>/dev/null || true
if ! databricks secrets list-secrets "$SCOPE" -p "$P" 2>/dev/null | grep -q "$KEY"; then
  echo "Creating a workspace PAT (14d) and storing as $SCOPE/$KEY"
  TOK=$(databricks tokens create --comment kmle-bench --lifetime-seconds 1209600 -p "$P" --output json | jq -r .token_value)
  databricks secrets put-secret "$SCOPE" "$KEY" --string-value "$TOK" -p "$P"
fi

echo "== upload packs / private / kickoff =="
databricks fs cp -r --overwrite packs "dbfs:/Volumes/$CAT/$SP/$VP" -p "$P"
databricks fs mkdir "dbfs:/Volumes/$CAT/$SP/$VP/_harness" -p "$P" 2>/dev/null || true
databricks fs cp harness/kickoff_prompt_ko.md "dbfs:/Volumes/$CAT/$SP/$VP/_harness/kickoff_prompt_ko.md" --overwrite -p "$P"
databricks fs cp -r --overwrite private "dbfs:/Volumes/$CAT/$SPR/$VPR" -p "$P"

echo "== import runner =="
databricks workspace mkdirs "$WDIR" -p "$P"
databricks workspace import "$WDIR/runner.py" --file harness/runner.py --format RAW --overwrite -p "$P"

echo "SETUP COMPLETE. Next: python harness/submit_matrix.py --mode smoke  (validate), then --mode full"

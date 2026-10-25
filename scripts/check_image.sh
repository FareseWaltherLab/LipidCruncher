#!/usr/bin/env bash
# Build the deployable image from the committed code and check, inside it,
# that the AI Studio sandbox's protections hold and the app answers its
# health check as the image's non-root user.
#
# The sandbox's security depends on the image (non-root USER, rlimits,
# seccomp on Linux x86_64), so tests/unit/test_sandbox_isolation.py only
# proves it on the machine it runs on. This runs it where it is deployed.
# Run it before every deploy; push the image it built and checked.
#
#   scripts/check_image.sh [image-tag]     (default lipidcruncher-check:<commit>)
#
# Builds from `git archive HEAD` (never the working tree), linux/amd64, with
# R (WITH_R=1). Exits non-zero, saying why, if anything fails or any
# isolation test is skipped (a skip means a protection was not checked).
# The stable-isotope-tracing tests that need R + IsoCorrectoR run here too:
# this is the only place in CI that has them, so a skip fails as well.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
commit=$(git rev-parse --short HEAD)
image=${1:-lipidcruncher-check:$commit}
platform=linux/amd64

fail() { echo "CHECK FAILED: $*" >&2; exit 1; }

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Note: uncommitted changes are not in the image; it is built from $commit."
fi

echo "== Building $image from $commit ($platform, WITH_R=1)"
git archive --format=tar HEAD |
    docker build --platform "$platform" --build-arg WITH_R=1 -t "$image" -

user=$(docker image inspect -f '{{.Config.User}}' "$image")
case "$user" in
    ''|root|0|0:*) fail "the image runs as '${user:-root}', not a non-root user" ;;
esac
echo "== Image user: $user"

echo "== Sandbox isolation and R-dependent isotope tests inside the image"
# pytest is a dev tool, so it goes into this throwaway container only.
log=$(mktemp)
if ! docker run --rm --platform "$platform" -e PYTHONPATH=src \
        --entrypoint sh "$image" -c '
        pip install --quiet --user --no-warn-script-location pytest==9.0.3 &&
        python -m pytest -q -rs -p no:cacheprovider \
            tests/unit/test_sandbox_isolation.py \
            tests/integration/test_isotope_tracing_golden.py \
            tests/integration/test_isotope_tracing_warnings.py
    ' 2>&1 | tee "$log"; then
    fail "in-image tests failed (see above)"
fi
grep -q " passed" "$log" || fail "the in-image tests did not run"
if grep -qiE "[0-9]+ skipped" "$log"; then
    fail "in-image tests were skipped: a sandbox protection or the R runtime was not checked"
fi
rm -f "$log"

echo "== App health check"
container=$(docker run -d --platform "$platform" -p 127.0.0.1::8501 "$image")
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
port=$(docker port "$container" 8501/tcp | head -1 | sed 's/.*://')
healthy=""
for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
        healthy=1
        break
    fi
    sleep 1
done
[ -n "$healthy" ] || { docker logs "$container" | tail -20; fail "/healthz did not answer within 60 s"; }
app_uid=$(docker exec "$container" stat -c %u /proc/1)
[ "$app_uid" != "0" ] || fail "the app process runs as root"
echo "== /healthz answered; the app runs as uid $app_uid"

echo "CHECK PASSED: $image (commit $commit)"

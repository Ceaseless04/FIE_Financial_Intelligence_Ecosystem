#!/bin/sh
# Nightly logical backup of the platform database.
#
# Runs in the production compose stack. Credentials arrive as PG* environment
# variables; nothing is read from a file in the image.
#
# This covers logical backups only. Point-in-time recovery (WAL archiving) and
# restore rehearsal are Phase 9 work — a backup that has never been restored is
# not a backup.

set -eu

BACKUP_DIR="${BACKUP_DIR:-/backups}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-14}"
INTERVAL_SECONDS="${BACKUP_INTERVAL_SECONDS:-86400}"

log() {
    printf '{"event":"%s","timestamp":"%s","message":"%s"}\n' \
        "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$2"
}

run_backup() {
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    target="${BACKUP_DIR}/${PGDATABASE}-${timestamp}.dump"

    log backup_started "writing ${target}"

    # Custom format so pg_restore can do selective and parallel restores.
    if pg_dump --format=custom --compress=9 --file="${target}" ; then
        # Verify the dump is readable before trusting it. A dump that cannot be
        # listed is not a backup, and finding that out during an incident is
        # too late.
        if pg_restore --list "${target}" > /dev/null 2>&1; then
            log backup_succeeded "${target} ($(wc -c < "${target}") bytes)"
        else
            log backup_corrupt "${target} failed verification"
            rm -f "${target}"
            return 1
        fi
    else
        log backup_failed "pg_dump exited non-zero"
        return 1
    fi

    deleted="$(find "${BACKUP_DIR}" -name "${PGDATABASE}-*.dump" \
        -mtime "+${RETENTION_DAYS}" -print -delete | wc -l)"
    log backup_pruned "removed ${deleted} backup(s) older than ${RETENTION_DAYS} days"
}

mkdir -p "${BACKUP_DIR}"
log backup_service_started "interval=${INTERVAL_SECONDS}s retention=${RETENTION_DAYS}d"

while true; do
    # A failed backup must not kill the loop — the next attempt should still run,
    # and the failure is visible in the logs for alerting.
    run_backup || log backup_cycle_failed "continuing to next cycle"
    sleep "${INTERVAL_SECONDS}"
done

"""Scheduler entrypoint — `python -m src.scheduler.runner`.

Phase 6 will populate this with APScheduler jobs. For Phase 1 it is a
no-op loop so `docker compose up` and `bash run.sh` succeed.
"""

from __future__ import annotations

import logging
import signal
import time

log = logging.getLogger("scheduler")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log.info("scheduler scaffold up; no jobs registered yet (Phase 6 to come)")

    stop = False

    def _handle(signum, _frame):  # noqa: ANN001
        nonlocal stop
        log.info("received signal %s, exiting", signum)
        stop = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    while not stop:
        time.sleep(1)


if __name__ == "__main__":
    main()

"""coldchain CLI.

coldchain migrate                 apply DB migrations
coldchain ingest-telemetry CSV    load fleet telemetry
coldchain ingest-sop DIR          (re)index SOP documents, incremental
coldchain detect-alerts           run the SOP rule engine over all readings
coldchain seed                    all of the above with the bundled sample data
coldchain serve                   run the API
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from coldchain.config import get_settings
from coldchain.logging import configure_logging

# repo/app root: holds alembic.ini, migrations/ and data/ (WORKDIR in the container)
ROOT = Path(os.getenv("COLDCHAIN_ROOT", Path.cwd()))
DEFAULT_CSV = ROOT / "data" / "raw" / "dynamic_supply_chain_logistics_dataset.csv"
DEFAULT_SOP_DIR = ROOT / "data" / "policy"


def migrate() -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    command.upgrade(cfg, "head")


def main(argv: list[str] | None = None) -> None:
    s = get_settings()
    configure_logging(s.log_level, json=False)

    p = argparse.ArgumentParser(prog="coldchain")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate")
    t = sub.add_parser("ingest-telemetry")
    t.add_argument("csv", type=Path, nargs="?", default=DEFAULT_CSV)
    d = sub.add_parser("ingest-sop")
    d.add_argument("directory", type=Path, nargs="?", default=DEFAULT_SOP_DIR)
    sub.add_parser("detect-alerts")
    sub.add_parser("seed")
    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--workers", type=int, default=2)
    args = p.parse_args(argv)

    if args.cmd == "migrate":
        migrate()
    elif args.cmd == "ingest-telemetry":
        from coldchain.ingestion.telemetry import ingest_telemetry

        ingest_telemetry(args.csv)
    elif args.cmd == "ingest-sop":
        from coldchain.ingestion.sop import ingest_sop_dir
        from coldchain.rag.embeddings import build_embedder

        print(json.dumps(ingest_sop_dir(args.directory, build_embedder(s))))
    elif args.cmd == "detect-alerts":
        from coldchain.domain.detection import run_detection

        print(json.dumps(run_detection()))
    elif args.cmd == "seed":
        from coldchain.domain.detection import run_detection
        from coldchain.ingestion.sop import ingest_sop_dir
        from coldchain.ingestion.telemetry import ingest_telemetry
        from coldchain.rag.embeddings import build_embedder

        migrate()
        ingest_telemetry(DEFAULT_CSV)
        ingest_sop_dir(DEFAULT_SOP_DIR, build_embedder(s))
        print(json.dumps(run_detection()))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run(
            "coldchain.api.app:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            proxy_headers=True,
        )


if __name__ == "__main__":
    main()

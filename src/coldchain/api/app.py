"""ASGI entrypoint: `uvicorn coldchain.api.app:app`."""

from coldchain.api.main import create_app

app = create_app()

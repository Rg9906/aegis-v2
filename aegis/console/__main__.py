"""
Start the live mission console: `python -m aegis.console [--port 8765] [--arch resnet18] [--open]`.

Serves the API (and the built frontend from frontend/dist, if present) on http://127.0.0.1:PORT.
For frontend development run `npm run dev` in frontend/ instead; Vite proxies /api to this port.
"""

import argparse
import os
import threading
import webbrowser


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v2"])
    p.add_argument("--threads", type=int, default=0, help="PyTorch CPU threads (default: aegis/runtime.py)")
    p.add_argument("--open", action="store_true", help="open the console in the default browser")
    args = p.parse_args()

    # Cached EuroSAT weights load offline; never block startup on the network
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import uvicorn

    from aegis.console.server import FRONTEND_DIST, create_app
    from aegis.runtime import configure_torch_threads

    threads = configure_torch_threads(args.threads)
    url = f"http://{args.host}:{args.port}"
    has_ui = os.path.isdir(FRONTEND_DIST)
    print(f"Aegis-v2 console engine on {url} ({threads} torch threads)")
    print(f"  UI: {url}/#/console" if has_ui else
          "  UI: not built; run `npm run build` in frontend/ (or `npm run dev` for http://localhost:5173)")
    if args.open and has_ui:
        threading.Timer(1.5, lambda: webbrowser.open(f"{url}/#/console")).start()
    uvicorn.run(create_app(arch=args.arch), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

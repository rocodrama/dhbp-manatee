from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Manatee Trace GUI.")
    parser.add_argument("--host", default=None, help="Server host passed to Gradio.")
    parser.add_argument("--port", type=int, default=None, help="Server port passed to Gradio.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from main.app.app import build_demo

    demo = build_demo()
    demo.queue()
    launch_kwargs = {}
    if args.host:
        launch_kwargs["server_name"] = args.host
    if args.port:
        launch_kwargs["server_port"] = args.port
    if args.share:
        launch_kwargs["share"] = True
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()

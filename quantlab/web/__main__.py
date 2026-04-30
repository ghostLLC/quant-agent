"""Allow running the dashboard via: python -m quantlab.web.app"""
import runpy
import sys

if __name__ == "__main__":
    # Remove __main__ from sys.argv so app.main() doesn't re-parse it
    if len(sys.argv) > 1 and sys.argv[0].endswith("__main__.py"):
        sys.argv.pop(0)
    runpy.run_module("quantlab.web.app", run_name="__main__")

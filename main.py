"""Compatibility shim: ``python main.py`` still works. The application lives in ``space_video_enhancer``."""

from space_video_enhancer.app import main

if __name__ == "__main__":
    raise SystemExit(main())

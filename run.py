"""啟動 Flask 審核介面。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ekr.logging_setup import setup_logging  # noqa: E402
from ekr.web.app import create_app  # noqa: E402

if __name__ == "__main__":
    setup_logging()
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

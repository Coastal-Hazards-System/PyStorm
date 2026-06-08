"""reduced_tc_suite — standalone preprocessing entry point.

Author / POC : Norberto C. Nadal-Caraballo, PhD  <norberto.c.nadal-caraballo@usace.army.mil>

Builds the canonical ``tc_data.h5`` store from the raw .mat / .csv source
files using the dataset, raw-filename, and MATLAB-variable settings
defined at the top of ``run_reduced_tc_suite.py``. That launcher is the
single source of truth for these values; this script just exposes a
standalone entry point so the preprocessor can be invoked without running
the full RTCS selection workflow.

To configure: edit ``DATASET``, ``RAW_FILES``, and ``PREPROCESS_METADATA``
in ``run_reduced_tc_suite.py``.

Usage
-----
  python scripts/preprocess.py
      Run the preprocessor with the launcher's settings.

  python scripts/preprocess.py --generate-config
      Print an annotated YAML config template and exit.

  python scripts/preprocess.py --validate <H5_FILE>
      Validate an existing store and exit.

  python scripts/preprocess.py --export-csv <H5_FILE> [--export-dir DIR]
      Export /X /Y /HC to CSV files and exit.
"""

import argparse
import sys
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
_PKG_PATH    = _MODULE_ROOT / "backend" / "python"
if str(_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(_PKG_PATH))
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))


def main():
    # noinspection PyUnresolvedReferences
    from reduced_tc_suite.io.store import export_to_csv
    # noinspection PyUnresolvedReferences
    from reduced_tc_suite.workflows.ingest import Preprocessor
    # noinspection PyUnresolvedReferences
    from reduced_tc_suite.config.loader import PREPROCESS_CONFIG_YAML

    parser = argparse.ArgumentParser(prog="preprocess.py")
    parser.add_argument("--generate-config", action="store_true")
    parser.add_argument("--validate",   metavar="H5_FILE")
    parser.add_argument("--export-csv", metavar="H5_FILE")
    parser.add_argument("--export-dir", metavar="DIR", default=None)
    args = parser.parse_args()

    if args.generate_config:
        print(PREPROCESS_CONFIG_YAML); return

    if args.validate:
        Preprocessor.validate(Path(args.validate)); return

    if args.export_csv:
        export_to_csv(Path(args.export_csv),
                      Path(args.export_dir) if args.export_dir else None)
        return

    # Pull DATASET, RAW_FILES, PREPROCESS_METADATA from the launcher so both
    # entry points share one source of truth.
    # noinspection PyUnresolvedReferences
    from run_reduced_tc_suite import _build_preprocess_config, DATASET
    print(f"\n[preprocess] Active dataset: {DATASET}")
    Preprocessor(_build_preprocess_config()).run()


if __name__ == "__main__":
    main()

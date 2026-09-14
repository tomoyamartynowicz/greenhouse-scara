from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenhouse_scara_3d_common.runtime import main

if __name__ == "__main__":
    main(Path(__file__).with_name("config.yaml"))

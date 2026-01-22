# run_main.py
import sys
import os

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from REC.main import main 

if __name__ == "__main__":
    main()

1. Install system dependencies:
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgtk-3-0
```

3. Clone or open the project folder, сreate and activate a virtual environment, install Python dependencies
```bash
cd MNIST_TEST
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Dataset preview pipeline:
   (Make sure you know the dataset annotation format and that the --format value exactly matches the format name registered in `Parsers/__init__.py`.)
   
  - Put the dataset into project root
  - Run:
    ```bash
    python ./scripts/Preview_module/main.py --format yolo --every 1 --max 30
    ```

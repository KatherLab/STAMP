# STAMP protocol
A protocol for Solid Tumor Associative Modeling in Pathology

## Installation
First, install OpenSlide using either the command below or the [official installation instructions](https://openslide.org/download/#distribution-packages):
```bash
apt update && apt install -y openslide-tools libgl1-mesa-glx # libgl1-mesa-glx is needed for OpenCV
```

Then, run the following commands (NOTE: will be updated once this is a PyPI package):
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Finally, to download required resources such as the weights of the CTransPath feature extractor, run the following command:
```bash
python -m stamp setup
```
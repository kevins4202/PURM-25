# Install
`git clone https://github.com/kevins4202/PURM-25.git && cd PURM-25`

# Create conda environment
```
conda create --name purm python=3.11
conda activate purm
conda config --append conda-forge services
conda install nvidia/label/cuda-12.6.3::cuda-toolkit
pip3 install torch torchvision torchaudio accelerate bitsandbytes
conda install --file requirements.txt
```

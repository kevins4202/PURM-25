# Install
`git clone https://github.com/kevins4202/PURM-25.git && cd PURM-25`

# Create conda environment
```
conda create --name purm python=3.13.1
conda activate purm
conda install --file requirements.txt
conda install conda-forge::pydantic
conda install services::mistral_common
conda install conda-forge::pillow
conda install conda-forge::transformers
conda install conda-forge::jsonschema
conda install conda-forge::sentencepiece
conda install conda-forge::tiktoken
conda install conda-forge::huggingface_hub
```

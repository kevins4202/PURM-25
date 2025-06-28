# Install
`git clone https://github.com/kevins4202/PURM-25.git && cd PURM-25`

# Create conda environment
```
conda create --name purm python=3.13.1
conda activate purm
conda install --file requirements.txt 
conda install conda-forge::pydantic services::mistral_common conda-forge::pillow conda-forge::transformers conda-forge::jsonschema conda-forge::sentencepiece conda-forge::tiktoken conda-forge::huggingface_hub conda-forge::accelerate
```
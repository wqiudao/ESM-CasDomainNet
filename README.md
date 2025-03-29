# ESM-CasDomainNet
ESMCasDomainNet mode is available on huggingface.
https://huggingface.co/wqiudao/ESMCasDomainNet_v0.1

GPU: CUDA-compatible GPU with more than 5GB VRAM
```
conda create -n esm2 python=3.9  -y
conda activate esm2 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib
pip install fair-esm 
conda install wqiudao::esmdomainpredictor -y
```




python ESMDomainPredictor.py  --use_gpu   ESMCasDomainNet_v0.1.pth Cas9D.fa

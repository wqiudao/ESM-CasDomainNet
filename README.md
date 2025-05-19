# ESM-CasDomainNet
ESMCasDomainNet mode is available on huggingface.
https://huggingface.co/wqiudao/ESMCasDomainNet_v0.1

GPU: CUDA-compatible GPU with more than 5GB VRAM
```
conda create -n esm2 python=3.9  -y
conda activate esm2 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121 


pip install matplotlib
pip install fair-esm 

```




python ESMDomainPredictor.py  --use_gpu   ESMCasDomainNet_v0.1.pth Cas9D.fa


To facilitate broad access, we deployed ESMCasDomainNet as an interactive web server at https://www.esmdomain.com ,

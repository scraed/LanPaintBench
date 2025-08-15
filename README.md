# LanPaint Benchmark
This is the repo for academic benchmark of [LanPaint](https://github.com/scraed/LanPaint)

## Prerequisites

- Python 3.9 or higher
- A Linux-based operating system (e.g., Ubuntu)
- `make` utility for build automation

## Setup Instructions

1. **Create a Python Virtual Environment**  
   Create and activate a new Python virtual environment to manage dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install System Dependencies**  
   Ensure `make` is installed on your Linux system. Run the following commands:
   ```bash
   sudo apt update
   sudo apt install make
   ```

3. **Install Python Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Pre-trained Model**  
   Download the pre-trained model checkpoint from the following link:  
   [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)  
   Place the downloaded file in the `./checkpoints` directory. 

5. **Run the Benchmark**  
   Execute the benchmark using the provided `make` command:
   ```bash
   make run_imagenet
   ```

6. **Check Results**  
   Metrics like LPIPS and FID will be output into `./results` as a CSV file.
   
## Citation

```
@misc{zheng2025lanpainttrainingfreediffusioninpainting,
      title={Lanpaint: Training-Free Diffusion Inpainting with Exact and Fast Conditional Inference}, 
      author={Candi Zheng and Yuan Lan and Yang Wang},
      year={2025},
      eprint={2502.03491},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2502.03491}, 
}
```







# RoSteALS+ViT

RoSteALS+ViT replaces the ResNet-50 decoder of [RoSteALS](https://github.com/TuBui/RoSteALS) with a Vision Transformer, improving robustness against blur, compression, noise, rescaling, and masking.

---

## Requirements
```bash
pip install -r requirements.txt
```

## Checkpoint

Download the model checkpoint [here](https://drive.google.com/file/d/1Ip3uNTVOklpUw51A1XVK6QJK7H6vl5Oy/view?usp=sharing) and place it at models/.
## Inference
```bash
python inference.py \
  --config models/VQ4_mir_inference_vit.yaml \
  --weight models/<checkpoint>.ckpt \
  --secret "secrets" \
  --cover examples/image.png \
  --output stego.png
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--config` | `models/VQ4_mir_inference_vit.yaml` | Path to config file |
| `--weight` | — | Path to checkpoint file |
| `--secret` | `secrets` | Secret message (7 characters max) |
| `--cover` | `examples/934.jpg` | Path to cover image |
| `--output` | `stego.png` | Path to output stego image |
| `--image_size` | `256` | Image resolution |

The script embeds the secret into the cover image and saves the watermarked result. It also prints image quality metrics (PSNR, SSIM, LPIPS) and the recovered bit accuracy.

---

## Results

Bit accuracy on 100 randomly sampled images from the CLIC dataset, compared to the original RoSteALS.

| Attack Category | RoSteALS | RoSteALS+ViT |
|---|---|---|
| Blur (Average) | 0.9356 | **0.9983** |
| Blur (Gaussian) | 0.9322 | **0.9983** |
| Brightness | 0.9332 | **0.9877** |
| Contrast | 0.8901 | **0.8907** |
| Crop (Center) | **0.7631** | 0.7465 |
| Crop (Random) | **0.7607** | 0.7431 |
| Downscale (Bicubic) | 0.8871 | **0.9089** |
| Downscale (Bilinear) | 0.8880 | **0.9086** |
| Downscale (Lanczos4) | 0.8870 | **0.9101** |
| Down+Up (Bicubic) | 0.9128 | **0.9746** |
| Down+Up (Bilinear) | 0.9252 | **0.9823** |
| Down+Up (Lanczos4) | 0.9097 | **0.9713** |
| Upscale (Bicubic) | 0.9403 | **1.0000** |
| Upscale (Bilinear) | 0.9406 | **1.0000** |
| Upscale (Lanczos4) | 0.9411 | **0.9987** |
| Gamma | 0.9311 | **0.9874** |
| Masking (2.5%) | 0.9403 | **0.9975** |
| Masking (5%) | 0.9359 | **0.9974** |
| Masking (10%) | 0.9336 | **0.9941** |
| Masking (15%) | 0.9235 | **0.9917** |
| Masking (20%) | 0.9175 | **0.9873** |
| Overlay Shapes | 0.9239 | **0.9859** |
| Salt & Pepper (s=0.2) | 0.9146 | **0.9795** |
| Salt & Pepper (s=0.5) | 0.9116 | **0.9730** |
| Salt & Pepper (s=0.8) | 0.9057 | **0.9716** |
| JPEG Compression | 0.9381 | **0.9969** |
| Rotation | **0.7726** | 0.7629 |
| Saturation | 0.9285 | **0.9661** |
| Gaussian Noise | 0.9397 | **0.9985** |
| **Overall Mean** | 0.9056 | **0.9520** |

---

## Acknowledgements

This work builds upon RoSteALS by Bui et al., licensed under CC BY-NC 4.0.

## Citation

If you use this work, please cite:
```bibtex
@inproceedings{bui2023rosteals,
  title={RoSteALS: Robust Steganography using Autoencoder Latent Space},
  author={Bui, Tu and Agarwal, Shruti and Yu, Ning and Collomosse, John},
  booktitle={Proc. CVPR Workshop},
  year={2023}
}
```

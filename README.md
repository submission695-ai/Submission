# EMPalm: Exfiltrating Palm Biometric Data via Electromagnetic Side-Channels
EMPalm is an EM side-channel reconstruction framework for palm biometrics. It passively captures EM emissions during scanner operation. 

We adapted [TempestSDR](https://github.com/martinmarinov/TempestSDR) and [TempestSDR_EMEye](https://github.com/longyan97/EMEye_Tutorial?tab=readme-ov-file) into [TempestSDR_EMPalm](https://github.com/submission695-ai/Submission) and augmented it with band localization and multiband combination to robustly recover bitstreams from EM captures. After converting the bitstreams into raw images, we employ conditional DiffPIR to restore fine palmprint and palm-vein textures, then evaluate the reconstructed images as spoofing inputs against mainstream recognition models.

## TempestSDR_EMPalm
### Hardware
- SDR (Software-defined Radio): USRP B205
- LNA (Low Noise Amplifier, 40dB)
### Software
- TempestSDR_EMPalm

## Diffusion for Palm image
After reconstructing raw palm images from EM bitstreams, the results are usually noisy, blurry, and missing fine details. To enhance these reconstructions, we leverage diffusion-based restoration models ([DiffPIR](https://github.com/yuanzhi-zhu/DiffPIR)).
-	The raw EM-reconstructed image is used as the degraded input.
- The diffusion model runs an iterative denoising process, gradually refining the image toward a clean, detailed palm image.
-	This restores ridge/vein textures that are critical for biometric matching.

This step bridges the gap between bit-level reconstruction and visually/forensically useful palm images, enabling reliable spoofing evaluation on mainstream recognition models.

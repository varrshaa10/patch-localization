# Citations and References

This project implements a classical computer vision approach to patch localization in synthetic SEM images without machine learning or deep learning models.

## Augmentation and Noise Model

The synthetic dataset generation incorporates physically-inspired noise and augmentation techniques based on scanning electron microscopy (SEM) principles:

### 1. Shot Noise in SEM Imaging
**Reimer, L.** "Scanning Electron Microscopy: Physics of Image Formation and Microanalysis", Springer, 1998.
- **Application**: Justifies the Poisson noise model used in `algorithm/dataset/generate_dataset.py`'s `add_noise()` function to simulate shot noise inherent in SEM image acquisition.

### 2. Edge Brightening and Charging Effects
**Goldstein, D. J., Newbury, D. E., Michael, J. R., Ritchie, N. W. M., and Scott, J. H. J.** "Scanning Electron Microscopy and X-Ray Microanalysis", Springer, 2018.
- **Application**: Justifies the edge-brightening augmentation in `algorithm/dataset/generate_dataset.py`'s `brighten_edges()` function, which models the charging effect and secondary electron collection near material boundaries in SEM imaging.

### 3. DRAM and FinFET Layout Geometry
**Kang, S. M. and Leblebici, Y.** "CMOS Digital Integrated Circuits: Analysis and Design", McGraw-Hill, 3rd Edition.
- **Application**: Basis for the synthetic DRAM (checkerboard) and FinFET (stripe pattern) geometries used in `algorithm/dataset/generate_dataset.py` to create realistic semiconductor microstructure patterns.

## Confidence Metric and Ambiguity Detection

### Lowe's Ratio Test for Feature Matching Confidence
**Lowe, D. G.** "Distinctive Image Features from Scale-Invariant Keypoints", *International Journal of Computer Vision*, 60(2), 91–110, 2004.
- **Application**: The ambiguity-ratio confidence detection in `algorithm/core/infer.py`'s `multi_peak_search()` function is inspired by Lowe's ratio test from SIFT. The algorithm compares the best-scoring match against the second-best distinct spatial cluster (similarity ratio). A ratio near 1.0 indicates ambiguous/low-confidence matches (multiple indistinguishable locations), while a ratio further from 1.0 indicates confident matches. This technique allows the matcher to flag failure cases due to periodic pattern ambiguity without training-based learning.

---

## Classical (Non-Deep-Learning) Approach

This project uses normalized cross-correlation (NCC) template matching with multi-scale and multi-rotation search. No neural networks, training data, or model weights are involved. This approach is suitable for the problem statement's requirement to handle localization on microstructure images where:
- Ground truth is perfectly known (synthetic data)
- Patterns may be periodic (creating the core challenge)
- Quick inference without GPU is required

---

## How the Literature Informs the Synthetic Augmentations

The synthetic image generator uses physically motivated effects to make the
navigation/localization task less idealized. These effects are simulation
choices rather than claims that the generated images reproduce a particular
SEM instrument exactly.

| Effect | Implementation in this project | Motivation |
|---|---|---|
| Independent sensor/statistical noise | Poisson + Gaussian components applied independently to reference and search images | SEM imaging is based on electron detection and image acquisition, so separate captures can contain different measurement noise |
| Charging / contrast variation | Edge-dependent intensity enhancement | SEM charging can produce measurable bright/dark contrast variations and can reveal or obscure structures |
| 10× field-of-view difference | 10,000×10,000 continuous layout downsampled to 1,000×1,000 | SEM magnification determines the physical area represented by each image pixel |
| Small rotation | Reference rotated within ±3° | Models small acquisition/orientation differences between independent captures |
| Feature/layout jitter | Small randomized feature dimensions and spacing | Semiconductor metrology explicitly studies line-edge roughness and dimensional variation |

### References

1. Zhang, H.-B., Feng, R.-J., & Ura, K. (2004).
   *Utilizing the Charging Effect in Scanning Electron Microscopy.*
   Scanning, 26(6), 258–265.
   DOI: https://doi.org/10.3184/003685004783238490

2. Bertness, K. A. (2017).
   *Dimensional Measurement of Nanostructures with Scanning Electron Microscopy.*
   NIST Special Publication 250-96.
   DOI: https://doi.org/10.6028/NIST.SP.250-96

3. Bunday, B. B., Bishop, M. R., Villarrubia, J. S., & Vladar, A.
   *CD-SEM Measurement of Line Edge Roughness Test Patterns for 193 nm Lithography.*
   NIST.

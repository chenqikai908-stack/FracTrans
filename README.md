# FracTrans
FracTrans is a novel transformer framework designed for medical hyperspectral image (MHSI) classification. It explicitly models non-stationary spatial–frequency correlations, enabling dynamic separation of fine pathological details from global anatomical structures.
# FracTrans: Fractional Fourier Transformer for Medical Hyperspectral Image Classification

**FracTrans** is a novel transformer framework designed for medical hyperspectral image (MHSI) classification. It explicitly models non-stationary spatial–frequency correlations, enabling dynamic separation of fine pathological details from global anatomical structures.

## 🔬 Motivation
- Medical hyperspectral images provide rich spatial–spectral information critical for computer-aided diagnosis.
- Existing methods struggle to jointly capture spatial–frequency correlations and fail to adaptively separate high-frequency lesion patterns from low-frequency global trends.
- Standard self-attention suffers from quadratic complexity, limiting scalability to high-dimensional MHSI data.

## 🚀 Key Innovations
- **Learnable Fractional Fourier Self-Attention**  
  Integrates a learnable Fractional Fourier Transform (FrFT) into self-attention. A learnable fractional order achieves adaptive spatial–frequency decomposition, dynamically isolating pathological details from global structures.
- **Stochastic Frequency Mode Sampling**  
  Overcomes the quadratic bottleneck of standard self-attention by randomly sampling frequency modes, significantly accelerating high-dimensional MHSI processing.
- **Hybrid Spectral–Spatial Modeling**  
  Combines sliding **Local Channel Spectral Attention** and **Global Channel Spectral Attention** to capture channel-wise spectral dependencies, together with multi-scale patch embedding and feature fusion to model spectral–spatial correlations.

## 📊 Performance
- Five-fold cross-validation on **choledoch** and **gastric precancerous lesion** datasets shows FracTrans significantly outperforms mainstream methods.
- Ablation studies demonstrate that the FrFT module alone brings a **+7.18% accuracy improvement**, validating its effectiveness.

## 💡 Applications
- Lesion detection and segmentation in medical hyperspectral imaging
- Computer-aided pathological diagnosis
- Any high-dimensional imaging task requiring joint spatial–frequency analysis

## 📖 Reference
If you find FracTrans useful, please cite our paper (details to be updated).

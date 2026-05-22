# Edge-Fog Computing-Enabled EEG Data Compression via Asymmetrical Variational Discrete Cosine Transform Network (AVDCT-Net)

A resource-efficient PyTorch-based framework designed for compressed sensing and low-latency transmission of multi-channel Electroencephalogram (EEG) signals in resource-constrained IoT, medical, and BCI Edge-Fog environments.

---

## 🌟 Project Overview
AVDCT-Net shifts the heavy computational reconstructive load to the Fog gateway while keeping the compression step at the Edge gateway extremely lightweight. By combining trainable linear projections, spectral domain sparsification (via Discrete Cosine Transform), learnable thresholding, and low-bit hybrid quantization, the framework achieves exceptional compression ratios while preserving superb signal reconstruction fidelity suitable for downstream clinical or BCI analysis.

---

## 🛠️ Key Architectural Components

```mermaid
graph TD
    %% Edge Gateway Section
    subgraph Edge Gateway [Edge Gateway - Resource Constrained]
        Input[Raw EEG Signal] --> SharedLinear[Trainable Shared Linear Layer]
        SharedLinear --> DCTMat[Discrete Cosine Transform]
        DCTMat --> Subbands[Multi-Subband Sparsification]
        Subbands --> Conv1x1[1x1 Conv1D Layer]
        Conv1x1 --> Quant[Low-Bit Quantization]
        Quant --> HybridCoding[Hybrid Coding & Zstandard Level 22]
    end

    %% Transmission
    HybridCoding -->|Compressed Payload| Network([Wireless Channel])

    %% Fog Gateway Section
    subgraph Fog Gateway [Fog Gateway - High Performance]
        Network --> ZstdDecompress[Zstd Decompression & Dequantization]
        ZstdDecompress --> IntraInter[Intra/Inter-Channel Predictor]
        IntraInter --> MHA[Multi-Head Self-Attention]
        MHA --> IDCTMat[Inverse DCT]
        IDCTMat --> FinalLinear[Trainable Final Linear Layer]
        FinalLinear --> Output[Reconstructed EEG]
    end

    style EdgeGateway fill:#f9f9f9,stroke:#1B4F72,stroke-width:2px;
    style FogGateway fill:#f9f9f9,stroke:#78281F,stroke-width:2px;
    style Network fill:#eaeded,stroke:#7f8c8d,stroke-dasharray: 5 5;
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/VaishnaviThirumala07/Edge-Fog-Computing-Enabled-EEG-Data-Compression.git
   cd Edge-Fog-Computing-Enabled-EEG-Data-Compression
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 How to Run & Reproduce

All pipeline executions, evaluations, and visualizations are self-contained within each Jupyter notebook, eliminating the need to manage external Python scripts:

*   **Bonn EEG Evaluation:** Run all cells in `BonnDataset.ipynb` to evaluate the 5 physiological subsets, generate the compression statistics, and plot performance.
*   **BCI Competition II:** Run all cells in `BCI2Dataset.ipynb` to run training loops, apply low-bit hybrid quantization, and output signal similarity charts.
*   **BCI Competition III:** Run all cells in `BCI3Dataset.ipynb` to execute cross-subject validation and analyze cognitive task motor imagery arrays under domain transfer.

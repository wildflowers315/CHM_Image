**Comparison of Methodologies: Pauls et al. (2025) vs. Pauls et al. (2024)**

---

### 🔧 **Methodological Advances in Pauls et al., 2025**

Compared to the 2024 version, the 2025 study introduces substantial methodological improvements for large-scale and **temporally explicit** canopy height estimation, with a specific emphasis on capturing **tall trees** (above 30–35 m), where previous models showed strong underestimation.

---

### 📌 **Key Differences in Methodology**

| Feature                        | **Pauls et al., 2024**                     | **Pauls et al., 2025**                                                           |
| ------------------------------ | ------------------------------------------ | -------------------------------------------------------------------------------- |
| **Model Input**                | Sentinel-1 + Sentinel-2 median composite   | Full 12-month Sentinel-2 time series + median Sentinel-1                         |
| **Temporal Dynamics**          | No time series modeling                    | Explicit 3D modeling of monthly seasonal dynamics                                |
| **Model Architecture**         | 2D U-Net with single input image           | 3D U-Net with temporal convolutions (volumetric)                                 |
| **Loss Function**              | Modified Huber loss                        | Same, but now includes spatial shift alignment to mitigate GEDI misregistration  |
| **Training Data**              | Single year (2020)                         | Multi-year (2019–2022) GEDI data                                                 |
| **Error for Tall Trees**       | Underestimated heights >30m by up to −10 m | Reduced underestimation to approx. −5 m for 35–40 m class                        |
| **Post-Processing**            | None or minimal                            | Quadratic smoothing spline to reduce temporal noise                              |
| **Geolocation Shift Handling** | Introduced static shift correction         | Retained, now more effectively leveraged through 3D spatio-temporal convolutions |

---

### 💻 **Coding-Level Enhancements (2025)**

1. **3D U-Net Architecture**

   * Uses 3D convolutions to process monthly Sentinel-2 time series:

     ```python
     # Input shape: [Batch, Channels, Time=12, Height, Width]
     nn.Conv3d(in_channels=12 * bands + radar_channels, out_channels=64, kernel_size=(3,3,3))
     ```
   * Radar (Sentinel-1) is repeated across time dimension to match the shape.

2. **Shift-Aware Supervision**

   * Incorporates per-track spatial offsets to compensate for GEDI LiDAR misalignment (±10m).
   * This corrects systematic biases during backpropagation.

3. **Data Augmentation**

   * UTM-tiled patches of 2.56×2.56 km size are sampled randomly, with only \~0.15% of pixels containing labels.
   * Dataset size: 800,000 patches (8TB).

4. **Loss Function**

   * Modified Huber loss applied only to labeled pixels:

     ```python
     def modified_huber_loss(pred, target):
         # delta = 1
         diff = pred - target
         loss = torch.where(diff.abs() < 1, 0.5 * diff ** 2, 1 * (diff.abs() - 0.5))
         return loss.mean()
     ```

5. **Temporal Smoothing**

   * A smoothing spline (quadratic, smoothing factor=5) is applied to reduce year-to-year prediction variance due to seasonal shifts and sensor noise.

---

### 🌲 **Addressing Underestimation of Tall Trees**

**Problem in 2024:** Canopy heights above 30m were systematically underestimated, likely due to:

* Loss of detail in median composites.
* Lack of temporal vegetation change cues.
* Model saturation in height prediction.

**Solutions in 2025:**

1. **Temporal Modeling:** Seasonal signals from monthly Sentinel-2 images capture growth patterns and structural cues (e.g., leaf-on/leaf-off).
2. **3D Convolutions:** Exploit temporal and spatial context, mitigating the height saturation problem.
3. **Better Supervision:** Use of GEDI rh98 metric and shift correction ensures higher alignment with true canopy peaks.
4. **Improved Training Regime:** Multi-year training and smoothing help generalize better across extremes in canopy height.

📊 **Results:**
Mean error for 35–40m trees reduced to \~−5 m from prior \~−10 m (see Figure 3 in the paper).

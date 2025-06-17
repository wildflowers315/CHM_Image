Certainly! Here's a **from-scratch coding-level description** of the **methodology used in Pauls et al., 2024**, specifically for generating a **global 10m canopy height map**, based on the full methodological detail from the paper:

---

## 📊 **1. Input Data Sources**

### Sentinel-1

* Four channels: VV & VH polarizations from ascending and descending orbits.
* **Time period**:

  * Northern hemisphere: April–October 2020
  * Southern hemisphere: October 2019–April 2020
* **Preprocessing**: Temporal median per-pixel composite.

```python
# Output: 4 channels of median values over time
channels_s1 = ['VV_asc', 'VV_desc', 'VH_asc', 'VH_desc']
```

### Sentinel-2

* 10 optical bands (BOA reflectance).
* Cloud filtering applied before temporal median:

  * Remove clouds (cloud prob > 30%)
  * Remove shadow pixels (using NIR band and sun angle)
  * Apply spatial masking (300m buffer)
* **Final step**: Per-pixel temporal median aggregation of cleaned pixels.

```python
# Custom cloud mask + pixel filtering before median
cloud_filtered_imgs = apply_cloud_mask(s2_images)
median_composite = compute_median(cloud_filtered_imgs)
```

---

## 🌲 **2. Ground Truth: GEDI LiDAR**

* **RH100 metric** used (100% waveform height).
* Filtered to exclude:

  * Daytime shots (solar elevation > 0)
  * Low-power beams
  * Quality flag ≠ 1
  * Slope > 20° (from SRTM data)

```python
valid_gedi = gedi[(gedi.beam_int > 5) &
                  (gedi.quality_flag == 1) &
                  (gedi.solar_elevation < 0) &
                  (srtm_slope < 20)]
```

---

## 🧠 **3. Model: ResNet-50 + U-Net**

### Input:

* Shape: `[batch_size, 14, 512, 512]`

  * 14 = 10 bands from Sentinel-2 + 4 bands from Sentinel-1

### Model:

* Backbone: ResNet-50
* Decoder: U-Net style upsampling
* Output: Pixel-wise height regression (1 channel)

```python
import torch.nn as nn
from torchvision.models import resnet50

class HeightUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet50(pretrained=False)
        self.decoder = UNetDecoder(...)  # custom upsampling
        self.head = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        x = self.decoder(features)
        return self.head(x)
```

---

## 📉 **4. Loss Function: Shifted Huber Loss**

### Problem:

* GEDI tracks suffer from systematic spatial shift (few pixels)

### Solution:

* During training, compute loss over a set of small shifts (within radius $r = \sqrt{2}$) and select **minimum-loss shift**.

```python
def shifted_huber_loss(pred, target_tracks, shifts=[(0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,-1)]):
    best_loss = float('inf')
    for dx, dy in shifts:
        shifted_target = shift(target_tracks, dx, dy)
        loss = huber(pred, shifted_target)
        best_loss = min(best_loss, loss)
    return best_loss
```

---

## ⚙️ **5. Training Setup**

* **Patch size**: 512×512
* **Batch size**: 32
* **Optimizer**: AdamW (`lr=0.001`, `weight_decay=0.001`)
* **LR Scheduler**: Linear warm-up (10%) + linear decay (90%)
* **Gradient clipping**: Yes
* **Weighted sampling**: For class imbalance (optional, not used in final)

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = LinearWarmupAndDecay(...)
```

---

## 🌍 **6. Global Inference**

* Inference on 512×512 patches
* Only center 312×312 pixels retained for final mosaic (100-pixel border context)
* Postprocessed to EPSG:3857
* Converted to COG (Cloud Optimized GeoTIFF)
* Reprojected & streamed via WMS

---

## 🔄 **7. Model Evaluation**

* Metrics: MAE, RMSE, MAPE, RRMSE
* Validation on non-overlapping geographic zones
* Underestimation of tall trees (>30 m) remains an issue due to data imbalance and saturation

---

### 📌 Summary of Outputs

* **Canopy Height Map**: 10m resolution, global
* **Best MAE**: 2.43 m (overall), 4.45 m for trees >5 m
* **Code**: Available at [AI4Forest GitHub](https://github.com/AI4Forest/Global-Canopy-Height-Map)


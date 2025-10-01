# Vicinity-Guided Discriminative Latent Diffusion (DVD) [NeurIPS 2025]

Official implementation of our NeurIPS 2025 paper:
**Vicinity-Guided Discriminative Latent Diffusion for Privacy-Preserving Domain Adaptation**

---

## 🔎 Motivation

Source-free domain adaptation (SFDA) is important when source data cannot be accessed due to privacy or storage constraints.
Most existing SFDA methods only implicitly align target features but lack an explicit and discriminative mechanism to transfer decision boundaries across domains.

Our method, **Discriminative Vicinity Diffusion (DVD)**, introduces a lightweight latent diffusion model trained on **source features only**. This frozen module generates **vicinity-aware cues** during adaptation, guiding the target model toward source-consistent decision regions without ever revisiting raw source data.

---

## ✨ Key Contributions

* **New Framework:** First to use latent diffusion for explicit discriminative transfer in SFDA.
* **Vicinity Guidance:** Gaussian priors are built from latent k-NNs, ensuring label consistency and stable diffusion.
* **Privacy-Preserving:** No need to generate raw images or share source data — only a frozen auxiliary diffusion module is used.
* **Strong Results:** Outperforms state-of-the-art SFDA methods on Office-31, Office-Home, and VisDA.
* **Generalization Benefits:** Improves source-supervised learning and boosts domain generalization.
* 
---

## 🔍 Method Overview

Our DVD consists of two main stages:

1. **Source Training** – A source encoder $G_s$ is trained with label supervision. Discriminative Vicinity Diffusion (DVD) leverages a Gaussian sampler and diffusion module $D$ to learn vicinity-aware latent representations that preserve decision boundaries.
2. **Target Adaptation** – The target encoder $G_t$ is adapted without source data. The frozen diffusion module $D$ generates source-like cues around target samples. These are aggregated by the SiLGA module to produce vicinity-aligned representations, which are optimized using classification and consistency losses.

<p align="center">
  <img src="training.png" alt="Method Overview" width="1000"/>
</p>

*Figure: Overview of Discriminative Vicinity Diffusion (DVD). (a) Source training with vicinity-guided diffusion. (b) Target adaptation with frozen DVD and SiLGA aggregator for label-consistent alignment.*

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 📊 Running Experiments

### Office-31 / Office-Home

**Step 1. Source representation training**

```bash
python source_pretrain_office.py
```

**Step 2. Move pre-trained weights**

```bash
mv san/* weight/
```

**Step 3. Diffusion model training**

```bash
python diffusion_office.py
```

**Step 4. Target adaptation**

```bash
python adaptation_office_with_diffusion.py
```

---

### VisDA

**Step 1. Source representation training**

```bash
python source_pretrain_visda.py
```

**Step 2. Move pre-trained weights**

```bash
mv san/* weight/
```

**Step 3. Diffusion model training**

```bash
python diffusion_visda.py
```

**Step 4. Target adaptation**

```bash
python adaptation_visda_with_diffusion.py
```

---

## 📂 Dataset Preparation

* **Office-31**: [Download](https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code)
* **Office-Home**: [Download](https://www.hemanthdv.org/officeHomeDataset.html)
* **VisDA-C**: [Download](http://ai.bu.edu/visda-2017/)

Place them under `./data/` as:

```
data/
  OfficeHome/
  Office31/
  VisDA/
```

---

## 📈 Results

| Dataset     | Backbone   | Baseline SFDA | DVD (Ours) |
| ----------- | ---------- | ------------- | ---------- |
| Office-31   | ResNet-50  | 89.8          | **91.2**   |
| Office-Home | ResNet-50  | 73.0          | **73.7**   |
| VisDA-C     | ResNet-101 | 87.8          | **88.9**   |

---

## 📝 Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{wang2025dvd,
  title={Vicinity-Guided Discriminative Latent Diffusion for Privacy-Preserving Domain Adaptation},
  author={Wang, Jing and Bae, Wonho and Chen, Jiahong and Wang, Wenxu and Noh, Junhyug},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

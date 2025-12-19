---
title: SMART (Next-token Prediction)
layout: default
parent: Notes
nav_order: 4
---

# SMART: Scalable Multi-agent Real-time Motion Generation via Next-token Prediction

> 작성일: 2025-12-19
> 목적: Multi-agent motion generation을 위한 GPT-style policy 분석

> **Paper**: [arXiv:2405.15677](https://arxiv.org/abs/2405.15677)
> **Code**: [github.com/rainmaker22/SMART](https://github.com/rainmaker22/SMART)
> **Project**: [smart-motion.github.io/smart](https://smart-motion.github.io/smart/)
> **License**: Apache-2.0

## Overview

SMART는 GPT-style의 next-token prediction을 사용하는 autonomous driving motion generation 프레임워크이다. Vectorized map과 agent trajectory를 discrete token으로 변환하여 decoder-only transformer로 처리한다.

**주요 성과:**
- Waymo Open Sim Agents Challenge 2024 **1위** (CVPR 2024 WAD Workshop)
- NeurIPS 2024 accepted
- nuPlan closed-loop planning SOTA (learning-based 중)
- **Zero-shot generalization**: nuPlan → WOMD

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SMART Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Tokenization                                         │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │   Road Tokens    │  │  Motion Tokens   │                │
│  │                  │  │                  │                │
│  │  - Position      │  │  - Position      │                │
│  │  - Direction     │  │  - Heading       │                │
│  │  - Type          │  │  - Shape         │                │
│  │  (≤5m segments)  │  │  (0.5s intervals)│                │
│  └────────┬─────────┘  └────────┬─────────┘                │
│           │                     │                           │
│           ▼                     ▼                           │
│  ┌──────────────────────────────────────────┐              │
│  │              RoadNet (Encoder)            │              │
│  │       Multi-head Self-Attention           │              │
│  │     + Relative Positional Embedding       │              │
│  └──────────────────┬───────────────────────┘              │
│                     │                                       │
│                     ▼                                       │
│  ┌──────────────────────────────────────────┐              │
│  │           MotionNet (Decoder)             │              │
│  │        Factorized Attention Layers        │              │
│  │                                           │              │
│  │  1. Temporal Attention                    │              │
│  │  2. Agent-Agent Attention                 │              │
│  │  3. Agent-Map Attention                   │              │
│  └──────────────────┬───────────────────────┘              │
│                     │                                       │
│                     ▼                                       │
│           Next Token Prediction                             │
│        (Categorical Distribution)                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Tokenization

### Agent Motion Tokenization

| Parameter | Value |
|:----------|:------|
| Time interval | **0.5초** |
| Clustering | k-means (k-disks algorithm) |
| Features | Position, heading, shape |
| Vocab | Agent type별 별도 vocabulary (Vehicle, Pedestrian, Cyclist) |

**Compounding Error 방지:**
- Training 시 exact match 대신 **top-k closest tokens**에서 sampling
- Noise injection으로 robustness 향상

### Road/Map Tokenization

| Parameter | Value |
|:----------|:------|
| Segment length | **≤ 5m** |
| Features | Start/end position, direction, road type |
| Processing | **Parallel** (temporal dependency 없음) |

```
Road Token Features:
├── Position of each point
├── Road direction at each point
└── Road type (lane, boundary, etc.)
```

---

## Training Details

### Hyperparameters

| Parameter | Value |
|:----------|:------|
| Optimizer | AdamW |
| Dropout | 0.1 |
| Weight decay | 0.1 |
| Initial LR | 0.0002 |
| LR schedule | Cosine annealing → 0 |
| Batch size | 4 scenarios |
| GPU memory | ≤ 30GB |
| Loss | Cross-entropy (categorical) |

### Scaling

1 billion motion tokens 수집하여 학습:
- Power-law scaling: β = -0.157
- 1M ~ 101M 파라미터 검증

---

## Inference Speed

### Benchmark Results

| Metric | Value |
|:-------|:------|
| Single-step inference | **5 ~ 20 ms** |
| Average | **< 10 ms** |
| 7M model benchmark | 17.21 ms/frame |

**Real-time 성능:**
- 10ms 평균 → **100Hz** 가능
- 20ms 최대 → **50Hz** 가능
- **VILS 10Hz 요구사항 충족**

---

## Zero-shot Generalization

### 방법

1. nuPlan 데이터셋으로만 학습
2. WOMD에서 직접 evaluation (fine-tuning 없음)

### 결과

| Model | Trained On | Tested On | Realism Score |
|:------|:-----------|:----------|:--------------|
| SMART (full) | WOMD | WOMD | 0.7591 |
| SMART (zero-shot) | nuPlan | WOMD | **0.7210** |

### 원리

Discrete tokenization이 continuous regression 대비 domain gap 감소:
- Position을 discrete token으로 양자화
- Dataset-specific한 continuous value 의존성 제거

---

## Benchmark Results

### WOMD Sim Agents 2024

| Model | Realism | Kinematic | Interactive |
|:------|:--------|:----------|:------------|
| SMART-tiny (7M) | 0.7591 | 0.8039 | 0.8632 |
| SMART-large | 0.7614 | - | - |
| SMART-zeroshot | 0.7210 | 0.7806 | - |

### nuPlan Closed-loop

- Learning-based algorithms 중 **SOTA**
- val14 benchmark

---

## Traffic Light Support

### 현재 상태

**⚠️ 명시적으로 다루지 않음**

논문과 코드에서 traffic light 처리에 대한 언급 없음:
- Road token에 traffic light status 포함 여부 불명확
- WOMD preprocessing 과정에서 처리될 수 있음 (확인 필요)

### VILS 적용 시 고려사항

Traffic light이 중요한 경우:
1. 코드 분석하여 input에 추가 가능한지 확인
2. Road token type에 traffic light state 추가 시도
3. 또는 VBD 사용 고려

---

## Installation

```bash
# Environment setup
conda env create -f environment.yml
conda activate SMART

# Dependencies
pip install -r requirements.txt

# PyG issues
bash install_pyg.sh
```

### Data Preparation

```bash
# 1. Download Waymo Open Motion Dataset (scenario format)
# 2. Install Waymo Open Dataset API
# 3. Preprocess
python data_preprocess.py \
    --input_dir ./data/waymo/scenario/training \
    --output_dir ./data/waymo_processed/training
```

**Expected structure:**
```
SMART/data/waymo_processed/
├── training/
├── validation/
└── testing/
```

---

## Usage

### Training
```bash
python train.py --config configs/train/train_scalable.yaml
```

### Evaluation
```bash
python eval.py --config ${config_path} --pretrain_ckpt ${ckpt_path}
```

---

## Pretrained Checkpoints

### 공개 상태

> "We will release the model parameters of a medium-sized model **not trained on Waymo data**. Users can fine-tune this model with Waymo data as needed."

- WOMD 데이터 없이 학습된 medium model 공개 예정
- Waymo 참가 약관 준수를 위함
- Fine-tuning 필요

---

## Code Structure

```
SMART/
├── smart/              # Core implementation
├── configs/            # Configuration files
│   └── train/
│       └── train_scalable.yaml
├── scripts/            # Utility scripts
├── data/               # Dataset directory
├── data_preprocess.py  # Preprocessing
├── train.py            # Training
├── val.py              # Validation
└── eval.py             # Evaluation
```

---

## VILS 적용 고려사항

### 장점

- ✅ **Real-time** (< 10ms, 100Hz 가능)
- ✅ Multi-agent 동시 처리
- ✅ Zero-shot generalization
- ✅ 오픈소스 (Apache-2.0)
- ✅ Checkpoint 공개 예정

### 단점/이슈

- ⚠️ **Traffic light 미지원** (명시적 처리 없음)
- ⚠️ Checkpoint이 WOMD 데이터 없이 학습됨 (fine-tuning 필요)
- ⚠️ WOMD format으로 맵 변환 필요

### Custom Map (FMTC) 적용 시 필요 작업

1. **Map Tokenization**
   - FMTC HD Map → Road tokens
   - ≤ 5m segment로 분할
   - (position, direction, type) 추출

2. **Agent Initialization**
   - Motion token vocabulary 확인
   - Initial token 설정

3. **Traffic Light 처리 (추가 개발 필요)**
   - Road token에 traffic light state 추가
   - 또는 별도 conditioning 방법 개발

---

## VBD vs SMART 비교

| Feature | VBD | SMART |
|:--------|:----|:------|
| **Speed** | ~6Hz (5 DDIM steps) | **100Hz** |
| **Traffic Light** | ✅ 지원 | ❌ 미지원 |
| **Architecture** | Diffusion | GPT-style |
| **Checkpoint** | 불명확 | 공개 예정 |
| **Controllability** | ✅ High | ⚠️ Limited |

**추천:**
- Traffic light 중요 → **VBD**
- Real-time 우선 → **SMART**
- 둘 다 필요 → SMART + traffic light 추가 개발

---

## References

```bibtex
@article{wu2024smart,
  title={SMART: Scalable Multi-agent Real-time Simulation via Next-token Prediction},
  author={Wu, Wei and Feng, Xiaoxin and Gao, Ziyan and Kan, Yuheng},
  journal={arXiv preprint arXiv:2405.15677},
  year={2024}
}
```

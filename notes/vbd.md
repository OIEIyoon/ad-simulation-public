---
title: VBD (Versatile Behavior Diffusion)
layout: default
parent: Notes
nav_order: 3
---

# VBD: Versatile Behavior Diffusion for Generalized Traffic Agent Simulation

> 작성일: 2025-12-19
> 목적: Multi-agent traffic simulation을 위한 diffusion 기반 policy 분석

> **Paper**: [arXiv:2404.02524](https://arxiv.org/abs/2404.02524)
> **Code**: [github.com/SafeRoboticsLab/VBD](https://github.com/SafeRoboticsLab/VBD)
> **Project**: [sites.google.com/view/versatile-behavior-diffusion](https://sites.google.com/view/versatile-behavior-diffusion)
> **License**: Apache-2.0

## Overview

VBD는 diffusion generative model을 사용하여 closed-loop 환경에서 scene-consistent하고 controllable한 multi-agent interaction을 예측하는 traffic scenario generation framework이다.

**주요 성과:**
- Waymo Open Sim Agents Challenge 2024 **2위**
- RSS 2024 Workshop (AVAS) **Best Paper Award**
- Waymo Sim Agents Benchmark SOTA

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        VBD Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐                                        │
│  │  Scene Context  │                                        │
│  │     Encoder     │  ← Query-centric Transformer           │
│  │                 │                                        │
│  │  - Agent history│                                        │
│  │  - Map polylines│                                        │
│  │  - Traffic lights│                                       │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │    Behavior     │  ← Multi-modal trajectory prediction   │
│  │    Predictor    │    (static anchor 기반)                │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │    Denoiser     │  ← Diffusion model                     │
│  │                 │    Joint control sequence 예측         │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│    Joint Multi-Agent Trajectories                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 핵심 특징

1. **Action Space 기반**: State space가 아닌 action space (acceleration, yaw rate)에서 동작
2. **Query-centric Encoding**: Translation invariance를 위해 local coordinate로 변환
3. **Rollout-based Supervision**: Noise prediction 대신 trajectory rollout supervision 사용

---

## Input Representation

### Agent History
| Feature | Description |
|:--------|:------------|
| Position | (x, y) coordinates |
| Heading | Orientation angle |
| Velocity | Speed vector |
| Bounding Box | (length, width) dimensions |
| **Timesteps** | 11 frames (1초, 10Hz) |

### Map Polylines
| Feature | Description |
|:--------|:------------|
| Polyline count | 256 polylines |
| Points per polyline | 30 waypoints |
| Content | Road geometry, lane boundaries |

### Traffic Lights
| Feature | Description |
|:--------|:------------|
| Count | 16 traffic control points |
| Content | Status (red/yellow/green), location |

---

## Training Details

### Diffusion Configuration
| Parameter | Value |
|:----------|:------|
| Diffusion steps (K) | 50 |
| Noise schedule | **Log schedule** (not cosine) |
| Sampling | DDIM |

### Noise Schedule 선택 이유

기존 cosine schedule 대신 log schedule 사용:
- 적절한 signal-to-noise ratio 유지
- "Short-cut learning" 방지
- Closed-loop에서 더 나은 성능

### Training Supervision

```
❌ Direct noise prediction (ε) → 의미있는 agent behavior 생성 실패
✅ Trajectory rollout supervision → Map adherence 유지
```

---

## Inference

### Speed
| DDIM Steps | Runtime | Quality |
|:-----------|:--------|:--------|
| 5 steps | **~0.16s** | Good |
| 50 steps | ~1.6s | Best |

- 5 steps로도 generation quality와 real-time performance의 균형 달성
- **~6Hz** 실시간 가능 (5 steps 기준)

### Controllability

Inference-time scenario editing 지원:
- Behavior prior로 agent 행동 조절
- Model-based optimization과 결합 가능
- Safety-critical scenario 생성

---

## Benchmark Results

### Waymo Sim Agents (44,920 test scenarios)

VBD는 autoregressive 모델들과 비교해 적은 파라미터로 competitive한 성능 달성:

| Aspect | Strength |
|:-------|:---------|
| Interaction modeling | Strong |
| Map compliance | Strong |
| Parameter efficiency | High |

---

## Installation

```bash
# Environment setup
conda env create -n vbd -f environment.yml
conda activate vbd

# Waymax installation
pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax

# VBD installation
pip install -e .
```

### Data Preparation

```bash
# Waymo Open Motion Dataset V1.2 tf_example 필요
python script/extract_data.py \
    --data_dir /path/to/waymo_open_motion_dataset_dir \
    --save_dir /path/to/data_save_dir \
    --num_workers 16 \
    --save_raw
```

---

## Usage

### Training
```bash
python script/train.py --cfg config/VBD.yaml --num_gpus 8
```

### Testing (Closed-Loop Simulation)
```bash
python script/test.py \
    --test_set /path/to/data \
    --model_path ./train_log/VBD/model.pth \
    --save_simulation
```

---

## Code Structure

```
VBD/
├── config/           # Configuration files
├── script/
│   ├── train.py      # Training script
│   ├── test.py       # Testing script
│   └── extract_data.py  # Data preprocessing
├── vbd/              # Core implementation
└── example/          # Jupyter notebooks
    ├── unguided_generation.ipynb
    └── goal_guided_generation.ipynb
```

---

## VILS 적용 고려사항

### 장점
- ✅ **Traffic light 지원** (16 control points)
- ✅ Closed-loop multi-agent interaction
- ✅ 오픈소스 (Apache-2.0)
- ✅ Controllable generation

### 단점/이슈
- ⚠️ Pretrained checkpoint 공개 여부 불명확 (GitHub 확인 필요)
- ⚠️ Waymax 의존성
- ⚠️ ~6Hz (10Hz 목표 대비 부족할 수 있음)
- ⚠️ WOMD format으로 맵 변환 필요

### Custom Map (FMTC) 적용 시 필요 작업

1. **Map Format 변환**
   - FMTC HD Map → Waymo polyline format
   - 256 polylines, 30 waypoints/polyline

2. **Traffic Light 연동**
   - 16개 traffic control points format 맞추기
   - Status encoding (red/yellow/green)

3. **Agent Initialization**
   - Spawn position, heading, velocity 설정
   - Bounding box dimensions

---

## References

```bibtex
@article{huang2024versatile,
  title={Versatile Behavior Diffusion for Generalized Traffic Agent Simulation},
  author={Huang, Yen-Ling and Zhang, Kai and ...},
  journal={arXiv preprint arXiv:2404.02524},
  year={2024}
}
```

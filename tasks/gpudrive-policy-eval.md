---
title: GPUDrive Policy Evaluation
permalink: /tasks/gpudrive-policy-eval/
---

# GPUDrive Policy Evaluation

Waymo 시나리오에서 학습된 Neural Policy와 원본 Log Replay 비교 평가

---

## Model

| 항목 | 값 |
|:-----|:---|
| **Policy** | `policy_S10_000_02_27` (Late Fusion, PPO) |
| **Parameters** | **51,228** (≈51K) |
| **Observation** | 2,984차원 (Ego 6 + Partner 378 + Road Graph 2,600) |
| **Action** | 91개 이산 액션 (13 steer × 7 accel) |
| **Dynamics** | Classic (acceleration + steering) |
| **Training** | Waymo Open Motion Dataset, 10K scenarios |

### Observation (2,984-dim)

3종류의 관측 정보를 flat concatenation하여 입력:

| 모달리티 | 차원 | 구성 | 비고 |
|:---------|-----:|:-----|:-----|
| **Ego State** | 6 | speed, vehicle_length, vehicle_width, rel_goal_x, rel_goal_y, is_collided | 자차 상태 (상대 좌표) |
| **Partner Obs** | 378 | 63 agents × 6 features (speed, rel_pos_x, rel_pos_y, orientation, length, width) | 주변 차량 상태 |
| **Road Graph** | 2,600 | 200 points × 13 features (x, y, seg_length, seg_width, seg_height, orientation + 7-class one-hot type) | 도로 구조 |

**Road Graph 7-class one-hot type**: RoadEdge, RoadLine, RoadLane, CrossWalk, SpeedBump, StopSign, None — 에이전트 중심 반경 내 가장 가까운 200개 도로 포인트 선택

### Architecture

```
Obs (2,984-dim)
├─ Ego State (6)       → Linear(6→64)  → LN → Tanh → Linear(64→64)     [4,672]
├─ Partners (63×6)     → Linear(6→64)  → LN → GELU → Linear(64→64)     [4,672]
│                        → MaxPool(63→1)
└─ Road Graph (200×13) → Linear(13→64) → LN → GELU → Linear(64→64)     [5,120]
                          → MaxPool(200→1)
                              ↓
                    Concat(192) → Linear(192→128) → Dropout              [24,704]
                       ├─ Actor  → Linear(128→91) → Categorical          [11,739]
                       └─ Critic → Linear(128→1)                         [   129]
                                                              Total:  51,228 params
```

---

## Setup

- **데이터**: Waymo Open Motion Dataset 1,000 시나리오 (HuggingFace `daphne-cornelisse/pufferdrive_train`)
- **시뮬레이터**: GPUDrive (Madrona C++ backend)
- **에피소드**: 91 steps (9.1초, dt=0.1s)
- **비교 조건**:
  - **Neural Policy**: 학습된 정책이 매 스텝 관측 → 액션 추론
  - **Log Replay**: Waymo 원본 기록 궤적 재생 (state dynamics)

---

## Results

각 시나리오별 **Neural Policy** vs **Log Replay** 비교.

{% assign scenarios = "tfrecord-00002-of-01000_345,tfrecord-00005-of-01000_266,tfrecord-00065-of-01000_207,tfrecord-00072-of-01000_33,tfrecord-00107-of-01000_443,tfrecord-00135-of-01000_173,tfrecord-00136-of-01000_351,tfrecord-00144-of-01000_130,tfrecord-00165-of-01000_188,tfrecord-00196-of-01000_398,tfrecord-00204-of-01000_11,tfrecord-00306-of-01000_321,tfrecord-00315-of-01000_384,tfrecord-00373-of-01000_209,tfrecord-00398-of-01000_475,tfrecord-00444-of-01000_464,tfrecord-00551-of-01000_267,tfrecord-00585-of-01000_140,tfrecord-00619-of-01000_326,tfrecord-00676-of-01000_450,tfrecord-00685-of-01000_305,tfrecord-00707-of-01000_112,tfrecord-00855-of-01000_180,tfrecord-00863-of-01000_388,tfrecord-00931-of-01000_384" | split: "," %}

{% for idx in scenarios %}
<div class="scenario-card">
<div class="scenario-card__header">Scenario {{ forloop.index }} — {{ idx }}</div>
<div class="scenario-card__label">Neural Policy</div>
<img src="{{ site.baseurl }}/assets/gpudrive-eval/policy/{{ idx }}_policy.gif" alt="Policy {{ forloop.index }}">
<div class="scenario-card__label">Log Replay</div>
<img src="{{ site.baseurl }}/assets/gpudrive-eval/replay/{{ idx }}_replay.gif" alt="Replay {{ forloop.index }}">
</div>
{% endfor %}

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

## Benchmark (Paper-Reported)

GPUDrive 논문(ICLR 2025)에서 보고된 정량적 성능 지표.

### Agent Performance

| Metric | Value | Note |
|:-------|------:|:-----|
| **Goal Rate** | 95% | 1,000 WOMD 시나리오, 15h 학습 |
| **Theoretical Ceiling** | ~98% | ~2% 시나리오는 WOMD 라벨 오류로 도달 불가 |
| **Collision Rate** | ~3–5% | 학습 완료 시점 기준 |
| **Off-road Rate** | ~2–4% | 학습 완료 시점 기준 |

### Simulation Speed

| Metric | Value | Note |
|:-------|------:|:-----|
| **Peak ASPS** | 1M+ | Agent Steps Per Second (전체 에이전트) |
| **Controlled ASPS** | 200K–500K | 제어 에이전트만 (PufferLib PPO) |
| **Speedup vs Nocturne** | 200–300× | 동일 10 시나리오 기준 |

### Scaling Efficiency

| Scenarios | Total Time | Per-Scene Cost |
|----------:|-----------:|---------------:|
| 10 | ~3 min | 18 sec |
| 100 | ~20 min | 12 sec |
| 1,024 | ~200 min | **15 sec** |

> Per-scene cost는 시나리오 수 증가에 따라 sub-linear 감소. 대규모 WOMD(100K scenes) 학습에도 학술 연구급 GPU 단일 장비로 가능.

### Evaluation Metrics 정의

| Metric | 조건 | 판정 |
|:-------|:-----|:-----|
| **Goal Achieved** | 목표 위치 $\delta$ 이내 도달 | `dist(agent, goal) < dist_to_goal_threshold` |
| **Collision** | 차량/도로/비차량 충돌 | `collidedWithVehicle + collidedWithRoad + collidedWithNonVehicle > 0` |
| **Off-road** | 도로 경계 이탈 | `off_road > 0` |
| **Other** | 목표 미달, 충돌/이탈 없음 | 시간 초과 (91 step 내 미도달) |

---

## Results

각 시나리오별 **Neural Policy** vs **Log Replay** 비교.

{% assign scenarios = "tfrecord-00002-of-01000_345,tfrecord-00005-of-01000_266,tfrecord-00065-of-01000_207,tfrecord-00072-of-01000_33,tfrecord-00107-of-01000_443,tfrecord-00135-of-01000_173,tfrecord-00136-of-01000_351,tfrecord-00144-of-01000_130,tfrecord-00165-of-01000_188,tfrecord-00196-of-01000_398,tfrecord-00204-of-01000_11,tfrecord-00306-of-01000_321,tfrecord-00315-of-01000_384,tfrecord-00373-of-01000_209,tfrecord-00398-of-01000_475,tfrecord-00444-of-01000_464,tfrecord-00551-of-01000_267,tfrecord-00585-of-01000_140,tfrecord-00619-of-01000_326,tfrecord-00676-of-01000_450,tfrecord-00685-of-01000_305,tfrecord-00707-of-01000_112,tfrecord-00855-of-01000_180,tfrecord-00863-of-01000_388,tfrecord-00931-of-01000_384" | split: "," %}

{% for idx in scenarios %}

---

**Scenario {{ forloop.index }} — {{ idx }}**

<table class="gif-grid">
<tr>
<td><img src="{{ site.baseurl }}/assets/gpudrive-eval/policy/{{ idx }}_policy.gif" alt="Policy {{ forloop.index }}"></td>
<td><img src="{{ site.baseurl }}/assets/gpudrive-eval/replay/{{ idx }}_replay.gif" alt="Replay {{ forloop.index }}"></td>
</tr>
</table>
{% endfor %}

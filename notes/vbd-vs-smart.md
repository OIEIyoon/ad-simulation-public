---
title: VBD vs SMART Comparison
layout: default
parent: Notes
nav_order: 5
---

# VBD vs SMART: Multi-agent Traffic Simulation 비교

WOMD Sim Agents Challenge 2024 상위 2개 모델 비교 분석

---

## Quick Comparison

| Aspect | VBD (2위) | SMART (1위) |
|--------|-----------|-------------|
| **Method** | Diffusion (DDIM) | GPT-style next-token |
| **Architecture** | Encoder + Predictor + Denoiser | Encoder-Decoder Transformer |
| **Latency** | ~160ms (5 steps) | **~10ms** |
| **Throughput** | ~6Hz | **~100Hz** |
| **Training Platform** | Waymax | WOMD + nuPlan |
| **Zero-shot** | - | ✅ nuPlan → WOMD |
| **Controllability** | ✅ High | Limited |
| **Checkpoint** | 불명확 | 공개 예정 (non-Waymo) |

---

## Method Comparison

### VBD: Diffusion-based Approach

**Paradigm**: Iterative denoising process
```
Noise → [Denoiser × K steps] → Clean trajectories
```

**Pros:**
- High controllability (inference-time editing)
- Diversity through stochastic sampling
- Strong scene consistency

**Cons:**
- Slower inference (iterative process)
- DDIM acceleration 필요 (5 steps minimum)

### SMART: Autoregressive Approach

**Paradigm**: Sequential token prediction
```
Token₁ → Token₂ → ... → Tokenₙ
```

**Pros:**
- Fast inference (single forward pass per token)
- Scalable (1M ~ 101M params)
- Strong generalization (discrete tokens)

**Cons:**
- Compounding error (autoregressive)
- Limited controllability

---

## Training Comparison

### Dataset

| Aspect | VBD | SMART |
|--------|-----|-------|
| Primary dataset | WOMD v1.2 | WOMD |
| Simulation platform | **Waymax** | - |
| Training tokens | - | **1 billion** |
| Zero-shot dataset | - | **nuPlan** |

### Supervision

| Model | Supervision Type | Key Strategy |
|-------|------------------|--------------|
| VBD | **Trajectory rollout** | Dynamics model로 rollout → GT 비교 |
| SMART | **Next-token CE** | Top-k closest tokens sampling |

---

## Inference Comparison

### Latency & Throughput

| Model | Latency (Single Scenario) | Throughput | Real-time |
|-------|---------------------------|------------|-----------|
| VBD (5 steps) | ~160ms | ~6Hz | △ |
| VBD (50 steps) | ~1.6s | ~0.6Hz | ✗ |
| SMART (avg) | **< 10ms** | **~100Hz** | ✅ |
| SMART (max) | ~20ms | ~50Hz | ✅ |

### Scalability

**VBD:**
- Fixed K steps (trade-off: quality vs speed)
- Parallelizable (all agents jointly)

**SMART:**
- Autoregressive (sequential tokens)
- Efficient factorized attention
- Model size scalable (7M ~ large)

---

## Benchmark Performance

### Waymo Sim Agents 2024

| Metric | VBD | SMART-tiny (7M) | SMART-large |
|--------|-----|-----------------|-------------|
| **Ranking** | **2위** | - | **1위** |
| Realism | SOTA | 0.7591 | 0.7614 |
| Kinematic | Strong | 0.8039 | - |
| Interactive | Strong | 0.8632 | - |

### Other Benchmarks

| Benchmark | VBD | SMART |
|-----------|-----|-------|
| nuPlan closed-loop | - | **SOTA** (learning-based) |
| Zero-shot (nuPlan → WOMD) | - | 0.7210 realism |

---

## Use Case Recommendations

### Choose VBD if:
1. **Controllability 중요**: Inference-time scenario editing 필요
2. **Diversity 필요**: Stochastic sampling으로 다양한 behavior 생성
3. **Safety-critical**: Constraint-based optimization 적용
4. Real-time 요구사항이 ~6Hz로 충분

### Choose SMART if:
1. **Real-time 필수**: 10Hz 이상 throughput 요구
2. **Generalization**: Zero-shot domain transfer 필요
3. **Scalability**: 모델 크기 조절 가능한 solution
4. Fast prototyping (pretrained checkpoint 활용)

---

## Implementation Considerations

### VBD

**Dependencies:**
- Waymax (필수)
- WOMD tf_example format

**Integration:**
```python
# VBD pipeline
scene_encoding = encoder(history, map, traffic_lights)
behavior_prior = predictor(scene_encoding)
trajectories = denoiser.sample(behavior_prior, K_steps=5)
```

### SMART

**Dependencies:**
- Waymo Open Dataset API
- PyTorch Geometric

**Integration:**
```python
# SMART pipeline
road_tokens = tokenize_map(map_data)
motion_tokens = tokenize_agents(agent_history)
next_tokens = model.predict(road_tokens, motion_tokens)
trajectories = detokenize(next_tokens)
```

---

## Summary

**Performance:** SMART 1위, VBD 2위 (둘 다 SOTA급)

**Speed:** SMART 압도적 (10ms vs 160ms)

**Flexibility:** VBD 우수 (controllability, editing)

**Generalization:** SMART 우수 (zero-shot 검증)

**선택 기준**: Use case와 requirements에 따라 결정
- Real-time + Scale → **SMART**
- Control + Safety → **VBD**

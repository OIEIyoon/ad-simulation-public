---
title: VILS Policy Options
layout: default
parent: Notes
nav_order: 6
---

# VILS를 위한 Policy 선택 분석

VILS (Virtual NPC Simulation) 서버에서 사용 가능한 policy 옵션 비교

---

## VILS 요구사항

### 시스템 목표
- FMTC PG에서 Ego vehicle과 가상 NPC의 상호작용 시뮬레이션
- 실시간 closed-loop simulation
- WebSocket API를 통한 10Hz 통신

### 핵심 요구사항

| Requirement | Target | Priority |
|-------------|--------|----------|
| **Real-time** | ≥ 10Hz | ✅ Critical |
| **Multi-agent** | 여러 NPC 동시 제어 | ✅ Critical |
| **Map format** | FMTC HD Map | ✅ Critical |
| **Interaction** | Ego-NPC reactive behavior | High |
| **Traffic light** | FMTC signal 연동 | Medium |

---

## Policy 옵션 분석

### 1. GPUDrive Policy (현재)

**Status**: 현재 VILS에서 사용 중

**Performance**:
- Latency: ~5ms/step
- Throughput: ~200Hz
- ✅ 10Hz 요구사항 충족 (40x margin)

**Pros**:
- ✅ 이미 구현 완료 (LightweightEngine)
- ✅ FMTC 맵 변환 완료
- ✅ Observation interface 구축 완료
- ✅ 검증된 동작

**Cons**:
- ⚠️ RL policy 특유의 불안정성 가능
- ⚠️ Interaction quality 제한적
- ⚠️ Generalization 불확실

**Integration Complexity**: ✅ 완료

---

### 2. SMART (1위 모델)

**Performance**:
- Latency: **~10ms/step** (평균)
- Throughput: **~100Hz**
- ✅ 10Hz 요구사항 충족 (10x margin)

**Method Fit**:

| Aspect | VILS Requirement | SMART Capability | Match |
|--------|------------------|------------------|-------|
| Real-time | 10Hz | **100Hz** | ✅ Excellent |
| Multi-agent | Joint control | ✅ Native support | ✅ Perfect |
| Interaction | Ego-NPC | ✅ Agent-agent attention | ✅ Good |
| Generalization | FMTC map | ✅ Zero-shot proven | ✅ Promising |

**Pros**:
- ✅ **WOMD Challenge 1위** (SOTA quality)
- ✅ Real-time performance (10ms)
- ✅ Multi-agent joint prediction
- ✅ Zero-shot generalization 검증됨
- ✅ Pretrained checkpoint 공개 예정
- ✅ 오픈소스 (Apache-2.0)

**Cons**:
- ⚠️ FMTC → Road token 변환 필요
- ⚠️ Tokenization/Detokenization 레이어 개발
- ⚠️ Checkpoint fine-tuning 필요할 수 있음
- ⚠️ Integration 복잡도 높음

**Integration Complexity**: 🔶 Medium-High

**Required Work**:
1. FMTC HD Map → Road token converter
2. Agent state → Motion token encoder
3. Motion token → Trajectory decoder
4. Pretrained model fine-tuning (optional)

---

### 3. VBD (2위 모델)

**Performance**:
- Latency: **~160ms/step** (5 DDIM steps)
- Throughput: **~6Hz**
- △ 10Hz 요구사항 아슬아슬 (margin 부족)

**Method Fit**:

| Aspect | VILS Requirement | VBD Capability | Match |
|--------|------------------|----------------|-------|
| Real-time | 10Hz | **~6Hz** | △ Borderline |
| Multi-agent | Joint control | ✅ Native support | ✅ Perfect |
| Interaction | Ego-NPC | ✅ Diffusion modeling | ✅ Good |
| Controllability | Scenario editing | ✅ High | ✅ Excellent |

**Pros**:
- ✅ **WOMD Challenge 2위** (SOTA quality)
- ✅ Multi-agent joint prediction
- ✅ **High controllability** (inference-time editing)
- ✅ Scene consistency 우수
- ✅ 오픈소스 (Apache-2.0)

**Cons**:
- ⚠️ **Performance bottleneck** (~6Hz, 10Hz 요구 근접)
- ⚠️ Waymax 의존성 (무거움)
- ⚠️ FMTC → WOMD format 변환 필요
- ⚠️ Pretrained checkpoint 공개 불명확

**Integration Complexity**: 🔶 Medium-High

**Required Work**:
1. FMTC HD Map → WOMD polyline format
2. Waymax 환경 설정 또는 우회
3. Diffusion sampling optimization (5 steps → faster?)

---

## 성능 비교

### Latency & Margin

| Policy | Latency | Margin vs 10Hz | Real-time |
|--------|---------|----------------|-----------|
| GPUDrive (현재) | **5ms** | **20x** | ✅ Excellent |
| SMART | **10ms** | **10x** | ✅ Good |
| VBD (5 steps) | **160ms** | **0.6x** | △ Borderline |

### Quality (Expectation)

| Aspect | GPUDrive | SMART | VBD |
|--------|----------|-------|-----|
| Realism | ? | **0.76** (WOMD) | **SOTA** (WOMD) |
| Interaction | Medium? | **0.86** (WOMD) | Strong |
| Controllability | Low | Medium | **High** |

---

## 추천 전략

### Option A: 현상 유지 + 점진적 개선
```
Phase 1: GPUDrive policy로 VILS 완성 (현재)
Phase 2: Edge case 및 interaction quality 평가
Phase 3: 문제 발생 시 → Option B/C 고려
```

**Pros**: 빠른 시스템 완성, 검증 후 개선
**Cons**: Quality가 부족할 수 있음

### Option B: SMART 마이그레이션 (권장)
```
Phase 1: GPUDrive로 VILS 완성
Phase 2: 병렬로 SMART integration 개발
  - FMTC → Road token converter
  - Tokenization pipeline
  - Pretrained model 평가
Phase 3: A/B 테스트
  - Realism
  - Interaction quality
  - Edge case handling
Phase 4: 성능 우수한 쪽 선택
```

**Pros**: SOTA 모델 활용, zero-shot generalization
**Cons**: Integration 복잡도, 개발 시간

### Option C: VBD (조건부)
```
조건: Controllability가 critical하고, 6Hz도 허용 가능한 경우
```

**Pros**: 최고 수준의 controllability
**Cons**: Performance bottleneck 위험

---

## 최종 판단

### SMART가 VILS에 적합한가?

**✅ Yes, but with caveats**

**적합성 평가**:
1. **Performance**: ✅ 10Hz 충족 (10ms latency)
2. **Quality**: ✅ WOMD 1위 (검증된 성능)
3. **Multi-agent**: ✅ Native support
4. **Generalization**: ✅ Zero-shot 입증
5. **Integration**: △ Medium complexity

**권장사항**:
1. **단기**: GPUDrive policy로 VILS 완성
2. **중기**: SMART integration 병렬 개발 (실험용)
3. **장기**: A/B 테스트 후 선택

**개발 우선순위**:
```
P0: VILS 기본 기능 완성 (GPUDrive)
P1: FMTC → Road token converter
P2: SMART tokenization pipeline
P3: Pretrained model evaluation
P4: A/B testing framework
```

---

## Next Steps

1. **GPUDrive policy 검증**
   - Edge case testing
   - Interaction quality 평가
   - Failure mode 분석

2. **SMART 실험 환경 구축**
   - GitHub repo clone
   - Environment setup
   - FMTC 샘플 데이터로 tokenization 테스트

3. **Checkpoint 확보**
   - SMART medium model release 모니터링
   - Evaluation on FMTC scenarios

4. **Integration Design**
   - Tokenizer architecture
   - VILS API integration point
   - Performance profiling plan

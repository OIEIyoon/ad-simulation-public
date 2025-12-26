---
title: VILS Server 개발
layout: default
parent: Tasks
nav_order: 1
has_children: true
---

# VILS Server 개발

FMTC PG에서 Ego vehicle이 가상 NPC들과 상호작용하며 자율주행 테스트할 수 있는 시뮬레이션 서버

## Tasks

- GPUDrive 분석 및 Policy 추출
- FMTC HD맵 변환
- Observation 인터페이싱
- VILS 시뮬레이터 개발 (LightweightEngine)
- GUI 시각화 도구
- WebSocket API
- Ego 상태 주입 및 NPC 반응
- 시나리오 관리

---

## Details

### GPUDrive 분석 및 Policy 추출

- **GPUDrive는 RL 학습용 시뮬레이터**
  - 여러 시나리오를 배치로 빠르게 돌리기 위한 환경
  - 에피소드 기반 (91 step 제한), 무한 시뮬레이션 불가
  - VILS용 실시간 시뮬레이터로 직접 사용하기 부적합
- GPUDrive 저자들이 학습해둔 Policy network 제공
  - `NeuralNet.from_pretrained("daphne-cornelisse/policy_S10_000_02_27")`
  - Observation 형식: 2984차원
  - Action space: 13 steer × 7 accel = 91개 이산 액션
- **결론**: GPUDrive 환경은 사용하지 않고, Policy network만 추출하여 활용

### FMTC HD맵 변환

- FMTC Shapefile → GPUDrive JSON 형식 변환
- 레이어 매핑:
  - A2_LINK(차선) → LANE_SURFACE_STREET
  - B2_SURFACELINEMARK(노면표시) → ROAD_LINE_*
  - C3_VEHICLEPROTECTIONSAFETY(방호시설) → ROAD_EDGE_*

### Observation 인터페이싱

- Policy 입력 형식(2984차원)에 맞게 obs 직접 구성
- 구조: ego_state(6) + partner_obs(378) + road_obs(2600)
- KDTree 기반 도로점 쿼리 (200개)

### VILS 시뮬레이터 개발 (LightweightEngine)

- **GPUDrive 환경 없이 독립 동작하는 VILS 전용 시뮬레이터**
- PyTorch만으로 동작 (Policy inference only)
- Bicycle model dynamics
- 무한 시뮬레이션, 동적 에이전트 생성/삭제
- 성능: ~5ms/step (10Hz 요구 대비 20x 마진)

### GUI 시각화 도구

- Pygame 기반 FMTC 맵 렌더러
- NPC 스폰: 드래그로 위치/yaw/속도 설정 → 클릭으로 goal 설정
- 맵 조작: 회전, 줌, 팬

### WebSocket API

- FastAPI + WebSocket
- Ego state 수신 → NPC states 송신 (10Hz)
- REST API: 상태 조회, NPC 추가/삭제

### Ego 상태 주입 및 NPC 반응

- TODO: 실차 위치를 시뮬레이션에 반영
- TODO: NPC가 Ego를 인식하고 반응하도록 구현

### 시나리오 관리

- TODO: 특정 시나리오 정의 및 로드
- TODO: 시나리오별 NPC 배치/행동 패턴

---

## Subtasks

- [Mock Policy 구현](vils-server-progress-2025-12-26) (2025-12-26)

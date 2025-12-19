---
title: GPUDrive Observation 구조
layout: default
parent: Notes
nav_order: 2
---

# GPUDrive Observation 구조

GPUDrive Policy network 입력으로 사용되는 observation 형식 분석

---

## 개요

- 각 에이전트마다 독립적으로 생성
- 모든 좌표는 **ego-centric** (해당 에이전트 기준 상대 좌표)
- Shape: `[num_worlds, num_agents, 2984]`

---

## 구조 (2984차원)

| 구성 요소 | 차원 | 인덱스 | 계산 |
|:----------|:----:|:------:|:-----|
| ego_state | 6 | 0-5 | 6 features |
| partner_obs | 378 | 6-383 | 63 agents × 6 features |
| road_map_obs | 2600 | 384-2983 | 200 points × 13 features |

---

## Ego State (6차원)

자기 자신의 상태

| 인덱스 | 필드 | 정규화 |
|:------:|:-----|:-------|
| 0 | speed | ÷100 |
| 1 | vehicle_length | ÷30 |
| 2 | vehicle_width | ÷15 |
| 3 | rel_goal_x | [-1000, 1000] → [-1, 1] |
| 4 | rel_goal_y | [-1000, 1000] → [-1, 1] |
| 5 | is_collided | {0, 1} |

---

## Partner Obs (378차원)

다른 에이전트들 정보 (최대 63개)

각 파트너당 6개 feature:

| 오프셋 | 필드 | 정규화 |
|:------:|:-----|:-------|
| +0 | speed | ÷100 |
| +1 | rel_pos_x | [-1000, 1000] → [-1, 1] |
| +2 | rel_pos_y | [-1000, 1000] → [-1, 1] |
| +3 | orientation | ÷2π |
| +4 | vehicle_length | ÷30 |
| +5 | vehicle_width | ÷15 |

인덱스: `6 + i×6 + offset` (i = 0..62)

---

## Road Map Obs (2600차원)

주변 도로 정보 (가장 가까운 200개 포인트)

각 포인트당 13개 feature:

| 오프셋 | 필드 | 정규화 |
|:------:|:-----|:-------|
| +0 | x | [-1000, 1000] → [-1, 1] |
| +1 | y | [-1000, 1000] → [-1, 1] |
| +2 | segment_length | ÷100 |
| +3 | segment_width | ÷100 |
| +4 | segment_height | ÷100 |
| +5 | orientation | ÷2π |
| +6~12 | type (one-hot 7종) | {0, 1} |

인덱스: `384 + j×13 + offset` (j = 0..199)

---

## Road Type (7종)

EntityType one-hot 인코딩:

| 인덱스 | 타입 | 설명 |
|:------:|:-----|:-----|
| 0 | None | 빈 슬롯 (padding) |
| 1 | RoadEdge | 도로 경계 |
| 2 | RoadLine | 차선 표시 (백색/황색) |
| 3 | RoadLane | 차선 중심선 |
| 4 | CrossWalk | 횡단보도 |
| 5 | SpeedBump | 과속방지턱 |
| 6 | StopSign | 정지 표지판 |

---

## Ego-Centric 좌표계

```
전역 좌표 (x_global, y_global)
         ↓
평행 이동: (x - ego_x, y - ego_y)
회전: ego heading → +X축
         ↓
상대 좌표 (x_rel, y_rel)
```

---

## 포함되지 않는 정보

- Traffic Light (신호등)
- Turn Signal (방향지시등)
- Speed Limit (제한속도)
- Lane Connectivity (차선 연결 정보)
- 절대 좌표

---

## 참고

- 상세 분석: [OBSERVATION_SPEC.md](/docs/OBSERVATION_SPEC.md)

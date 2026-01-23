---
title: PufferDrive
layout: default
parent: Notes
nav_order: 5
---

# PufferDrive 종합 분석

> PufferDrive는 RL 기반 자율주행 에이전트 학습을 위한 고처리량 시뮬레이터.
> 단일 GPU에서 초당 300K-320K 스텝, 10,000개 멀티에이전트 시나리오를 약 15분 내 학습 가능.

---

## 목차

1. [아키텍처 개요](#1-아키텍처-개요)
2. [데이터 파이프라인](#2-데이터-파이프라인)
3. [Model Input/Output](#3-model-inputoutput)
4. [Entity 및 도로 객체 유형](#4-entity-및-도로-객체-유형)
5. [RL 환경 구조](#5-rl-환경-구조)
6. [Episode 및 Reset 로직](#6-episode-및-reset-로직)
7. [Goal Point 시스템](#7-goal-point-시스템)
8. [Environment 설정](#8-environment-설정)
9. [물리 엔진](#9-물리-엔진)
10. [핵심 상수 및 정규화](#10-핵심-상수-및-정규화)

---

## 1. 아키텍처 개요

### 핵심 컴포넌트

```
pufferlib/ocean/drive/
├── drive.h          # 코어 시뮬레이터 (C 헤더, ~118KB)
├── drive.c          # C 구현체 및 데모
├── drive.py         # Python 환경 래퍼 (Gymnasium 호환)
├── binding.c        # Python-C 바인딩 (ctypes)
├── visualize.c      # Raylib 기반 렌더링
├── drivenet.h       # C 기반 신경망 추론
└── error.h          # 에러 처리
```

### 데이터 흐름

```
CARLA/Waymo 원본 → JSON → Binary Map → Drive 환경 → Observation → Policy Network
```

---

## 2. 데이터 파이프라인

### 2.1 XODR (OpenDRIVE)

**정의**: ASAM 표준 도로 네트워크 XML 포맷

**구조**:
- 도로(Road): 중심선, 차선, 폭, 경사
- 연결(Junction): 교차로 정보
- 레인(Lane): 차선 타입, 경계
- 오브젝트: 신호등, 표지판 등

**파이프라인**:
```
Town05.xodr → pyxodr 파싱 → 좌표 배열 추출 → JSON
```

**제한사항** (pyxodr):
- 슈퍼엘리베이션/도로 형상 미지원
- 신호, 철도 미지원
- 일부 링크 요소 미지원

### 2.2 JSON 맵 포맷

```json
{
  "sdc_track_index": 0,
  "tracks_to_predict": [1, 2, 3],
  "objects": [
    {
      "id": 1,
      "type": "vehicle",
      "position": [{"x": 0, "y": 0, "z": 0}, ...],  // 91개
      "velocity": [{"x": 0, "y": 0}, ...],           // 91개
      "heading": [0.0, ...],                          // 91개
      "valid": [true, ...],                           // 91개
      "width": 2.0,
      "length": 4.5,
      "height": 1.8,
      "goalPosition": {"x": 50, "y": 10, "z": 0},
      "mark_as_expert": false
    }
  ],
  "roads": [
    {
      "id": 100,
      "type": "ROAD_LANE",
      "geometry": [{"x": 0, "y": 0, "z": 0}, ...]
    }
  ]
}
```

### 2.3 Binary 맵 포맷

**저장 위치**: `resources/drive/binaries/`

**바이너리 레이아웃**:
```
[헤더]
├─ sdc_track_index: int (4 bytes)
├─ num_tracks_to_predict: int (4 bytes)
│   └─ track_indices[]: int × N
├─ num_objects: int (4 bytes)
└─ num_roads: int (4 bytes)

[객체 섹션] × num_objects
├─ unique_map_id: int
├─ type: int (1=vehicle, 2=pedestrian, 3=cyclist)
├─ id: int
├─ array_size: int (91)
├─ Trajectory 배열:
│   ├─ traj_x[91]: float × 91
│   ├─ traj_y[91]: float × 91
│   ├─ traj_z[91]: float × 91
│   ├─ traj_vx[91]: float × 91
│   ├─ traj_vy[91]: float × 91
│   ├─ traj_vz[91]: float × 91
│   ├─ traj_heading[91]: float × 91
│   └─ traj_valid[91]: int × 91
├─ width, length, height: float × 3
├─ goal_x, goal_y, goal_z: float × 3
└─ mark_as_expert: int

[도로 섹션] × num_roads
├─ unique_map_id: int
├─ road_type: int
├─ id: int
├─ array_size: int (기하학 점 개수)
├─ geometry_x[]: float × N
├─ geometry_y[]: float × N
├─ geometry_z[]: float × N
├─ width, length, height: float × 3
├─ goal_x, goal_y, goal_z: float × 3
└─ mark_as_expert: int
```

**변환 명령**:
```bash
python pufferlib/ocean/drive/drive.py  # JSON → Binary
```

---

## 3. Model Input/Output

### 3.1 Observation Space 구조

**총 크기**: 1120 (CLASSIC) 또는 1123 (JERK)

```
┌────────────────────────────────────────────────────────────────────┐
│ EGO Features │ Partner Agents (31×7) │ Road Segments (128×7)      │
│   (7 or 10)  │       (217)           │        (896)               │
└────────────────────────────────────────────────────────────────────┘
  idx 0-6/9       idx 7-223               idx 224-1119
```

### 3.2 EGO Features

**CLASSIC 모드 (7개)**:

| Index | Feature | 정규화 | 범위 |
|-------|---------|--------|------|
| 0 | rel_goal_x | × 0.005 | 목표까지 상대 X |
| 1 | rel_goal_y | × 0.005 | 목표까지 상대 Y |
| 2 | signed_speed | ÷ 100 | 현재 속도 (m/s) |
| 3 | width | ÷ 15 | 차량 폭 |
| 4 | length | ÷ 30 | 차량 길이 |
| 5 | collision_state | 0 or 1 | 충돌 여부 |
| 6 | respawn_flag | 0 or 1 | 리스폰 여부 |

**JERK 모드 (10개)** - 추가 3개:

| Index | Feature | 정규화 | 범위 |
|-------|---------|--------|------|
| 6 | steering_angle | ÷ π | 조향각 (rad) |
| 7 | a_long | 비대칭 | 종방향 가속도 |
| 8 | a_lat | ÷ 4.0 | 횡방향 가속도 |
| 9 | respawn_flag | 0 or 1 | 리스폰 여부 |

### 3.3 Partner Agent Features (31개 × 7)

각 에이전트당:

| Offset | Feature | 정규화 | 설명 |
|--------|---------|--------|------|
| +0 | rel_x | × 0.02 | 상대 X 위치 |
| +1 | rel_y | × 0.02 | 상대 Y 위치 |
| +2 | width | ÷ 15 | 폭 |
| +3 | length | ÷ 30 | 길이 |
| +4 | rel_heading_x | cos | 상대 방향 코사인 |
| +5 | rel_heading_y | sin | 상대 방향 사인 |
| +6 | signed_speed | ÷ 100 | 상대 속도 |

**필터링**:
- 거리 > 50m 제외
- 리스폰된 에이전트 제외
- 최대 31개 (나머지 0 패딩)

### 3.4 Road Segment Features (128개 × 7)

각 도로 세그먼트당:

| Offset | Feature | 정규화 | 설명 |
|--------|---------|--------|------|
| +0 | rel_x | × 0.02 | 세그먼트 중점 X |
| +1 | rel_y | × 0.02 | 세그먼트 중점 Y |
| +2 | length | ÷ 100 | 세그먼트 길이 |
| +3 | width | ÷ 100 | 세그먼트 폭 |
| +4 | cos_angle | [-1, 1] | 방향 코사인 |
| +5 | sin_angle | [-1, 1] | 방향 사인 |
| +6 | road_type | 0-6 | 도로 타입 (type - 4) |

### 3.5 좌표 변환

모든 좌표는 **자차(Ego) 중심 좌표계**로 변환:

```c
// 월드 → 자차 로컬
float cos_h = cosf(ego_heading);
float sin_h = sinf(ego_heading);
float rel_x =  (world_x - ego_x) * cos_h + (world_y - ego_y) * sin_h;
float rel_y = -(world_x - ego_x) * sin_h + (world_y - ego_y) * cos_h;
```

### 3.6 Action Space

**CLASSIC 모드 (Discrete: 91개)**:

```
행동 = accel_idx × 13 + steer_idx
```

| 가속도 (7개) | 값 (m/s²) |
|-------------|-----------|
| 0 | -4.0 |
| 1 | -2.667 |
| 2 | -1.333 |
| 3 | 0.0 |
| 4 | 1.333 |
| 5 | 2.667 |
| 6 | 4.0 |

| 조향각 (13개) | 값 |
|--------------|-----|
| 0 | -1.0 |
| 1 | -0.833 |
| ... | ... |
| 6 | 0.0 |
| ... | ... |
| 12 | 1.0 |

**JERK 모드 (Discrete: 12개)**:

```
행동 = jerk_long_idx × 3 + jerk_lat_idx
```

| 종방향 Jerk (4개) | 값 (m/s³) |
|------------------|-----------|
| 0 | -15.0 |
| 1 | -4.0 |
| 2 | 0.0 |
| 3 | 4.0 |

| 횡방향 Jerk (3개) | 값 (m/s³) |
|------------------|-----------|
| 0 | -4.0 |
| 1 | 0.0 |
| 2 | 4.0 |

**Continuous 모드**:
- 2D 벡터: `[-1, 1] × [-1, 1]`
- 가속/감속 + 조향

---

## 4. Entity 및 도로 객체 유형

### 4.1 Entity Types (drive.h)

```c
#define NONE        0
#define VEHICLE     1   // 차량 (제어 가능)
#define PEDESTRIAN  2   // 보행자 (제어 가능)
#define CYCLIST     3   // 자전거 (제어 가능)
#define ROAD_LANE   4   // 차선
#define ROAD_LINE   5   // 차선 구분선
#define ROAD_EDGE   6   // 도로 경계
#define STOP_SIGN   7   // 정지 표지판
#define CROSSWALK   8   // 횡단보도
#define SPEED_BUMP  9   // 과속방지턱
#define DRIVEWAY    10  // 진입로
```

### 4.2 분류

**제어 가능 액터 (1-3)**:
- `VEHICLE (1)`: 차량, 정책으로 제어
- `PEDESTRIAN (2)`: 보행자
- `CYCLIST (3)`: 자전거

**도로 요소 (4-6)**:
- `ROAD_LANE (4)`: 주행 가능 차선
- `ROAD_LINE (5)`: 차선 구분선 (백색, 황색)
- `ROAD_EDGE (6)`: 도로 경계 (연석 등)

**특수 요소 (7-10)**:
- `STOP_SIGN (7)`: 정지 표지판
- `CROSSWALK (8)`: 횡단보도
- `SPEED_BUMP (9)`: 과속방지턱
- `DRIVEWAY (10)`: 진입로/출입구

### 4.3 JSON → Binary 타입 매핑

```python
type_mapping = {
    "ROAD_LANE": 4,           # 0-3 → 4
    "ROAD_LINE_BROKEN_SINGLE_WHITE": 5,  # 5-13 → 5
    "ROAD_EDGE_BOUNDARY": 6,  # 14-16 → 6
    "STOP_SIGN": 7,           # → 7
    "CROSSWALK": 8,           # → 8
    "SPEED_BUMP": 9,          # → 9
    "DRIVEWAY": 10,           # → 10
}
```

---

## 5. RL 환경 구조

### 5.1 Reward 함수

**기본 보상 항목** (drive.ini):

| 항목 | 기본값 | 설명 |
|------|--------|------|
| reward_vehicle_collision | -0.5 | 차량 충돌 |
| reward_offroad_collision | -0.5 | 도로 이탈 |
| reward_goal | 1.0 | 목표 도달 (첫 번째) |
| reward_goal_post_respawn | 0.25 | 리스폰 후 목표 도달 |

**보상 계산 로직**:

```c
float reward = 0.0f;

// 1. 충돌 패널티
if (collision_state == VEHICLE_COLLISION) {
    reward += reward_vehicle_collision;  // -0.5
} else if (collision_state == OFFROAD) {
    reward += reward_offroad_collision;  // -0.5
}

// 2. 목표 도달 보상
if (within_goal_radius && within_goal_speed) {
    if (goal_behavior == GOAL_RESPAWN) {
        reward += respawned ? reward_goal_post_respawn : reward_goal;
    } else if (goal_behavior == GOAL_GENERATE_NEW) {
        reward += reward_goal;
        sample_new_goal(agent);
    }
}

// 3. Jerk 패널티 (CLASSIC 모델)
if (dynamics_model == CLASSIC) {
    float jerk = fabsf(delta_v) / dt;
    reward -= 0.0002f * jerk;
}
```

### 5.2 Metrics 추적

```c
float metrics_array[4];
// [0] COLLISION_IDX: 차량 충돌 여부 (0/1)
// [1] OFFROAD_IDX: 도로 이탈 여부 (0/1)
// [2] REACHED_GOAL_IDX: 목표 도달 여부 (0/1)
// [3] LANE_ALIGNED_IDX: 차선 정렬 여부 (±15도 이내)
```

### 5.3 충돌 감지

**AABB 회전 충돌**:
```c
// Oriented Bounding Box 충돌 감지
bool check_collision(Entity *a, Entity *b) {
    // 두 박스의 축에 대해 SAT(Separating Axis Theorem) 적용
    // 각 축에서 겹침 확인
    return overlaps_on_all_axes;
}
```

**충돌 상태**:
```c
#define NO_COLLISION      0
#define VEHICLE_COLLISION 1
#define OFFROAD           2
```

---

## 6. Episode 및 Reset 로직

### 6.1 Episode 시작 (c_reset)

```c
void c_reset(Drive *env) {
    env->timestep = env->init_steps;

    for (int i = 0; i < env->num_active_agents; i++) {
        Entity *agent = &env->entities[env->active_agent_indices[i]];

        // Trajectory에서 초기 위치 설정
        set_start_position(agent, env->init_steps);

        // 상태 초기화
        agent->collision_state = NO_COLLISION;
        agent->respawned = false;
        agent->goals_reached_this_episode = 0;
    }

    // 관찰값 계산
    compute_observations(env);
}
```

### 6.2 Episode 종료 조건

```c
// c_step 함수 내
if (env->timestep >= env->episode_length ||
    (!originals_remaining && env->termination_mode == 1)) {
    add_log(env);
    c_reset(env);
}
```

**termination_mode 옵션**:
- `0`: `timestep >= episode_length`일 때만 종료
- `1`: 모든 에이전트 리스폰 완료 OR episode_length 도달 시 종료

### 6.3 에이전트 리스폰

**goal_behavior에 따른 동작**:

| 값 | 모드 | 동작 |
|----|------|------|
| 0 | GOAL_RESPAWN | 목표 도달 시 초기 위치로 리스폰 |
| 1 | GOAL_GENERATE_NEW | 목표 도달 시 새 목표 생성 |
| 2 | GOAL_STOP | 목표 도달 시 정지 |

**collision_behavior에 따른 동작**:

| 값 | 모드 | 동작 |
|----|------|------|
| 0 | IGNORE | 충돌 무시, 계속 진행 |
| 1 | STOP | 충돌 시 정지 |
| 2 | REMOVE | 충돌 시 제거 |

### 6.4 Active Agent 선택

```c
// should_control_agent 함수
bool should_control_agent(Entity *entity, int control_mode) {
    // 목표까지 거리 확인
    if (distance_to_goal < MIN_DISTANCE_TO_GOAL) return false;

    // 전문가 에이전트 제외
    if (entity->mark_as_expert) return false;

    // control_mode에 따른 필터링
    switch (control_mode) {
        case CONTROL_VEHICLES:
            return entity->type == VEHICLE;
        case CONTROL_AGENTS:
            return entity->type <= CYCLIST;
        case CONTROL_WOSAC:
            return entity->type <= CYCLIST;
        case CONTROL_SDC_ONLY:
            return entity->id == env->sdc_track_index;
    }
}
```

---

## 7. Goal Point 시스템

### 7.1 Goal 설정 방식

**초기 Goal**:
- 맵 바이너리의 `goal_x`, `goal_y`, `goal_z` 사용
- Trajectory의 마지막 유효 위치에서 계산됨

**동적 Goal 생성 (sample_new_goal)**:

```c
void sample_new_goal(Entity *agent, Drive *env) {
    float best_dist_diff = FLT_MAX;

    // 모든 ROAD_LANE 점 순회
    for (int i = 0; i < env->num_roads; i++) {
        Entity *road = &env->roads[i];
        if (road->type != ROAD_LANE) continue;

        for (int j = 0; j < road->array_size; j++) {
            // 에이전트 정면 방향의 점만 선택
            float dx = road->traj_x[j] - agent->x;
            float dy = road->traj_y[j] - agent->y;
            float dot = dx * cosf(agent->heading) + dy * sinf(agent->heading);
            if (dot <= 0) continue;  // 뒤쪽 제외

            float dist = sqrtf(dx*dx + dy*dy);
            float dist_diff = fabsf(dist - goal_target_distance);

            if (dist_diff < best_dist_diff) {
                best_dist_diff = dist_diff;
                agent->goal_position_x = road->traj_x[j];
                agent->goal_position_y = road->traj_y[j];
            }
        }
    }
}
```

### 7.2 Goal 도달 조건

```c
bool within_distance = distance_to_goal < goal_radius;   // 기본 2.0m
bool within_speed = current_speed <= goal_speed;         // 기본 100 m/s

if (within_distance && within_speed) {
    // Goal 도달!
}
```

### 7.3 Goal 관련 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| goal_radius | 2.0 | 목표 인식 거리 (m) |
| goal_speed | 100.0 | 목표 도달 속도 임계값 (m/s) |
| goal_target_distance | 30.0 | 새 목표 생성 거리 (m) |
| goal_behavior | 0 | 0:respawn, 1:generate_new, 2:stop |

---

## 8. Environment 설정

### 8.1 drive.ini 주요 섹션

**[env] - 환경 설정**:
```ini
num_agents = 1024              # 병렬 에이전트 수
action_type = discrete         # discrete / continuous
dynamics_model = classic       # classic / jerk
dt = 0.1                       # 시뮬레이션 타임스텝 (초)
episode_length = 91            # 에피소드 길이 (스텝)

# 보상
reward_vehicle_collision = -0.5
reward_offroad_collision = -0.5
reward_goal = 1.0
reward_goal_post_respawn = 0.25

# 목표
goal_radius = 2.0
goal_speed = 100.0
goal_behavior = 0              # 0:respawn, 1:new, 2:stop
goal_target_distance = 30.0

# 충돌/이탈 동작
collision_behavior = 0         # 0:ignore, 1:stop, 2:remove
offroad_behavior = 0

# 초기화
init_steps = 0
init_mode = "create_all_valid"
control_mode = "control_vehicles"

# 맵
map_dir = "resources/drive/binaries/training"
num_maps = 10000
resample_frequency = 910

# 종료
termination_mode = 1
```

**[vec] - 벡터화 설정**:
```ini
num_workers = 16               # CPU 워커 수
num_envs = 16                  # 워커당 환경 수
batch_size = 4                 # 배치 크기
```

**[train] - 학습 설정**:
```ini
total_timesteps = 2_000_000_000
batch_size = 524288            # ~512K
minibatch_size = 32768
bptt_horizon = 32              # BPTT 길이

# PPO 하이퍼파라미터
gamma = 0.98                   # 할인율
learning_rate = 0.003
ent_coef = 0.005               # 엔트로피 계수
gae_lambda = 0.95
clip_coef = 0.2
vf_coef = 2                    # Value 손실 가중치

# 체크포인트
checkpoint_interval = 1000
render = True
render_interval = 1000
```

**[eval] - 평가 설정**:
```ini
eval_interval = 1000

# WOSAC 평가
wosac_realism_eval = False
wosac_num_maps = 20
wosac_num_rollouts = 32
wosac_init_steps = 10
wosac_control_mode = "control_wosac"
wosac_goal_behavior = 2

# Human-Replay 평가
human_replay_eval = False
human_replay_control_mode = "control_sdc_only"
```

### 8.2 Control Mode 옵션

```c
#define CONTROL_VEHICLES  0    // 차량만 제어
#define CONTROL_AGENTS    1    // 모든 에이전트 제어
#define CONTROL_WOSAC     2    // WOSAC 평가 모드
#define CONTROL_SDC_ONLY  3    // SDC만 제어 (Human-replay)
```

---

## 9. 물리 엔진

### 9.1 CLASSIC 동작 모델 (Kinematic Bicycle)

```c
// 자전거 모델 동역학
float wheelbase = 0.6f * length;
float beta = atanf(0.5f * tanf(steering_angle));

// Yaw rate
float yaw_rate = (signed_speed * cosf(beta) * tanf(steering_angle)) / wheelbase;

// 속도 계산
float new_vx = signed_speed * cosf(heading + beta);
float new_vy = signed_speed * sinf(heading + beta);

// 위치 업데이트
x += new_vx * dt;
y += new_vy * dt;
heading += yaw_rate * dt;

// 속도 업데이트
signed_speed = clamp(signed_speed + acceleration * dt, -MAX_SPEED, MAX_SPEED);
```

### 9.2 JERK 동작 모델

```c
// 상태 범위
float a_long_range[2] = {-5.0f, 2.5f};   // 종가속도 (m/s²)
float a_lat_range[2] = {-4.0f, 4.0f};    // 횡가속도 (m/s²)
float v_range[2] = {-2.0f, 20.0f};       // 속도 (m/s)
float steering_range[2] = {-0.55f, 0.55f};  // 조향각 (rad)

// Jerk 적용
a_long_new = clamp(a_long + jerk_long * dt, a_long_range);
a_lat_new = clamp(a_lat + jerk_lat * dt, a_lat_range);

// 속도/조향 업데이트
v_new = clamp(v + a_long * dt, v_range);
steering_new = clamp(steering + a_lat * k * dt, steering_range);

// 위치 업데이트 (자전거 모델)
float d = 0.5f * (v_new + v) * dt;
float curvature = tanf(steering) / wheelbase;
float theta = d * curvature;
```

### 9.3 충돌 감지 그리드

```c
#define GRID_CELL_SIZE 5.0f

// 공간 해싱
int getGridIndex(float x, float y) {
    int gx = (int)((x - min_x) / GRID_CELL_SIZE);
    int gy = (int)((y - min_y) / GRID_CELL_SIZE);
    return gy * grid_width + gx;
}

// 인접 셀에서 객체 검색
for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
        int neighbor_idx = getGridIndex(x + dx*GRID_CELL_SIZE, y + dy*GRID_CELL_SIZE);
        // 해당 셀의 객체들과 충돌 체크
    }
}
```

---

## 10. 핵심 상수 및 정규화

### 10.1 상수 정의 (drive.h)

| 상수 | 값 | 설명 |
|------|-----|------|
| MAX_AGENTS | 32 | 최대 에이전트 수 |
| TRAJECTORY_LENGTH | 91 | 궤적 길이 (프레임) |
| MAX_SPEED | 100.0 | 최대 속도 (m/s) |
| MAX_VEH_LEN | 30.0 | 최대 차량 길이 (m) |
| MAX_VEH_WIDTH | 15.0 | 최대 차량 폭 (m) |
| MAX_VEH_HEIGHT | 10.0 | 최대 차량 높이 (m) |
| MAX_ROAD_SEGMENT_OBSERVATIONS | 128 | 관찰 도로 세그먼트 수 |
| MAX_ROAD_SEGMENT_LENGTH | 100.0 | 최대 도로 세그먼트 길이 (m) |
| MAX_ROAD_SCALE | 100.0 | 도로 스케일 정규화 |
| MIN_DISTANCE_TO_GOAL | 2.0 | 제어 필터 거리 (m) |
| GRID_CELL_SIZE | 5.0 | 공간 그리드 셀 크기 (m) |
| EPISODE_LENGTH | 91 | 기본 에피소드 길이 |

### 10.2 정규화 요약

| Feature | 정규화 방법 | 결과 범위 |
|---------|------------|-----------|
| 위치 (거리) | × 0.02 또는 × 0.005 | [-1, 1] |
| 속도 | ÷ 100 | [-1, 1] |
| 차량 폭 | ÷ 15 | [0, 1] |
| 차량 길이 | ÷ 30 | [0, 1] |
| 도로 길이 | ÷ 100 | [0, 1] |
| 조향각 | ÷ π | [-1, 1] |
| 방향 | cos/sin | [-1, 1] |

---

## 부록: 명령어 요약

```bash
# 설치
uv venv .venv && source .venv/bin/activate
uv pip install -e .
python setup.py build_ext --inplace --force

# 학습
puffer train puffer_drive

# 평가
puffer eval puffer_drive --eval.wosac-realism-eval True

# 시각화
bash scripts/build_ocean.sh drive local
./drive

# 테스트
pytest tests/
```

---

## 파일 경로 참조

| 용도 | 경로 |
|------|------|
| 코어 시뮬레이터 | `pufferlib/ocean/drive/drive.h` |
| Python 래퍼 | `pufferlib/ocean/drive/drive.py` |
| 환경 설정 | `pufferlib/config/ocean/drive.ini` |
| 맵 바이너리 | `pufferlib/resources/drive/binaries/` |
| CARLA 생성기 | `data_utils/carla/generate_carla_agents.py` |
| 벤치마크 | `pufferlib/ocean/benchmark/` |

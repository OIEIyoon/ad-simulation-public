---
title: ViLS - Traffic Simulation Server
layout: default
parent: Tasks
nav_order: 1
has_children: true
---

# ViLS - Traffic Simulation Server

실차(Ego)가 가상 NPC들과 상호작용하며 자율주행 테스트할 수 있는 시뮬레이션 서버

---

## Tasks

### Server Infrastructure
- [x] Simulation Engine (Vehicle Dynamics, Bicycle Model)
- [x] Map & Road Network (FMTC → Lane Graph, Route Generation)
- [x] GUI Visualization (Pygame 기반 렌더러)
- [x] WebSocket API (FastAPI, 10Hz)

### NPC Policy
- [x] Rule-based Policy (IDM + Lookahead) - 서버 인프라 검증용
- [ ] Learned Policy (SMART 등) - Neural Planner 통합

### Ego Integration
- [ ] 실차 상태 주입 (Position, Velocity, Heading)
- [ ] NPC의 Ego 인식 및 반응
- [ ] 시나리오 관리

---

## Subtasks

- [IDM 기반 Rule-based Policy](vils-server-progress-2025-12-26) (2025-12-26)

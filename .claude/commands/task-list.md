---
description: Task 목록 조회 및 index.md 동기화
allowed-tools: Read, Glob, Write, Edit
---

# Task 목록 관리

1. `/home/oiei/ad-simulation-public/tasks/` 디렉토리의 모든 .md 파일 조회
2. 각 파일에서 제목과 Objective 섹션 추출
3. 테이블 형태로 출력

## 출력 형식

| Task | Objective | Link |
|------|-----------|------|
| Task 이름 | 목표 요약 | [상세](tasks/xxx.md) |

필요시 `/home/oiei/ad-simulation-public/index.md` 자동 갱신.

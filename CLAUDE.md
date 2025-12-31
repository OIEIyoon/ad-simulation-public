# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jekyll 기반 GitHub Pages 사이트로, 자율주행 시뮬레이션 연구 문서를 작성하고 publish하는 저장소.
- **Theme**: Just the Docs (remote theme)
- **배포**: GitHub Pages 자동 배포 (push → 자동 빌드)

## Directory Structure

```
/
├── tasks/           # 연구 Task 문서 (활성 작업)
├── notes/           # 연구 노트 (분석, 비교 문서)
├── _includes/       # HTML snippets (MathJax, Giscus)
├── _sass/custom/    # 커스텀 스타일링
└── .claude/         # Claude Code 설정
    ├── agents/      # AI 에이전트 정의
    └── commands/    # 슬래시 커맨드 (/task-new, /task-list)
```

## Available Commands

- `/task-new <task-name>`: 새 Task 문서 생성 (`tasks/<task-name>.md`)
- `/task-list`: Task 목록 조회 및 index.md 동기화

## Writing Conventions

### 문체
- **개조식, 계층적 구조**
- **명사형 종결** (~다, ~요 지양)
  - Good: "학습 기반 정책 적용"
  - Bad: "학습 기반 정책을 적용합니다"
- 간결하게, 불필요한 수식어 제거

### 시각화
- **Mermaid 다이어그램**: flowchart, sequenceDiagram, graph
- **테이블**: 옵션 비교, 결과 정리
- **LaTeX**: `$...$` (inline), `$$...$$` (display)

### Front Matter

```yaml
---
title: [제목]
layout: default
parent: Tasks  # 또는 Notes
nav_order: [순서]
---
```

## Task Document Template

```markdown
# [Task 이름]

## Objective
- 핵심 목표 (1-2줄)

## Tasks
- [ ] Task 1
- [ ] Task 2

## Details

### Task 1
- 상세 내용

### Task 2
- 상세 내용
```

## Technical Notes

- TOC 불필요 - Just the Docs 자동 생성
- MathJax: `_includes/head_custom.html`에서 설정
- Giscus 댓글: `_includes/footer_custom.html`에서 설정

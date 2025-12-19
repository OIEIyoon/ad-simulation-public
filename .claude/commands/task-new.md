---
description: 새 연구 Task 문서 생성
argument-hint: <task-name>
allowed-tools: Write, Read, Glob, Edit
---

# Task 문서 생성

Task 이름: $ARGUMENTS

`./tasks/$ARGUMENTS.md` 파일을 아래 템플릿으로 생성.

## 템플릿

```markdown
---
title: [Task 이름]
layout: default
parent: Tasks
nav_order: [다음 순서]
---

# [Task 이름]

[한 줄 설명]

## Tasks

- Task 1
- Task 2
- Task 3

---

## Details

### Task 1

- 상세 내용

### Task 2

- 상세 내용

### Task 3

- 상세 내용
```

## 작성 규칙
1. Tasks: 해야 할 일을 간단명료하게 나열
2. Details: 각 Task의 상세 내용
3. 개조식, 명사형 종결
4. 간결하게 작성

## 후처리
파일 생성 후 `./index.md`의 Task 테이블에 새 항목 추가.

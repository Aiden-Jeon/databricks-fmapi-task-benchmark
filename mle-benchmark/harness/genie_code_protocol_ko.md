# L4 — Genie Code 실행 프로토콜 (S2)

## 자동화 (권장): Playwright 드라이버
`harness/genie_l4_playwright.py`가 이 프로토콜을 UI 자동화로 실행합니다.
Genie Code는 공개 API/CLI가 없어 브라우저를 사람처럼 조작합니다.

1. 최초 1회 로그인 캡처: `.venv/bin/python harness/genie_l4_login.py`
   (헤디드 브라우저에서 SSO/MFA 완료 → 세션이 `.secrets/genie_state.json`에 저장)
2. 최초 실행은 셀렉터 보정: `... genie_l4_playwright.py --tasks t3_ynat --calibrate`
   (개발 머신에서 Genie Code DOM을 볼 수 없어, 각 UI 단계에서 Playwright
   inspector로 셀렉터를 확인/수정 → `.secrets/genie_selectors.json`에 저장)
3. 이후 무인 배치: `... genie_l4_playwright.py --tasks t1_pubg t2_spooky ...`
   (pack 스테이징 → 스레드 생성 → auto-approve → KR 킥오프 → submission.csv
   폴링(2h cap) → 산출물/스레드/타이밍 저장). private/ 답안은 절대 스테이징 안 함.

전제조건: 워크스페이스에 Genie Code(Full page, Beta)가 활성화되어 있어야 함.
2026-07-31 확인: fevm-newjeans-ontos에서 활성화 여부 미확인 → 사용자 UI 확인 필요.

## 수동 대체 (Playwright 실패 시)

아래는 사람이 직접 수행하는 반자동 절차입니다. 공정성을 위해 아래 절차를 벗어나는 개입은
금지됩니다. 과제당 1회, 총 5세션.

## 사전 준비 (1회)
1. 워크스페이스 `fevm-newjeans-ontos`에서 Genie Code(Full page, Beta) 사용
   가능 여부 확인 — 우측 상단 ✨ 아이콘 또는 좌측 내비게이션.
2. 작업 폴더 준비: 워크스페이스 홈에 `kmle_genie/<task_id>/` 생성 후 해당
   과제의 pack 파일(spec.md, train.csv, test.csv, sample_submission.csv)을
   volume에서 복사. (private/ 은 절대 복사 금지.)

## 세션 절차 (과제당)
1. 시계 시작 (기록: 시작 시각).
2. Full-page Genie Code 열기 → 새 스레드 → 컨텍스트를 해당 작업 폴더로 지정.
3. 승인 모드: **"Allow in current thread"** (auto-approve) 선택.
4. 표준 킥오프 프롬프트(`kickoff_prompt_ko.md` 전문)를 붙여넣고 전송.
   앞에 한 줄 추가: "작업 폴더: kmle_genie/<task_id>/ — 이 폴더 안에서만
   작업하십시오."
5. **개입 금지.** 질문을 받아도 추가 정보 제공 금지 — 유일하게 허용되는
   응답: "spec.md에 있는 정보로 진행하세요." (회수 기록)
6. 종료 조건: (a) outputs/submission.csv 생성 완료 선언, (b) 2시간 경과,
   (c) 스레드가 진행 불능 상태. 시계 종료 (기록: 종료 시각).
7. `outputs/submission.csv`, 생성된 코드, 스레드 전체 내용(내보내기/캡처)을
   `/Volumes/newjeans_ontos_catalog/kmle_results/artifacts/L4_<task_id>_<ts>/`
   에 복사.

## 기록 항목 (세션당)
- 시작/종료 시각, 총 소요 시간
- 허용 응답("spec대로 진행") 횟수
- submission 생성 여부
- 스레드 링크

## 비용 산정 (S2 확인 사항)
- Genie Code는 2026-07-08부터 PAYG + 사용자별 무료 한도.
- 실행 후 `system.billing.usage`에서 해당 시간 창의 GENIE 관련 SKU 행 확인:
  `usage_metadata`, `billing_origin_product` 필터로 세션 창(시작~종료)과
  대조. 과제 단위 귀속이 불가하면 세션 시간 창 기준 근사치로 보고하고
  리포트에 명시.

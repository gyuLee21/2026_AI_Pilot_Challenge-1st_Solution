# 출처와 권리

| 구성 요소 | 출처 · 취급 |
|---|---|
| `cuda_fdm`, `claude_code`, `src/dogfight` | 팀 개발 코드. 대회 제공 시뮬레이터 · 정책 · 통신 코드에서 출발한 부분을 포함합니다 |
| `cuda_fdm/ref`, `cuda_fdm/gen` | JSBSim v1.0.0 F-16 모델을 Python/CUDA로 재구현. 원본 모델 데이터의 출처 표시를 유지합니다 |
| `controllers/mpc/release` | 팀 MPC 번들의 소스 · 설정 · 테스트. 빌드 결과물은 제외 |
| `controllers/bt/.../BT_Content` | 대회 SDK에 적용하는 커스텀 노드 소스. BehaviorTree.CPP 등 SDK 본체는 포함하지 않습니다 |
| `research/competition_36500/runtime` | 최종 3-9 학습에 사용한 원본 소스. 파일별 SHA-256을 `source_manifest.json`에 기록 |
| `third_party/competition_runtime` | 대회 제공 DLL · wrapper · F-16 데이터를 원본 바이트로 보관. [NOTICE](../third_party/competition_runtime/NOTICE.md) 참고 |
| `artifacts/` | 가중치 · 팀원 모델 · 실행 로그 등 로컬 전용 자료. 배포하지 않습니다 |

이 저장소에는 별도의 오픈소스 라이선스를 지정하지 않았습니다.
기존 소스의 저작권 표시는 그대로 유지하며, 외부 자료는 원 권리자의 조건을 따릅니다.

## 저장소만으로 가능한 범위

| 가능 | 추가로 필요한 것 |
|---|---|
| 소스 테스트, 리그 상태 로직 검사 | — |
| 새 학습 실행 | CUDA GPU 환경 |
| 과거 학습 이어가기 | 원본 체크포인트와 상대 archive |
| CPU 교전 평가 | 대회 SDK 런타임, 모델 파일 |
| MPC native 테스트 · BT 빌드 | 빌드 도구, 원본 SDK |
| 제출 실행파일 | [Releases](https://github.com/gyuLee21/2026_AI_Pilot_Challenge-1st_Solution/releases)에서 제공 |

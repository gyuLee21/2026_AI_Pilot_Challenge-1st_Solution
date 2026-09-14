# Behavior-tree controllers

`aip_dcs/BehaviorTree/BT_Content/`는 기존 AIP_DCS 프로젝트의 커스텀 노드 소스입니다.

- `Competition/`: 위협·교전 기하 조건과 기동 선택
- `BlackBoard/`: 관측 및 기동 상태 공유
- `Decorator/`, `Service/`, `Task/`: 조건 검사, 상태 갱신, 실행 노드
- `Functions.*`: 공통 BT 계산

이 폴더는 **독립 실행 프로그램이 아닌 SDK 소스 overlay**입니다.
원본 SDK의 같은 상대 경로에 적용해 AIP_DCS 프로젝트에서 빌드합니다.
BehaviorTree.CPP 헤더/구현, Geometry, 프로젝트 파일 등 원본 의존성이 별도로 필요합니다.
이번 정리에서 DLL을 새로 빌드하거나 BT 성능을 다시 검증한 것은 아닙니다.

대회 SDK 전체나 다른 팀의 BT DLL을 복제하지 않았습니다.
테스트 상대 바이너리/XML은 로컬 `artifacts/models/bt/`에서 따로 관리합니다.
폴더명만으로 특정 DLL이 이 소스에서 만들어졌다고 단정하지 마세요.
기반 SDK와 커스텀 수정 코드가 섞여 있으므로 외부 공개 전 출처/라이선스 검토가 필요합니다.

# MPC controllers

`release/`는 기존 `Release_MPC_team_share` 개발 번들의 **소스 전용 사본**입니다.
CEM 기반 2초 예측, 10Hz 재계획, 60Hz 제어 및 C++ reduced-order F-16 predictor를 포함합니다.
원본 설정과 소스의 디렉터리 상대 경로를 유지했습니다.

```powershell
cd controllers/mpc/release
python -m pip install -r requirements.txt
tools\build_native.cmd
python -m pytest tests -q
python student/my_submission.py --help
```

빌드는 Visual Studio 2022 C++와 CMake 3.20 이상이 필요합니다.
빌드 도구 경로는 `tools/build_native.cmd`에서 확인하세요.
생성 DLL은 `runtime/predictor/Release/`에 놓이고 Git에서는 제외됩니다.
원본 번들의 문서에 있는 “prebuilt DLL 포함” 설명은 원본 배포 패키지에 대한 것으로,
**이 소스 저장소에는 해당 바이너리가 없습니다**.

자세한 내용: [구조](release/docs/ARCHITECTURE.md),
[빌드/연동](release/docs/BUILD_AND_INTEGRATION.md), [기존 검증 기록](release/docs/VALIDATION.md).
ROM/NMPC 실험 원본은 로컬 `artifacts/models/mpc/rom_nmpc_snapshot`에 보존합니다.
실험 결과·자동 생성 solver·SDK 의존 코드를 검증 없이 제품 코드처럼 올리지 않습니다.

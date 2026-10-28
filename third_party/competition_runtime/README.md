# Competition-provided native runtime (Windows x64)

대회에서 제공받아 보존한 DLL 3개, Python wrapper 3개, F-16/engine XML 및
기본 rule을 원본 바이트 그대로 분리 보관한다. 전체 SDK·Unreal 서버·다른 팀 정책은 아니다.
DLL 합계 약 8.92 MB, asset/wrapper 포함 약 9.03 MB이므로 Git LFS 없이 보관한다.

| 파일 | 역할 |
|---|---|
| `win64/JSBSimAIPLib.dll` | Native CPU FDM / simulator API |
| `win64/AIP_BASE.dll` | 대회 기본 rule-based pilot |
| `win64/AIP_BASE_target.dll` | target 슬롯용 기본 pilot; 이 버전은 위 DLL과 동일 바이트 |
| `python/` | 제공된 wrapper; 프로젝트에서 수정한 평가 wrapper와 구분 |
| `assets/` | F-16 aircraft, engine XML, 기본 rule 및 기존 설명/출처 |

DLL 3개는 `idearendil/2026_AI_Pilot_Challenge-1st_Solution`의 commit
`e677b4bdf93947663501224d5c9122bf71c9d37a`에 공개된 동일 이름 파일과 Git blob SHA-1이
일치한다. 출처와 파일별 SHA-256은 [manifest.json](manifest.json)에 있다.
공개 사본과 일치한다는 사실은 재배포 라이선스가 확인됐다는 뜻이 아니다.
[NOTICE](NOTICE.md)를 읽고 원 권리자의 조건을 준수해야 한다.

```powershell
# Default: hashes only. Does not load DLLs, download files or create a runtime.
python scripts/prepare_runtime.py

# Explicitly create an isolated local directory; different existing files are preserved.
python scripts/prepare_runtime.py --install
```

설치 대상은 기본 `artifacts/runtimes/competition/`다. 기존 `cpu_baselines/`와
학습/평가 프로세스의 runtime은 덮어쓰지 않는다. 필요한 모델·BT rule·정책 어댑터는 별도다.
준비 성공은 DLL 로딩, CRT 설치, 실제 교전 성공을 의미하지 않는다.
일부 배포 DLL은 Visual C++ debug CRT에 의존할 수 있으며 해당 런타임을 이 저장소에서
임의 복제하지 않는다. 문제가 있으면 제공 환경/SDK 문서에 따른 의존성을 확인한다.

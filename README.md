# Market AD Simulator

전통시장(전주 남부시장 모티프) 누전(I_leak) · 누수(Q) 시계열 합성 데이터셋과 Streamlit 인터랙티브 대시보드.
**Multivariate 시계열 이상탐지 (TSAD)** 학습/검증용 데이터 생성을 목표로 함.

- **11개 동(D1~D10 + 청년몰), 점포 ~120곳, 점포당 6채널 + 동/분전반 단위 집계**
- 1초 base sampling, 1m/30m/1h 다운샘플 export
- 핫트리거 한 번에 spike / 점진 증가 패턴이 무작위 다중 anomaly로 주입
- 도메인 임계치(점포·SDP·MDP trip / 누수 임계) 위반 시점을 anomaly 시작점으로 시각화
- 출력 포맷: TSB-AD-M wide CSV + `label.csv` + `anomaly_index.json`

---

## 0. 요구사항

- Python **3.10 이상** (3.11 권장)
- pip
- (선택) GPU 불필요, 모든 연산 CPU/NumPy

---

## 1. 설치 — Linux / macOS

```bash
# 저장소 클론
git clone https://github.com/<USER>/Market_simulatior.git
cd Market_simulatior

# 가상환경
python3 -m venv .venv
source .venv/bin/activate

# 의존성
pip install --upgrade pip
pip install -r requirements.txt
```

비활성화: `deactivate`

## 1-W. 설치 — Windows

### PowerShell

```powershell
# 저장소 클론
git clone https://github.com/<USER>/Market_simulatior.git
cd Market_simulatior

# 가상환경
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 정책 오류 시 (PowerShell이 스크립트 차단할 때, 1회만)
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 의존성
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Command Prompt (cmd.exe)

```cmd
git clone https://github.com/<USER>/Market_simulatior.git
cd Market_simulatior

py -3 -m venv .venv
.venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt
```

비활성화: `deactivate`

> Windows에서 한글 출력이 깨질 때는 PowerShell/cmd에서 `chcp 65001` 한 번 실행.

---

## 2. Streamlit 대시보드 실행 (메인 사용처)

### Linux / macOS

```bash
source .venv/bin/activate
streamlit run dashboard/app.py
```

### Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```

### Windows (cmd)

```cmd
.venv\Scripts\activate.bat
streamlit run dashboard/app.py
```

브라우저: <http://localhost:8501>

### 외부 포트 노출 (원격 머신)

```bash
streamlit run dashboard/app.py --server.port 8765 --server.address 0.0.0.0
```

### 대시보드 사용 흐름

1. 좌측 사이드바 → 기간/seed 정하고 **▶ 시뮬레이션 실행** 클릭
2. 상단 토폴로지(시장 지도/계통 트리) 보기 전환
3. **좌측 핫트리거** 패널에서 동별 밸브 차단 / 분전반 트립 클릭
   - 한 번 클릭 시 2~4건의 anomaly가 spike / 점진증가 패턴으로 랜덤 주입
   - 시간상 겹치지 않도록 자동 배치, 타임라인 중간 80% 구간 내
   - 임계치 crossing 시점부터 빨간 음영 시작
4. **우측 차트** 3종(누전 / 누수 / 환경)에서 실시간 결과 확인
   - 사이드바 선택이 트리거 대상 패널로 자동 점프
5. 하단 이상 인스턴스 표 + CSV 다운로드

---

## 3. CLI로 데이터 생성 (대용량 일괄)

대시보드 없이 곧장 CSV를 만들고 싶을 때:

```bash
# 기본 — 30일, seed=42, 1m/30m/1h 마켓 wide CSV
python -m src.simulator

# 빠른 확인 — 3일치, 1m·30m만, 자동 anomaly 없음 (clean baseline)
python -m src.simulator --days 3 --seed 42 --export-freqs 1m,30m

# 시나리오 기반 자동 anomaly 주입
python -m src.simulator --days 7 --inject-anomalies

# 전 점포 1s 덤프 (수십 GB, 디스크 주의)
python -m src.simulator --days 7 --store-level
```

Windows에서도 동일. `py -m src.simulator ...`도 가능.

### CLI 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--config` | `config/` | YAML 설정 디렉터리 |
| `--out` | `data/` | 출력 루트 |
| `--days` | 30 | 시뮬레이션 기간(일) |
| `--seed` | 42 | 난수 시드 |
| `--export-freqs` | `1m,30m,1h` | 마켓 wide export 주기 |
| `--store-level` | off | 모든 점포의 1s wide CSV 덤프 |
| `--store-sample` | 3 | (`--store-level` 없을 때) 샘플 점포 수 |
| `--inject-anomalies` | off | 시나리오 기반 자동 anomaly 적용 |

### 출력 구조

```
data/tsb_format/
  market_1m.csv          # (timestamp, store__channel ..., dong__channel ...)
  market_1m.label.csv    # binary anomaly label per timestamp
  market_30m.csv, ...
  store_<sid>_1s.csv     # 점포별 6채널 1s
  store_<sid>_1s.label.csv
data/meta/
  anomaly_index.json     # 모든 instance 메타 (scenario_type, severity 등)
```

---

## 4. 테스트

```bash
pytest tests/ -v
```

`tests/test_smoke.py`가 1일치 end-to-end 시뮬레이션 + export까지 검증.

Windows에서도 동일.

---

## 5. 설정 변경

`config/` 디렉터리의 YAML만 바꿔 재실행. 코드 수정 불필요.

- `config/market_topology.yaml` — 동/분전반/점포 수, 배관 길이, MDP/SDP trip 임계, 시작 날짜
- `config/store_profiles.yaml` — 업종 mix, 부하/유량, 점포 임계, 환경(기온/습도/장마), 주말 가중치
- `config/anomaly_scenarios.yaml` — 시나리오 가중치, 지속 시간, 강도 (CLI `--inject-anomalies`용)

대시보드는 YAML mtime을 캐시 키로 쓰므로, 파일 저장 후 **▶ 시뮬레이션 실행** 다시 누르면 반영.

---

## 6. 도메인 모델

### 신호 채널 (per 점포)

| 채널 | 단위 | 정상 범위 | 임계 |
|---|---|---|---|
| `I_leak` | mA | 0.3 ~ 5 | 점포 15 (습식) / 30 (건식) |
| `I_load` | A | 0.5 ~ 20 | — |
| `Q` | L/min (계단형, 0.5 step) | 0 ~ 5 | — |
| `P` | bar | 2 ~ 4 | 저압 2.0 |
| `T_env` | °C | 21 ~ 33 | — |
| `H_env` | %RH | 35 ~ 92 | 80 (위험) |

### 집계 신호

- `I_leak_panel` — 분전반(SDP)별 누설 합 → 임계 **100 mA**
- `I_leak_dong` — 동(MDP)별 누설 합 → 임계 **300 mA**
- `Q_main` — 동 메인 유량 합 → 임계 **점포수 × 8 L/min**
- `P_main` — 동 평균 압력

### Anomaly 주입 메커니즘 (핫트리거)

- `🔒 밸브 차단` → 대상 동의 모든 점포 Q 채널에 패턴 적용 (`VALVE-{dong}-{seq}`)
- `⚡ 트립` → 대상 분전반 모든 점포의 I_load + I_leak에 패턴 적용 (`BREAK-{panel}-{seq}`)
- 패턴 2종: **`spike`** (가우시안 ↑), **`ctx_inc`** (선형 점진 증가)
- 클릭당 2~4건 랜덤 주입, 시간 비중첩, 중간 80% 구간 클램프
- 빨간 음영의 시작점 = 신호가 해당 채널의 임계치를 처음 넘는 시점 (`_first_crossing`)

---

## 7. 코드 구조

```
config/                  YAML 설정 3종
src/
  topology.py            Market/Dong/Panel/Store + networkx 그래프
  baseline.py            daily/weekly/seasonal + AR(1) noise + Q 계단형 양자화
  anomaly.py             시나리오 기반 (A1/A2/A3/N1)
  simulator.py           end-to-end 루프 + CLI (--inject-anomalies)
  exporter.py            TSB-AD-M wide CSV + 다운샘플
  viz.py                 Plotly 토폴로지/시계열/임계 라인
dashboard/app.py         Streamlit UI (핫트리거 + 시계열 + KPI)
tests/test_smoke.py      end-to-end 스모크
market_ad_simulator_plan_v0.2.md  설계 문서
```

---

## 8. 트러블슈팅

| 증상 | 원인/해결 |
|---|---|
| Streamlit이 시작 안 됨 | venv 활성화됐는지 확인. `streamlit --version`으로 확인 |
| `ModuleNotFoundError: src` | 프로젝트 루트에서 실행해야 함. CLI는 `python -m src.simulator` |
| 차트가 안 바뀜 | 사이드바 **▶ 시뮬레이션 실행** 다시 클릭 (캐시 무효화) |
| Windows에서 venv 활성화 안 됨 | PowerShell 정책 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| 한글 깨짐 (Windows cmd) | `chcp 65001` |
| 메모리 부족 | `--days` 줄이기. 14일 이상이면 CLI 권장 |

---

## License

(설정 없음 — 기본적으로 모든 권리 보유)

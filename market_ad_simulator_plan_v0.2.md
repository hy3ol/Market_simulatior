# 전통시장 누전·누수 TSAD 데이터셋 & 시뮬레이터 구현 계획서

**버전:** v0.2 (슬림)
**작성일:** 2026-05-13
**변경 사항 (vs v0.1):**
- 1ms/30ms 샘플링 제거 → 1s base + 다운샘플로 통합
- 간이 hydraulic/electrical 물리 솔버 제거 → 통계 주입 휴리스틱
- 이상 시나리오 9종 → 4종 (채널-무관 패턴 라이브러리)
- 마일스톤 10 → 5
- 미확정 항목 → 합리적 default로 결정

---

## 1. 개요

전통시장 환경에서 **누전(15/30 mA) 및 누수(MNF 1.0 m³/hr·km)** 모사 합성 시계열을 생성하고, 외란 주입을 시각화하는 Streamlit 대시보드를 만든다.

- **데이터셋**: TSB-AD-M 호환 포맷, Point + Range 라벨, 1개월 분량
- **시뮬레이터**: Streamlit 대시보드, 외란 토글 → 시계열 변화 → CSV 다운로드
- **접근**: 통계/현상학 only (물리 솔버 없음). 충실도보다 **AD 평가용 합리적 합성**에 집중.

---

## 2. 도메인 모델

### 2.1 대상 토폴로지
| 단위 | 점포 수 | 전기 공급 |
|---|---|---|
| 6동 | 34 | 변전실 경유 |
| 8동 | 15 | 변전실 경유 |
| 청년몰 | 25 (default, 추후 조정) | 개별 분전함 |
| **합계** | **74** | — |

> 1차 구현은 6/8/청년몰만. 나머지 동(1~5동)은 추후 확장.

### 2.2 전기 토폴로지 (다층)
```
[동 집합계량기] ─── 3채널 (6동, 8동 변전실 + 청년몰 가상)
       │
       └─ [점포 분전함] ─── 74채널
```
정상 시 ∑(점포) ≈ 동 집계 제약 활용.

### 2.3 수도 토폴로지
- 점포별 + 동별 메인
- 배관 길이: `topology.yaml`에 default 부여 (점포당 10 m, 동 main 200 m)
- 6동-8동 인접 연결관 50 m (밸브 시나리오용)

### 2.4 점포 업종 mix (default)
| 업종 | 비율 | 전기 | 수도 | 누전 임계 |
|---|---|---|---|---|
| 식당/분식 | 25% | 고 | 고 | **15 mA** (습기) |
| 정육/수산 | 10% | 중 | 중-고 | **15 mA** (습기) |
| 청과/채소 | 15% | 저 | 중 | 30 mA |
| 의류/잡화 | 20% | 저 | 저 | 30 mA |
| 떡/제과 | 10% | 중 | 중 | 30 mA |
| 야간업종 | 8% | 중-고(야간) | 중(야간) | 30 mA |
| 기타 | 12% | 저 | 저 | 30 mA |

---

## 3. 측정 채널 (단순화)

### 3.1 점포 채널 (점포당 5개)
| 채널 | 단위 | 비고 |
|---|---|---|
| I_leak | mA | 누설전류 |
| I_load | A | 부하전류 |
| Q | L/min | 유량 |
| P | bar | 압력 |
| T_env | °C | 환경 온도 (점포 그룹 공유) |
| H_env | %RH | 환경 습도 (점포 그룹 공유) |

### 3.2 동 단위 추가
- 동 집합 I_leak_sum (합산)
- 동 main Q_main, P_main

### 3.3 총 채널 수
- 점포 채널: 74 × ~5 ≈ 370
- 동 단위: 3동 × 3 ≈ 9
- **합계 ~380 채널** (사용자 요청 "다변량일수록 좋아" 충족)

---

## 4. 샘플링 (단순화)

**Base = 1s**, 1개월(약 260만 샘플/채널) 연속 생성.

다른 주기는 1s에서 다운샘플:
| 주기 | 생성 방식 | 용도 |
|---|---|---|
| 1s | base | 고해상도 분석 |
| 30s | 평균/중앙값 집계 | |
| 1m | 평균/중앙값 집계 | TSB-AD-M 기본 |
| 30m | 평균 집계 | 트렌드 |
| 1h | 평균 집계 | MNF 분석 |

1ms/30ms 제거 → ring buffer, 이벤트 트리거 캡처 로직 전체 삭제.

---

## 5. 정상 Baseline 생성

각 채널은 다음 합성으로 생성:
```
y(t) = baseline_const
     + daily(t)        ← 업종별 일중 곡선 (사인/sigmoid 조합)
     + weekly(t)       ← 평일/주말/휴무일
     + seasonal(t)     ← 외기온/습도와 결합 (1개월 내 점진 변화)
     + noise(t)        ← AR(1) + Gaussian
```

- **외기온/습도**: 동 단위 공유 외란 (점포-점포 상관성 부여)
- **인버터 부하 점포**: 일부 점포에 inrush 패턴 부여 (N1 시나리오 base)
- **시작 계절 default**: 여름(7월) — 장마 + 누전 위험 자연스럽게 표현 가능

---

## 6. 외란(이상) 시나리오 (4종)

**채널-무관 패턴 라이브러리.** "누전이냐 누수냐"는 주입 채널로 결정.

| ID | 패턴 | 수식 (개략) | 라벨 | 비고 |
|---|---|---|---|---|
| **A1** | Drift | y' = y + α·(t−t0)/Δ | Range | 점진 상승, 시간당 ε 증가 |
| **A2** | Spike/Burst | y' = y + β·exp(−(t−t0)²/2σ²) | Point + Range | 급격 펄스 (≤수 분) |
| **A3** | Intermittent | y' = y + γ·1[조건] | Range (간헐) | 습도 H>80% & 야간일 때만 ON |
| **N1** | Inverter false-trip | I_load spike + I_leak 동반 + 차단기 OFF | Range, **별도 클래스** | "이상이지만 누전 아님" → false-alarm 평가용 |

### 6.1 주입 규칙
- 1개월 데이터, 점포별 0~3개 시나리오 무작위 주입
- 비율 default: A1:A2:A3:N1 = 4:3:2:1
- 누전(I_leak)/누수(Q) 채널 비율은 5:5
- 각 시나리오 ID당 최소 10 instance 보장

### 6.2 라벨 스키마 (TSB-AD-M 호환)
- `label.csv`: `timestamp, label` (binary 0/1)
- `meta.json`: 각 anomaly instance의 `{id, channel, start, end, severity, scenario_type}` 보존
- N1은 label=1 + `scenario_type=nuisance`로 구분

### 6.3 임계 트리거
- 누전: I_leak ≥ 15 mA (습기 업종) or 30 mA (그 외) → "임계 초과" flag
- 누수: 야간 00–04시 Q ≥ 1.0 × length(km) → "임계 초과" flag
- 임계 초과 ≠ label=1. 라벨은 시나리오 주입 구간으로 결정. (임계는 baseline AD 비교용 reference signal)

---

## 7. 시뮬레이터 아키텍처

순수 통계 주입. 물리 솔버 없음.

```python
for t in timesteps:                          # 1s tick
    env = update_environment(t)              # 외기온/습도
    for store in market.stores:
        y = store.profile.sample(t, env)
        for ev in active_events(t, store):
            y = ev.perturb(y, t)             # A1/A2/A3/N1
        store.state = y

    # 동 집계 (단순 합)
    aggregate_dong_meters(market)

    # 밸브 토글 휴리스틱 (대시보드용)
    if valve_closed(store):
        store.Q = 0
        # 인접 점포에 +α 분배 (휴리스틱)

    record(t, all_channels)
```

---

## 8. Streamlit 대시보드

### 8.1 화면 (3-pane)
```
┌─────────────────────────────────────────────────────────────┐
│ [좌: 제어판]            │ [중앙: 토폴로지 + 시계열]          │
│  - 시뮬레이션 시작/정지  │  ┌──── 동·점포 네트워크 그래프 ──┐│
│  - 시간 슬라이더         │  │ 상태 색상: 정상/주의/이상     ││
│  - 동/점포 선택          │  └────────────────────────────────┘│
│  - 외란 주입 패널        │  ┌──── 채널 시계열 (Plotly) ────┐│
│     · 시나리오 (A1~N1)   │  │ 선택 점포의 5채널 + 이상 음영 ││
│     · 대상 점포          │  └────────────────────────────────┘│
│     · 채널 (전기/수도)   │                                     │
│     · 심각도 슬라이더    │ [우: 통계]                          │
│     · 주입 버튼          │  - 임계 초과 점포 수               │
│  - 밸브/차단기 토글      │  - 활성 시나리오 리스트            │
│  - CSV 다운로드 버튼     │  - 동별 집계                       │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 핵심 인터랙션
1. **A2 주입**: 점포 6-012 + 채널 I_leak + 심각도 0.8 → spike → 토폴로지 빨간색
2. **밸브 닫기**: 8동 메인 → 하류 점포 Q=0 → 인접 6동 점포 약간 상승
3. **N1 데모**: inverter inrush spike → 차단기 OFF → "nuisance" 라벨 표시
4. **CSV 다운로드**: 누적 데이터 + 라벨 zip 받기

### 8.3 성능 메모
- 380 채널 실시간 plot 무거움 → **선택된 점포만 상세 plot**, 나머지는 토폴로지 색상으로만
- 시뮬레이션은 batch 모드 권장 (전체 1개월 미리 생성 후 슬라이더로 재생)

---

## 9. 출력 (TSB-AD-M 호환)

```
data/tsb_format/
├── market_6dong_1m.csv             # 동 통합, 1m 주기
├── market_6dong_1m.label.csv
├── market_6dong_store012_1s.csv    # 점포별, 1s 주기
├── market_6dong_store012_1s.label.csv
└── ...

data/meta/
└── anomaly_index.json              # 모든 instance 메타
```

- 컬럼: `timestamp, ch_1, ..., ch_N` + 별도 label 파일
- TSB-AD-M의 6개 지표(AUC-PR/ROC, VUS-PR/ROC, Std-F1, PA-F1) 그대로 계산 가능

---

## 10. 코드 구조 (슬림)

```
market-ad-simulator/
├── README.md
├── requirements.txt
├── config/
│   ├── market_topology.yaml
│   ├── store_profiles.yaml
│   └── anomaly_scenarios.yaml
├── src/
│   ├── topology.py          # Market, Store, networkx 그래프
│   ├── baseline.py          # 정상 패턴 (daily/weekly/seasonal)
│   ├── anomaly.py           # A1/A2/A3/N1 4종
│   ├── simulator.py         # 메인 루프
│   ├── exporter.py          # TSB-AD-M 포맷
│   └── viz.py               # 정적 시각화 헬퍼
├── dashboard/
│   └── app.py               # Streamlit
├── data/
└── notebooks/
    ├── 01_topology_check.ipynb
    ├── 02_baseline_check.ipynb
    └── 03_anomaly_examples.ipynb
```

8개 파일 + config 3개. 매우 가벼움.

---

## 11. 마일스톤 (5단계, ~10일)

각 M 끝에 sanity-check 노트북으로 검증하고 다음으로 이동.

### M1. 토폴로지 + 정상 Baseline (3일)
- `config/*.yaml` 작성
- `topology.py`, `baseline.py` 구현
- 점포별 1개월 정상 시계열 생성 (1s base)
- 노트북 01, 02
- ✅ 검증: 일중/주중/계절 패턴 시각, 점포 간 상관

### M2. 외란 주입 + 라벨 (2일)
- `anomaly.py` (A1/A2/A3/N1)
- `simulator.py` 통합
- 무작위 instance 분포 생성
- 노트북 03
- ✅ 검증: 각 시나리오 시계열 + 라벨 시각

### M3. TSB-AD-M Export (1일)
- `exporter.py`
- 1s/30s/1m/30m/1h 다운샘플
- `meta.json`
- ✅ 검증: 기존 TSB-AD-M 평가 스크립트로 dummy run

### M4. Streamlit 대시보드 (3일)
- `dashboard/app.py`
- 토폴로지 + 시계열 + 외란 주입 + 다운로드
- ✅ 검증: 사용자 시나리오 4종 (A2/밸브/N1/다운로드)

### M5. 통합 검증 + 문서 (1일)
- 전체 dataset 통계 노트북
- README
- baseline AD 알고리즘 1종 dry-run
- ✅ 검증: end-to-end

**총 ~10일.**

---

## 12. 미확정 → Default 결정

| 항목 | Default |
|---|---|
| 청년몰 점포 수 | 25 |
| 업종 mix 비율 | §2.4 표 |
| 시작 계절 | 여름 (7월, 장마) |
| 시나리오 비율 | A1:A2:A3:N1 = 4:3:2:1 |
| UI 언어 | 한국어 |
| 출력 단위 | 점포별 + 동 통합 모두 |

→ 실제 작업 시 yaml만 수정해서 변경 가능.

---

## 13. 리스크 & 한계 (간단)

- **합성 데이터의 비현실성**: 실측 부재로 검증 불가. "synthetic" 명시.
- **물리 부정합**: 밸브 인접 분배 등 휴리스틱. 직관적 데모용일 뿐.
- **Streamlit 성능**: 380 채널 동시 plot 무리 → 선택적 상세로 회피.
- **컨소시엄 실측 도착 시**: yaml 파라미터 calibration으로 부분 보정 가능, 구조 변경 불필요.

---

## 부록. 의존성

```
numpy, pandas, scipy
networkx
plotly
streamlit
pyyaml
pytest
```

EPANET·hydraulic solver 등 외부 도구 없음.

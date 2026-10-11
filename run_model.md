# 장애물 경고 코드 명세 (짧은 버전)

> 코드: `test_model3.py`  · 결과 형식: `RUN_FORMAT.md`
> 결과를 맞추는 공식 방법은 **모두 같은 커밋의 `test_model3.py`를 같은 인자로 돌리는 것**이다. 이 문서만 보고 코드를 다시 짜도 같은 숫자가 나오도록, 계산 규칙(4장)은 반올림까지 빠짐없이 적었다.

---

## 1. 무엇을 하나

영상 프레임마다 앞 경로에서 가장 가까운 장애물까지의 거리를 잰다. 3 m보다 가까우면 **피할 방향**으로 진동 명령(`left` / `right` / `neck`)을 기록한다.

```
프레임 → 폭 640px로 축소 → YOLO 박스 + depth 지도(m)
      → ① 경로 사각형 안 depth 거리   ② 경로에 걸친 YOLO 박스 거리
      → 둘 중 가까운 값 < 3 m 이면 경고
      → 장애물 위치로 방향(왼쪽/정면/오른쪽) → 반대쪽 모터
      → frames.csv에 한 줄
```

- YOLO는 **박스 위치만** 쓴다. 클래스(사람·의자 등)는 화면 표시와 기록에만 쓰고 판정에는 안 쓴다.
- YOLO가 못 잡는 물체(A4 등)도 경로 안에 있으면 depth로 잡는다.

---

## 2. 실행

```bash
python test_model3.py --video 1011_front_01.mp4 --runner 연수
```

| 인자 | 뜻 |
|---|---|
| `--video` | 영상 경로. `0`이면 맥북 카메라(결과 저장 안 함). 없으면 코드의 `SOURCE` |
| `--runner` | 돌린 사람 이름. 영상이면 필수 |
| `--trial` | 시행ID. 없으면 영상 파일명에서 확장자를 뺀 것 |
| `--expected_dir`, `--gt_2m_sec`, `--note` | 기록만 함. 계산에는 안 씀 |
| `--depth_every_n` | N프레임마다 depth 1번 (기본 1). **결과가 달라지니 팀 전체가 같은 값을 쓴다** |

- 시행ID 형식은 `<MMDD>_<front|left|right|none>_<2자리>`이고, 영상 파일명과 같게 한다. 예: `1011_left_01.mp4`
- 키 조작 (영상 창을 클릭해 둔 상태): `space` 일시정지, `q` 종료, 마우스 클릭 → 그 점의 거리

---

## 3. 설정값 (코드 맨 위)

```python
FRAME_WIDTH = 640                     # 프레임 축소 폭
MODEL_PATH = "yolo11n.pt"
CONF_THRES = 0.4                      # YOLO 확률 하한
IMG_SIZE = 416                        # YOLO imgsz
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEPTH_SCALE = 1.0                     # depth 보정 배율
PATH_WIDTH_M = 0.8                    # 경로 폭 (기록용)
HFOV_DEG = 70.0                       # 카메라 가로 화각 (기록용)
WARN_M = 3                            # 이보다 가까우면 경고
PATH_Y = (0.25, 0.75)                 # (기록용)
PATH_ROI = (0.40, 0.60, 0.25, 0.75)   # 경로 사각형 (x 시작, x 끝, y 시작, y 끝) 화면 비율
NEAR_PERCENTILE = 5                   # 경로 depth 하위 %
NEAR_BAND_M = 0.3                     # 경로 거리 + 이 값 안의 픽셀들 중심 = 장애물 위치
BOX_SHRINK = 0.2                      # 박스를 각 변에서 줄이는 비율
BOX_PERCENTILE = 30                   # 박스 안 depth 하위 %
BOX_OVERLAP_MIN = 0.3                 # 박스 폭 중 경로 안 비율 하한
ZONE_MODE = "path"                    # 방향 판정 칸: 경로 사각형 3등분
ZONE_DRAW = "screen"                  # 화면에 그리는 칸: 화면 전체 3등분 (보기용)
ZONE_PATH_SPLIT = (1 / 3, 2 / 3)
HOLD_DIRECTION = True                 # 경고 동안 첫 방향 유지
MOTOR_MAP = {"왼쪽": "right", "오른쪽": "left", "정면": "neck"}
VIB_COUNT = {1: 1, 2: 2}
MOTOR_TEXT_SCALE = 1.6                # 화면 모터 글자 크기
RUNS_DIR = "runs"
SCHEMA_VERSION = 1
CHOICES = {"2-1": "A", "2-2": "B", "2-3": "경고하지 않음", "2-5": "path", "2-7": "B",
           "2-9": "B (PATH_ROI 다시 계산)", "2-10": "A (정면 → neck)", "2-11": "B", "2-12": "B"}
```

`CHOICES`는 `run_meta.json`에 그대로 기록된다. 번호의 뜻은 부록에 있다. 이 값이 다르면 `score.py`가 "설정이 다른 run"으로 경고한다.

---

## 4. 계산 규칙 (바꾸면 결과가 달라지는 부분)

`W`, `H`는 축소한 프레임의 폭·높이다. 계산 좌표는 모두 축소 프레임 픽셀 기준이다. 백분위는 `np.percentile` 기본값(선형 보간)을 쓴다.

### 4-1. 프레임 준비
- 원본 폭 `w`가 640보다 크면 `cv2.resize(frame, (640, int(h * 640 / w)))`(기본 보간). 아니면 그대로 쓴다.
- `sx = w / W`: 저장할 때 원본 픽셀로 되돌리는 배율
- YOLO: `yolo.predict(frame, imgsz=416, conf=0.4, device=DEVICE)`. 박스 좌표는 `int()`로 소수점을 버린다.
- Depth: 축소 프레임을 RGB로 바꿔 pipeline에 넣고, `predicted_depth`를 `cv2.resize(d, (W, H))`(기본 보간)한 뒤 `DEPTH_SCALE`을 곱한다.
  - `frame_idx % depth_every_n == 0`인 프레임에서만 새로 계산하고, 나머지 프레임은 직전 depth를 쓴다.
  - 첫 depth가 나오기 전에는 거리가 없다. 박스 거리는 `None`이고 경고도 없다.
- `DEVICE`는 mps가 있으면 mps, 없으면 cpu다. 장치가 다르면 소수점 끝자리가 다를 수 있다.

### 4-2. YOLO 박스: 거리와 "경로 안" 판정
- **거리**: 박스를 각 변에서 폭·높이의 20%씩 줄인 영역의 depth 하위 30% 백분위
  - `dx = (x2-x1)·0.2`, `dy = (y2-y1)·0.2`
  - 줄인 영역은 `x: round(x1+dx) ~ round(x2-dx)`, `y: round(y1+dy) ~ round(y2-dy)`. 끝은 포함하지 않고(파이썬 슬라이스), 화면 밖으로 나간 부분은 잘라낸다.
  - 줄인 영역의 폭이나 높이가 1픽셀보다 작으면 원래 박스 영역을 쓴다.
- **경로 안**: 아래 두 조건을 모두 만족해야 한다. 박스 폭이 0이면 경로 밖이다.
  - 박스 폭 중 `0.40·W ~ 0.60·W` 안에 들어온 비율이 0.3 이상
  - 박스 세로 범위가 `0.25·H ~ 0.75·H`와 겹침 (`min(y2, 0.75H) > max(y1, 0.25H)`)
  - 이 경계값은 소수 그대로 쓴다 (`int` 하지 않음).

### 4-3. 경로 depth: 거리와 위치
- 경로 사각형 픽셀: `x int(0.40W) ~ int(0.60W)`, `y int(0.25H) ~ int(0.75H)` (끝은 포함하지 않음)
- depth 전체에 `cv2.medianBlur(depth.astype(np.float32), 5)`를 적용한 뒤 사각형을 잘라낸다.
- 거리 `near` = 사각형 안 하위 5% 백분위
- 위치 = 사각형 안에서 값이 `near + 0.3` 이하인 픽셀들의 x 평균과 y 평균을 각각 `round`한 값 + 사각형 시작점

### 4-4. 합치기와 경고
- 후보 = [경로 depth 거리] + [경로 안 YOLO 박스 거리들 (YOLO 출력 순서)]. 경로 밖 박스는 후보가 아니다.
- 장애물 거리 = 후보 중 최솟값. 같은 값이 여럿이면 앞의 것을 고른다 (DEPTH가 맨 앞).
- 경고 = 장애물 거리 **< 3**. 정확히 3이면 경고가 아니다.
- 경고 대상: DEPTH면 4-3의 위치, YOLO면 그 박스

### 4-5. 방향
- 판정에 쓰는 x: DEPTH면 4-3 위치의 x, YOLO면 `(x1 + x2) / 2`
- 경계: `x0 = 0.40·W`, `x1 = 0.60·W`일 때
  - `L = int(x0 + (x1 - x0) * (1/3))`
  - `R = int(x0 + (x1 - x0) * (2/3))`
  - W = 640이면 L = 298, R = 341
- `x < L` → 왼쪽, `x > R` → 오른쪽, 그 외(경계값 포함) → 정면
- **방향 유지**: 경고가 이어지는 동안에는 그 경고 구간 첫 프레임의 방향을 계속 쓴다. 경고가 아닌 프레임이 한 번이라도 나오면 초기화하고, 다음 경고에서 새로 정한다.
- 화면에 그려지는 칸 선(W/3, 2W/3)은 보기용이다. 판정 경계와 다르다.

### 4-6. 진동 명령
진동은 **피할 방향**을 알려준다.

| 장애물 위치 `direction` | `motor` | 뜻 |
|---|---|---|
| 왼쪽 | `right` | 오른쪽으로 피하라 |
| 오른쪽 | `left` | 왼쪽으로 피하라 |
| 정면 | `neck` | 정면 장애물 |

- `vib_count` = 2 (경고 = 레벨 2). 경고가 아니면 `motor`와 `vib_count` 모두 빈칸이다.
- 레벨 1(주의)·3(위험)은 쓰지 않는다.

---

## 5. 저장 (영상 입력일 때만)

폴더는 `runs/<runner>/<시행ID>_<MMDD>_<HHMMSS>/`이다 (뒤의 날짜·시각은 돌린 시각). 폴더나 파일이 이미 있으면 오류가 나고 덮어쓰지 않는다. 표시 영상은 저장하지 않는다.

### frames.csv
프레임마다 한 줄씩 쓴다. 인코딩은 utf-8-sig이고, 일시정지 중에는 줄을 쓰지 않는다. 칸 순서:

`trial, frame_idx, t_ms, processed, n_dets, labels, dets_json, target_label, target_conf, target_cx, dist, ttc, level, level_name, direction, beep, vibrate_ms, tts, yolo_ms, depth_ms, target_box, motor, vib_count`

| 칸 | 값 |
|---|---|
| `trial` | 시행ID |
| `frame_idx` | 0부터 |
| `t_ms` | `cap.read()` 직후 읽은 `CAP_PROP_POS_MSEC`를 `round(max(값, 0))`. 두 번째 프레임부터 값이 0 이하로 나오면 그 뒤로는 `round(frame_idx × 1000 / fps)`를 쓰고 터미널에 한 번 경고 |
| `processed` | 1 |
| `n_dets`, `labels` | 박스 수, 클래스 이름을 `;`로 이은 것 |
| `dets_json` | `json.dumps([{"label", "conf", "box", "dist"}, ...], ensure_ascii=False)`. conf·dist는 소수 3자리(dist 없으면 null), box는 원본 픽셀 정수 `[round(x1·sx), …]` |
| `dist` | 장애물 거리, 소수 3자리 (경고가 아니어도 기록. 없으면 빈칸) |
| `level`, `level_name` | 경고면 `2` / `경고`, 아니면 `0` / `없음` |
| `target_label` | 경고일 때: YOLO 클래스 이름 또는 `DEPTH` |
| `target_conf` | 경고 + YOLO일 때: 소수 3자리 |
| `target_cx` | 경고일 때: `round(판정 x × sx)` |
| `target_box` | 경고 + YOLO일 때: `x1 y1 x2 y2` (각각 `round(v × sx)`, 공백으로 구분) |
| `direction` | 경고일 때: `왼쪽` / `정면` / `오른쪽` |
| `motor`, `vib_count` | 경고일 때: 4-6 |
| `yolo_ms`, `depth_ms` | 처리 시간, 소수 1자리 (depth를 안 돌린 프레임은 빈칸) |
| `ttc`, `beep`, `vibrate_ms`, `tts` | 항상 빈칸 |

### summary.json
종료할 때 한 번 쓴다. `q`나 Ctrl+C로 중간에 끝내도 쓴다.

| 칸 | 값 |
|---|---|
| `trial`, `video` | 시행ID, 영상 파일명 |
| `frames`, `processed` | frames.csv 줄 수, 처리한 줄 수 |
| `frame_size`, `video_fps`, `duration_sec` | 원본 크기, fps, 길이 (= 프레임 수 / fps, 소수 3자리) |
| `frames_with_det`, `frames_with_target` | 박스가 있던 프레임 수, 경고 프레임 수 |
| `label_counts` | 클래스별 등장 횟수 |
| `first_level2` | 첫 경고 줄의 `{t_sec, dist, direction}` 또는 null |
| `first_vibration` | `motor`가 처음 채워진 줄의 `{t_sec, dist, direction, target_label}` 또는 null |
| `first_level1`, `first_level3` | null |
| `max_level` | 가장 높은 레벨 |
| `yolo_ms_median`, `depth_ms_median` | 처리 시간 중앙값, 소수 1자리 |
| `wall_sec` | 실제 걸린 시간 |
| `expected_dir`, `gt_2m_sec`, `note` | 인자 그대로 |

### run_meta.json
시작할 때 한 번 쓴다.

| 칸 | 값 |
|---|---|
| `run_at` | 돌린 시각 |
| `args` | 인자 전부 (`trial`은 실제로 쓴 값) |
| `pipeline` | device, YOLO 가중치, depth 모델, conf, imgsz, torch·ultralytics·transformers·opencv 버전 |
| `alarm_constants` | 3장 값 전부 (`CHOICES` 포함) |
| `runner` | 이름 |
| `git_commit` | `git rev-parse HEAD`. 실패하면 `"unknown"` |
| `video_file` | 파일명만 (경로 제외) |
| `schema_version` | 1 |

---

## 6. 채점 (`score.py`, 만들 예정)

```bash
python score.py --runs runs/<이름> --trials trials.csv
```

run 폴더는 읽기만 하고, 결과는 `scores/<MMDD_HHMMSS>/`에 쓴다 (`score.json`, `trial_scores.csv`).

### trials.csv (사람이 채우는 정답표, 영상 1개당 1줄)

| 칸 | 값 |
|---|---|
| `trial` | 시행ID |
| `condition` | `front` / `left` / `right` / `none` |
| `place` | `복도` / `스룸` |
| `expected_dir` | 장애물 위치. 비우면 front→정면, left→오른쪽, right→왼쪽 |
| `gt_far_sec` | 바깥 경계 통과 시각(초). 복도는 4 m, 스룸은 3 m 지점 |
| `gt_2m_sec` | 2 m 지점 통과 시각(초) |
| `end_sec` | none 시행만: 장애물이 있던 위치를 지난 시각(초). 비우면 영상 끝까지 |
| `stopped` | 실착용 시행만: 장애물 전에 멈췄으면 1, 닿았으면 0 |
| `note` | 메모 |

조건 이름이 곧 피할 방향이다. `left`는 장애물이 오른쪽에 걸쳐 있어 왼쪽으로 피하는 경우이고, 정답 모터는 `left`다. `right`의 정답 모터는 `right`, `front`의 정답 모터는 `neck`이다. 바닥에 2 m와 4 m(스룸은 3 m) 지점을 테이프로 표시해 두면 영상에서 통과 시각을 잴 수 있다.

### 시행별 판정
"진동 줄" = `level ≥ 2`이고 `motor`가 빈칸이 아닌 줄

- **장애물 시행**: 영상 시각이 `gt_far_sec` 이상, `gt_2m_sec` 미만인 구간에서
  - 구간 안 첫 진동의 모터가 정답과 같음 → **성공**
  - 첫 진동의 모터가 다름 → **방향틀림**
  - 구간 안엔 없고 2 m 이후에만 있음 → **늦음**
  - 아예 없음 → **없음**
- **none 시행**: `end_sec` 전에 진동 줄이 하나라도 있으면 **오경고**, 없으면 **정상**

### 지표

| 지표 | 식 | 목표 |
|---|---|---|
| 회피 안내 성공률 | 성공 ÷ 장애물 시행 (전체와 방향별 둘 다) | ≥ 85% |
| 오경고율 | 오경고 ÷ none 시행 | ≤ 10% |
| 정지 성공률 | `stopped = 1` ÷ (진동이 있었고 `stopped`를 적은 시행) | 보고만 |

- 두 목표를 점추정치로 모두 만족하면 "달성"이다. Wilson 95% 신뢰구간을 같이 적는다.
  - 확인용 값: 26/30 → 86.7% [70.3, 94.7], 3/30 → 10.0% [3.5, 25.6]
- 시행이 30회보다 적으면 "시행 수 부족"이라고 표시한다.

---

## 7. 진동 인지 시험 (11주, `score.py --haptic`, 보조 지표)

모터 위치(`left` / `right` / `neck`)와 횟수(1·2회)를 무작위 순서로 울리고 착용자가 맞힌다. 결과는 `haptic_test.csv`에 제시 1번당 1줄로 적는다.

칸: `participant, trial_no, shown_motor, shown_count, answer_motor, answer_count, rt_ms`

결과로 나오는 것:
- 위치 구분율, 횟수 구분율 (Wilson 구간 포함)
- 혼동표: 위치 3×3, 횟수 2×2
- 반응 시간 중앙값

---

## 8. 파일럿 4회

조건마다 1회씩 촬영해서 촬영 → 실행 → 채점이 끝까지 이어지는지 확인한다. 4회로는 목표 달성 여부를 말할 수 없다.

1. `<MMDD>_front_01`, `_left_01`, `_right_01`, `_none_01`을 촬영한다. 같은 5 m 경로, 같은 촬영 모드로 찍는다.
2. 영상마다 통과 시각을 재서 `trials.csv` 4줄을 쓴다.
3. 코드를 커밋한 뒤 4번 돌린다: `python test_model3.py --video <영상> --runner <이름>`
4. `python score.py --runs runs/<이름> --trials trials.csv`
5. 결과와 해석을 `experiments.md`에 적는다.

볼 것:
- left/right 시행의 첫 경고가 DEPTH인지 YOLO인지
- 옆 벽이 경로 사각형에 들어와 처음부터 경고가 뜨지 않는지 (벽에 붙어 걷지 않기)
- none 시행 오경고의 원인
- 추정 거리와 실제 거리 차이 (`DEPTH_SCALE` 보정이 필요한지)

---

## 9. 돌린 뒤 확인

- [ ] run 폴더에 `run_meta.json`, `frames.csv`, `summary.json` 세 파일이 있다
- [ ] `frames.csv` 줄 수 = `summary.json`의 `frames`
- [ ] `git_commit`이 `unknown`이 아니다 (커밋하고 돌렸다)
- [ ] 폴더명의 시행ID = 영상 파일명
- [ ] 경고 줄에만 `motor`가 있고, 한 경고 구간 안에서는 `motor`가 바뀌지 않는다
- [ ] 정면 장애물 시행은 경고 내내 `neck`
- [ ] 엑셀로 열었을 때 한글이 안 깨진다

---

## 10. 하지 말 것

- 각자 코드를 고쳐서 돌리기. 바꿀 게 있으면 팀에서 정하고 커밋한 뒤 모두 다시 돌린다.
- 시행마다 코드의 `SOURCE` 고치기. `--video`를 쓴다.
- 클래스별로 다르게 처리하기 (사람이면 ~, 의자면 ~)
- RUN_FORMAT 칸 이름 바꾸기·칸 더하기. 바꾸려면 팀 합의 후 `schema_version`을 올린다.
- 성공률 같은 채점 결과를 run 폴더에 넣거나 run 폴더 파일을 손으로 고치기
- 새 패키지 추가하기. `score.py`는 파이썬 기본 모듈만 쓴다.

---

## 11. 아직 정할 것

- `WARN_M` 3 m, 정면 장애물이면 `neck`: 팀 확정
- 목표의 "30회"가 방향별 30회인지 전체 30회인지
- `--sim_fps`의 뜻과 기본값
- `trials.csv`의 `place`, `gt_far_sec`, `end_sec`를 RUN_FORMAT §4에 추가할지
- 촬영 모드 확정, 카메라 화각(`HFOV_DEG`) 실측

---

## 부록: 왜 이렇게 정했나 (코드 주석의 번호)

| 번호 | 선택 | 이유 |
|---|---|---|
| 2-1 A | 경로 = 고정 사각형 (x 0.40~0.60, y 0.25~0.75) | 3 m 앞에서 폭 0.8 m가 되도록 화각 70°로 계산함. y는 천장·바닥 제외 |
| 2-2 B | 박스 거리 = 줄인 영역 하위 30% | 박스 중앙 한 점은 배경에 걸리면 틀림 |
| 2-3 | 경로 밖 박스는 경고 안 함 | 거리만 표시 |
| 2-5 B | 방향 = 경로 사각형 3등분 | 화면 전체 3등분이면 경로가 통째로 정면 칸이라 DEPTH 경고가 항상 `neck`이 됨 |
| 2-7 B | 박스 폭 30% 이상이 경로 안이면 경로 안 | 가장자리에 살짝 닿은 박스는 무시 |
| 2-9 B | 경고 거리 3 m | 2 m면 경고가 2 m 안에서만 시작돼 성공률이 0%가 됨 |
| 2-10 A | 정면 장애물이면 `neck` | 피할 방향 대신 멈춤 신호 |
| 2-11 B | DEPTH 위치 = 가까운 픽셀들의 중심 | 한 점은 프레임마다 흔들려 좌우가 섞임 |
| 2-12 B | 경고 동안 첫 방향 유지 | 착용자에게 좌우가 번갈아 울리지 않게. 단, 첫 방향이 틀리면 그 경고 내내 틀림 |

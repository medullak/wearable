"""
4.10 뽑은 프레임에 yolo11n.pt를 돌려 기존 80개 클래스 박스를 YOLO txt로 저장 (자동 라벨)

실행:  python tools/prelabel.py datasets/a4/frames/val --out datasets/a4/prelabel --name val
결과:  <out>/<name>/labels/<이미지이름>.txt  (박스가 하나도 없는 이미지는 txt가 안 생김)
다음:  웹 도구(CVAT, Roboflow 등)에 이미지와 이 txt를 불러와 A4 박스(클래스 80)를 추가하고
       틀린 자동 박스를 고친 뒤 YOLO txt로 내보낸다 → datasets/a4/images/<train|val>, labels/<train|val>
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("images", help="jpg가 들어 있는 폴더")
parser.add_argument("--out", default=str(HERE / "datasets" / "a4" / "prelabel"), help="결과 상위 폴더")
parser.add_argument("--name", default="labels_auto", help="결과 하위 폴더 이름")
args = parser.parse_args()

model = YOLO(str(HERE / "yolo11n.pt"))
n = 0
for _ in model.predict(source=args.images, conf=0.4, save_txt=True, save=False,
                       project=args.out, name=args.name, exist_ok=True, stream=True, verbose=False):
    n += 1
print(f"{n}장 자동 라벨 완료 → {Path(args.out) / args.name / 'labels'}")

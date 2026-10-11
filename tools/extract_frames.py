"""
4.10 영상에서 N프레임마다 jpg 저장 (A4 학습 데이터용)

실행:  python tools/extract_frames.py paper1.mp4 paper_no.mp4 --every 10 --out datasets/a4/frames/val
       train/val은 영상 단위로 나눈다: 영상마다 어느 쪽인지 정해서 --out을 따로 준다 (paper1.mp4는 val)
저장 이름: <영상이름>_<프레임번호 5자리>.jpg
"""

import argparse
from pathlib import Path

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("videos", nargs="+", help="영상 파일들")
parser.add_argument("--every", type=int, default=10, help="N프레임마다 1장 저장")
parser.add_argument("--out", required=True, help="jpg 저장 폴더")
args = parser.parse_args()

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

for video in args.videos:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[건너뜀] 열 수 없어요: {video}")
        continue
    stem = Path(video).stem
    idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            cv2.imwrite(str(out_dir / f"{stem}_{idx:05d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"{video}: {idx}프레임 중 {saved}장 저장 → {out_dir}")

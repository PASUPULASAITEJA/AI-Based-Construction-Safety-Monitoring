import os
from cv.pipeline import SafetyPipeline

pipeline = SafetyPipeline(model_path='d:/Safe/model/best.pt', conf=0.035)

tests = [
    ("image886.jpeg (Worker 1: Helmet + Lime Vest, Worker 2: Helmet + Blue Overalls)", "d:/Safe/uploads/upload_20260825_095059_image886.jpeg"),
    ("image1005.jpeg (Foreground worker carrying mud, background worker)", "d:/Safe/uploads/upload_20260824_225326_image1005.jpeg"),
    ("image1003.jpg (Standard Construction Worker with PPE)", "d:/Safe/construction/images/test/image1003.jpg"),
    ("image1007.jpg (Construction Personnel with Hardhat & Vest)", "d:/Safe/construction/images/test/image1007.jpg"),
    ("image1009.jpg (Construction Personnel with Full Gear)", "d:/Safe/construction/images/test/image1009.jpg"),
]

for title, path in tests:
    if not os.path.exists(path):
        print(f"\n[SKIP] File not found: {path}")
        continue
    print(f"\n==================================================")
    print(f"EVALUATING: {title}")
    print(f"==================================================")
    _, stats = pipeline.process_single_image(path)
    print(f"Workers Detected: {stats['worker_count']} | Total Violations: {stats['violation_count']}")
    for w in stats['workers']:
        viols = [v['type'] for v in w['violations']]
        print(f"  -> {w['label']}: Helmet={'YES' if w['helmet'] else 'NO'} (Conf: {w['helmet_conf']}) | Vest={'YES' if w['vest'] else 'NO'} (Conf: {w['vest_conf']}) | Status={w['status']} | Violations={viols}")

from cv.pipeline import SafetyPipeline

pipeline = SafetyPipeline(model_path='d:/Safe/model/best.pt', conf=0.035)

tests = [
    ("image939.jpeg (2 Workers, Flannel & Black T-Shirt, Both Wearing Helmets, No Vests)", "d:/Safe/uploads/upload_20260824_223843_image939.jpeg"),
    ("image1005.jpeg (Foreground worker carrying mud, background worker)", "d:/Safe/uploads/upload_20260824_223253_image1005.jpeg"),
    ("image321.jpg (4 Workers Standing in a Row with Helmets & Vests)", "d:/Safe/uploads/upload_20260824_222248_image321.jpg"),
    ("image522.jpeg (Radio Tower Climber with White Helmet & Vest)", "d:/Safe/uploads/upload_20260824_220809_image522.jpeg"),
    ("image546.jpg (Lime Vest, No Helmet on Head)", "d:/Safe/uploads/upload_20260824_204649_image546.jpg"),
    ("image988.jpeg (2 Workers, Beige & Black T-shirts, No PPE)", "d:/Safe/uploads/upload_20260824_205553_image988.jpeg"),
    ("image21.jpeg (Workers Digging with Helmets & Vests)", "d:/Safe/uploads/upload_20260824_211358_image21.jpeg"),
    ("image1003.jpg (Standard Construction Worker with PPE)", "d:/Safe/construction/images/test/image1003.jpg"),
]

for title, path in tests:
    print(f"\n==================================================")
    print(f"EVALUATING: {title}")
    print(f"==================================================")
    _, stats = pipeline.process_single_image(path)
    print(f"Workers Detected: {stats['worker_count']} | Total Violations: {stats['violation_count']}")
    for w in stats['workers']:
        viols = [v['type'] for v in w['violations']]
        print(f"  -> {w['label']}: Helmet={'YES' if w['helmet'] else 'NO'} (Conf: {w['helmet_conf']}) | Vest={'YES' if w['vest'] else 'NO'} (Conf: {w['vest_conf']}) | Status={w['status']} | Violations={viols}")

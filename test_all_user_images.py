import os
if __name__ == "__main__":
    from cv.pipeline import SafetyPipeline
    pipeline = SafetyPipeline(conf=0.035)

    tests = [
        ("image1003.jpg (Standard Construction Worker with PPE)", "construction/images/test/image1003.jpg"),
    ]

    for title, path in tests:
        if not os.path.exists(path):
            continue
        print(f"\n==================================================")
        print(f"EVALUATING: {title}")
        print(f"==================================================")
        _, stats = pipeline.process_single_image(path)
        print(f"Workers Detected: {stats['worker_count']} | Total Violations: {stats['violation_count']}")


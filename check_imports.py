"""
Connectivity check — verifies all project modules can be imported cleanly.
Run from project root: python check_imports.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODULES = [
    "config",
    "syllabus",
    "main",
    "experiments",
    "etl.extract",
    "etl.transform",
    "etl.load",
    "rag.retriever",
    "rag.generator",
    "modes.quiz",
    "modes.exam",
    "modes.learning",
    "modes.syllabus",
    "analytics.clustering",
    "analytics.evaluator",
    "analytics.dashboard",
    "launcher",
]

ok = []
errors = []

print("\n" + "="*55)
print("  MODULE CONNECTIVITY CHECK")
print("="*55)

for mod in MODULES:
    try:
        __import__(mod)
        print(f"  ✅  {mod}")
        ok.append(mod)
    except Exception as e:
        short_err = str(e)[:80]
        print(f"  ❌  {mod}")
        print(f"       └─ {short_err}")
        errors.append((mod, str(e)))

print("="*55)
print(f"  RESULT: {len(ok)}/{len(MODULES)} OK  |  {len(errors)} FAILED")
print("="*55 + "\n")

if errors:
    print("FAILURES DETAIL:")
    for mod, err in errors:
        print(f"\n  [{mod}]\n  {err}")
    sys.exit(1)
else:
    print("All modules connected OK!\n")
    sys.exit(0)

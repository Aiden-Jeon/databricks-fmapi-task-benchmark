"""tasks.yaml의 datasets 참조 키가 registry.yaml에 모두 존재하는지 교차 검증."""

import sys

import yaml


def main() -> int:
    tasks = yaml.safe_load(open("config/tasks.yaml", encoding="utf-8"))
    reg = {k for k in yaml.safe_load(open("datasets/registry.yaml", encoding="utf-8")) if not str(k).startswith("#")}
    used: set[str] = set()
    missing = []
    for t in tasks.get("image_tasks", []) + tasks.get("text_tasks", []):
        for lang, key in (t.get("datasets") or {}).items():
            used.add(key)
            ok = key in reg
            if not ok:
                missing.append((t["id"], lang, key))
            print(f"  {t['id']:<6} {lang}: {key:<20} [{'OK' if ok else 'MISSING'}]")
    print()
    print("  registry에 있지만 안 쓰는 키:", sorted(reg - used))
    if missing:
        print("\n  !! 누락:", missing)
        return 1
    print("\n  전부 정합 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

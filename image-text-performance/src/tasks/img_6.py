"""IMG-6 표 이미지 → 표 구조 추출 (vision, Cell-F1 메트릭).

TXT-3(텍스트 셀 → HTML)와 달리, 이 태스크는 **표 이미지**를 모델에 주고 이미지 속 표를
HTML로 복원하게 한다. PDF/스캔 표처럼 텍스트 메타정보 없이 픽셀만 있는 표에서 구조·내용을
읽어내는 vision 역량을 측정한다(PDF 파서가 아니라 모델의 표-이미지 이해력).

- 입력: 표 이미지(PIL) — 다양한 형태(병합셀·헤더·다열 등)일수록 좋음.
- 출력: 모델이 생성한 HTML 표.
- 채점: 정답 GT HTML과 예측 HTML을 셀 단위로 파싱해 Cell-F1(TXT-3의 로직 재사용).
- vision=True → glm 등 vision 미지원 모델은 러너가 자동 N/A 스킵.

데이터셋: `datasets/registry.yaml`의 `table_image` 키(이미지 + GT HTML을 함께 제공).
한/영 무관(표 구조는 시각 기반). config의 datasets 맵으로 참조해 언어 확장 가능.
"""

from __future__ import annotations

import re
from typing import Any

from src.adapters.fmapi import build_image_message
from src.adapters.images import bytes_to_data_url, pil_to_data_url
from src.datasets_loader import load_hf_split, load_registry, resolve_dataset_entry
from src.scoring.accumulators import MeanAccumulator
# TXT-3의 HTML 파싱·Cell-F1 로직을 그대로 재사용(중복 구현 방지, 채점 일관성).
from src.tasks.txt_3 import cell_f1_score, parse_html_table
from src.tasks.base import Task, Sample, register


def _to_data_url(image: Any) -> str | None:
    """HF 이미지 컬럼을 data URL로 정규화.

    - PIL.Image → pil_to_data_url
    - {"bytes": ..., ...} dict(HF Image 미디코드 형태) → bytes_to_data_url
    - str(URL/data URL) → 그대로(또는 urlopen은 하지 않고 그대로 넘김; 대부분 data/http)
    지원 못하면 None.
    """
    if image is None:
        return None
    # PIL.Image?
    if hasattr(image, "size") and hasattr(image, "mode"):
        try:
            return pil_to_data_url(image, max_side=1024)  # 표는 세부가 중요 → 큰 변 넉넉히
        except Exception:
            return None
    # HF Image dict {"bytes": b"...", "path": ...}
    if isinstance(image, dict) and image.get("bytes"):
        try:
            return bytes_to_data_url(image["bytes"])
        except Exception:
            return None
    # 이미 문자열 URL/data URL
    if isinstance(image, str) and (image.startswith("data:") or image.startswith("http")):
        return image
    return None


@register
class Img6Task(Task):
    """표 이미지 → HTML 표 추출 (IMG-6). Cell-F1 채점."""

    task_id: str = "IMG-6"
    kind: str = "extraction"
    is_vision: bool = True

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """표 이미지 + 정답 HTML을 로드.

        각 샘플:
        - inputs: {"image": PIL/bytes/URL 원본} (프롬프트 빌드 시 data URL로 변환)
        - reference: 정답 HTML 표 마크업(문자열)

        데이터셋은 config의 datasets 맵(예: {en: table_image})으로 참조.
        이미지·HTML 컬럼명은 registry 항목의 image_column/html_column으로 지정(없으면 자동 감지).
        실제 로드 실패 시 예외(합성 폴백 없음 — 표준 데이터셋 원칙).
        """
        registry = load_registry()
        config = self.config
        if "datasets" not in config:
            raise ValueError("config에 datasets 맵이 없음")

        samples: list[Sample] = []
        sample_id = 0
        # 이미지 표는 GT 파싱 실패·이미지 없음으로 버려지는 샘플이 있어 넉넉히 로드 후 필터.
        load_n = max(n * 2, 40)

        for lang, dataset_key in config["datasets"].items():
            entry = resolve_dataset_entry(registry, dataset_key)
            hf_id = entry["hf_id"]
            split = entry.get("split", "validation")
            config_name = entry.get("config")
            img_col = entry.get("image_column")
            html_col = entry.get("html_column")

            hf_ds = load_hf_split(hf_id, split, load_n, seed, config_name)
            if not hf_ds:
                continue
            if img_col is None:
                img_col = self._detect_image_column(hf_ds[0])
            if html_col is None:
                html_col = self._detect_html_column(hf_ds[0])

            for idx, row in enumerate(hf_ds):
                image = row.get(img_col)
                gold_html = row.get(html_col)
                if image is None or not gold_html or not isinstance(gold_html, str):
                    continue
                gold_html = self._normalize_gold_html(gold_html)
                # 정답이 실제 파싱 가능한 표인지 확인(빈 표·파싱불가 샘플 제외).
                if not parse_html_table(gold_html):
                    continue

                samples.append(
                    Sample(
                        sample_id=sample_id,
                        inputs={"image": image},
                        reference=gold_html,
                        lang=lang,
                        meta={"dataset": dataset_key, "source_idx": idx},
                    )
                )
                sample_id += 1
                if sample_id >= n:
                    break
            if sample_id >= n:
                break

        return samples

    def _detect_image_column(self, row: dict[str, Any]) -> str:
        """이미지 컬럼명 자동 감지(PIL/bytes-dict). 실패 시 예외."""
        for col in ("image", "img", "table_image", "png", "figure"):
            if col in row:
                return col
        # 값 기반 탐지: PIL.Image 또는 {"bytes":...} dict
        for k, v in row.items():
            if hasattr(v, "size") and hasattr(v, "mode"):
                return k
            if isinstance(v, dict) and v.get("bytes"):
                return k
        raise ValueError(f"이미지 컬럼을 찾을 수 없음. 컬럼: {list(row.keys())}")

    def _detect_html_column(self, row: dict[str, Any]) -> str:
        """정답 HTML 컬럼명 자동 감지. 실패 시 예외."""
        for col in ("html_table", "html", "structure_html", "table", "gt"):
            if col in row and isinstance(row.get(col), str) and "<t" in row[col].lower():
                return col
        # 값 기반: <table>/<tr>/<td>가 들어있는 문자열 컬럼
        for k, v in row.items():
            if isinstance(v, str) and re.search(r"<t(able|r|d)\b", v, re.IGNORECASE):
                return k
        raise ValueError(f"정답 HTML 컬럼을 찾을 수 없음. 컬럼: {list(row.keys())}")

    @staticmethod
    def _normalize_gold_html(html: str) -> str:
        """GT가 <table> 래퍼 없이 <tr>...만 오는 경우(PubTabNet류) 래핑해 파싱 일관성 확보."""
        if "<table" not in html.lower():
            return f"<table>{html}</table>"
        return html

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """표 이미지를 보고 HTML 표로 복원하는 멀티모달 프롬프트."""
        data_url = _to_data_url(sample.inputs["image"])
        if data_url is None:
            # 이미지 변환 실패 → 모델이 표를 못 봄(낮은 F1로 반영). 조용히 0점 나면 원인 파악이
            # 어려우니 경고를 남긴다(태스크당 소수라 로그 스팸 아님).
            print(f"  [IMG-6 이미지 변환 실패] s{sample.sample_id}: 빈 이미지로 진행(낮은 F1 예상)")
            data_url = ""
        prompt = (
            "This image contains a table. Extract the table and output it as a valid HTML "
            "table using <table>, <tr>, and <td> tags, preserving the row/column structure "
            "and cell text exactly as shown. Respond with ONLY the HTML, no explanation."
        )
        return build_image_message(prompt, data_url)

    def parse_output(self, raw_text: str, sample: Sample) -> str:
        """응답에서 <table>...</table> 추출(없으면 전체 반환 → 낮은 F1)."""
        m = re.search(r"<table[^>]*>.*?</table>", raw_text, re.DOTALL | re.IGNORECASE)
        return m.group(0) if m else raw_text.strip()

    def score(self, parsed: list[str], samples: list[Sample]) -> dict[str, Any]:
        """예측 HTML vs 정답 HTML을 Cell-F1로 채점(TXT-3와 동일 로직)."""
        f1s = []
        for pred_html, sample in zip(parsed, samples):
            pred_cells = parse_html_table(pred_html or "")
            gold_cells = parse_html_table(sample.reference)
            f1s.append(cell_f1_score(pred_cells, gold_cells))
        avg = sum(f1s) / len(f1s) if f1s else 0.0
        return {
            "cell_f1": avg,
            "n_evaluated": len(f1s),
            "notes": "표 이미지 → HTML, Cell-F1(셀 퍼지 매칭 threshold=0.8). TXT-3와 동일 채점.",
        }

    def make_accumulator(self) -> MeanAccumulator:
        """스트리밍 O(1) 채점기. score()와 동일(cell_f1 평균 + n_evaluated + notes).

        모든 샘플 채점(빈 예측도 낮은 F1로 포함) → count_all, n_evaluated=len(parsed).
        """
        def value_fn(pred_html, sample):
            pred_cells = parse_html_table(pred_html or "")
            gold_cells = parse_html_table(sample.reference)
            return cell_f1_score(pred_cells, gold_cells)

        return MeanAccumulator(
            out_key="cell_f1",
            value_fn=value_fn,
            count_all=True,
            static={"notes": "표 이미지 → HTML, Cell-F1(셀 퍼지 매칭 threshold=0.8). TXT-3와 동일 채점."},
        )

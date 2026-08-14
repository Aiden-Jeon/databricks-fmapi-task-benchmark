"""Common utilities for KLUE-NER task."""
import csv
import re
from collections import Counter, defaultdict

TASK_DIR = "/tmp/kmle/M7_t20_klue_ner_full_20260813_074004/task"
LABELS = ["PS", "LC", "OG", "DT", "TI", "QT"]


def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_entities(s):
    """Parse 'expr:TYPE|expr:TYPE' -> list of (expr, TYPE). Empty string -> []."""
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    out = []
    for tok in s.split('|'):
        tok = tok.strip()
        if not tok:
            continue
        expr, _, typ = tok.rpartition(':')
        out.append((expr, typ))
    return out


def entities_to_str(ents):
    return '|'.join(f"{e}:{t}" for e, t in ents)


def score_f1(gold_map, pred_map):
    """Entity-level micro-F1 over multisets. Maps: id -> list of (expr, type)."""
    tp = fp = fn = 0
    for k in gold_map:
        g = Counter(gold_map[k])
        p = Counter(pred_map.get(k, []))
        for key in set(g) | set(p):
            tp += min(g[key], p[key])
            fp += max(0, p[key] - g[key])
            fn += max(0, g[key] - p[key])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


# ---------------- Rule-based span extraction ----------------

# Date patterns
DT_PATTERNS = [
    # YYYY년 MM월 DD일 and variants
    re.compile(r'\d{1,4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일'),
    re.compile(r'\d{1,4}\s*년\s*\d{1,2}\s*월(?!\s*호)'),
    re.compile(r'\d{1,4}\s*년(?!\대)'),
    re.compile(r'\d{1,2}\s*월\s*\d{1,2}\s*일'),
    re.compile(r'\d{1,2}\s*월(?!\s*호)'),
    re.compile(r'\d{1,2}\s*일'),
    # MM.DD or YYYY.MM.DD
    re.compile(r'\d{4}\.\d{1,2}\.\d{1,2}'),
    re.compile(r'\d{1,2}\.\d{1,2}'),
    # MM/DD
    re.compile(r'\d{1,2}/\d{1,2}'),
    # 세기/연대
    re.compile(r'\d{1,2}세기'),
    re.compile(r'\d{2,4}년대'),
    # 요일
    re.compile(r'[월화수목금토일]요일'),
    # 상대 날짜
    re.compile(r'(그제제?|어저께|어제|오늘|내일|모레|글피|이틀\s*후|사흘\s*후|나흘\s*후)'),
    re.compile(r'(작년|금년|올해|내년|재작년|후년|이번\s*해|지난해|다음\s*해)'),
    re.compile(r'(이번|지난|다음|오는|지난번)\s*(주|달|월|분기|반기|해|연도|시즌)'),
    re.compile(r'(매년|매월|매일|매주|연초|연말|월초|월말|연중|주초|주말)'),
    re.compile(r'(상반기|하반기|전반기|후반기|[1-4]분기)'),
    re.compile(r'(오는날|이날|당일|당시|현재|최근|과거|미래|지금|요즘|올\s*초|올\s*말)'),
    re.compile(r'(어젯밤|오늘밤|내일밤)'),
    re.compile(r'(연휴|방학|휴가철|성수기|비수기)'),
    re.compile(r'\d{1,2}월초|\d{1,2}월말|\d{1,2}월중'),
]

# Time patterns
TI_PATTERNS = [
    # HH:MM(:SS)
    re.compile(r'\d{1,2}:\d{2}(:\d{2})?'),
    # X시 Y분 Z초
    re.compile(r'\d{1,2}\s*시\s*\d{1,2}\s*분\s*\d{1,2}\s*초'),
    re.compile(r'\d{1,2}\s*시\s*\d{1,2}\s*분'),
    re.compile(r'\d{1,2}\s*시\s*반'),
    re.compile(r'\d{1,2}\s*시(?!\간)'),
    re.compile(r'\d{1,2}\s*분\s*\d{1,2}\s*초'),
    re.compile(r'\d{1,2}\s*분(?!\대)'),
    re.compile(r'\d{1,2}\s*초'),
    re.compile(r'(오전|오후|새벽|아침|낮|정오|저녁|밤|자정|심야|평일|주중|주말|점심시간)'),
]

# Quantity patterns
QT_PATTERNS = [
    # percent
    re.compile(r'\d+(\.\d+)?\s*%|%\s*\d+(\.\d+)?'),
    re.compile(r'\d+(\.\d+)?\s*퍼센트'),
    re.compile(r'\d+(\.\d+)?\s*ppm'),
    # money: 숫자+원/달러/엔/유로/위안/파운드/원화 etc
    re.compile(r'\d[\d,]*(\.\d+)?\s*(억|조|만|천)?\s*(원|달러|엔|유로|위안|파운드|루피|동|페소|링깃|바트|호주달러|홍콩달러)'),
    re.compile(r'(수|약|총|연|월|일|최대|최소|평균)?\s*\d[\d,]*(\.\d+)?\s*(억|조)\s*원'),
    # counts with 만/억/조/천
    re.compile(r'\d+(\.\d+)?\s*(만|억|조|천)\s*(명|개|건|대|가구|마리|표|원|건수|세대|권|점|톤|t|kg|g|m|km|명분|인분|배)?'),
    # 수량 단위
    re.compile(r'\d[\d,]*(\.\d+)?\s*(명|개|건|대|가구|마리|표|권|점|톤|t|kg|g|mg|m|cm|mm|km|미터|킬로미터|센티미터|밀리미터|리터|l|ml|cc|도|배|회|차례|번|가지|종류|곳|개국|개월|시간|일|주|주일|개년|년|층|동|호|호수|채|필지|헥타르|ha|평|㎡|m2|㎢|kw|kwh|mw|gw|kcal|칼로리|인분|명분|마력|석|말|되|홉|자|치|푼|냥|돈|근|관|척|벌|켤레|갑|통|병|캔|잔|컵|숟가락|스푼|알|정|포기|송이|자루|줄|다발|묶음|세트|조각|장|매|쪽|페이지|p|부|사본|링크|바이트|byte|kb|mb|gb|tb|bit|bps|hz|khz|mhz|ghz|볼트|v|암페어|a|옴|와트|w|줄|j|뉴턴|n|파스칼|pa|기압|hpa|mmhg|노트|마일|해리|피트|ft|인치|inch|야드|파운드|lb|온스|oz|갤런|gal|배럴|bbl|포인트|p|점수|득점|실점|타점|안타|홈런|삼진|볼넷|도루|승|패|무|세이브|이닝|타율|방어율|할|푼|리|문|방|정|초|분|시|초등|중등|고등|대학|학년|학기|학점|점)'),
    # 카운트 한국어
    re.compile(r'(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|스무|서른|마흔|쉰|예순|일흔|여든|아흔)\s*(명|개|건|대|가구|마리|표|권|점|번|가지|종류|곳|시간|일|주|해|년|개월|달|층|동|채|살|세|배)'),
    # 나이
    re.compile(r'\d{1,3}\s*세'),
    # 순위
    re.compile(r'\d+\s*위'),
    # 기타 숫자+한국어 단위는 QT일 가능성 (보수적으로)
    re.compile(r'\d[\d,]*(\.\d+)?\s*(배)'),
]

ALL_RULE_PATTERNS = [(DT_PATTERNS, 'DT'), (TI_PATTERNS, 'TI'), (QT_PATTERNS, 'QT')]


def extract_rule_spans(sentence):
    """Return list of (start, end, type) from regex rules. Greedy priority DT>TI>QT, longest first."""
    spans = []
    occupied = [False] * len(sentence)
    for patterns, typ in ALL_RULE_PATTERNS:
        cands = []
        for pat in patterns:
            for m in pat.finditer(sentence):
                cands.append((m.start(), m.end(), typ))
        # longest first, then leftmost
        cands.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
        for s, e, t in cands:
            if any(occupied[s:e]):
                continue
            for i in range(s, e):
                occupied[i] = True
            spans.append((s, e, t))
    spans.sort()
    return spans


def build_gazetteer(rows):
    """Build gazetteer: expr -> Counter(type), and per-token info."""
    cnt = defaultdict(Counter)
    for row in rows:
        for expr, typ in parse_entities(row.get('entities', '')):
            cnt[expr][typ] += 1
    return cnt

"""Chu-Liu-Edmonds maximum spanning arborescence for dependency decoding.

그래프: 노드 0..N. 0 = 가상 ROOT. 각 노드는 최대 하나의 head를 가지며,
전체는 0을 루트로 하는 spanning arborescence(트리)를 이룬다.
엣지 가중치는 scores[parent][child] 형태의 음이 아닌 점수(클수록 좋음).
"""
import numpy as np


def chu_liu_edmonds(scores):
    """scores: (N+1, N+1) array, scores[h, m] = head h -> modifier m.
    h=m 인 대각선은 -inf. scores[0, m] 은 root->m. m=0 인 열은 사용 안 함(-inf).
    반환: heads 리스트, heads[m] = m의 head (1-indexed 상의 노드, 0=ROOT). 길이 N+1, heads[0]= -1.
    """
    N = scores.shape[0] - 1  # 노드 1..N, 0은 ROOT
    # 작업용 그래프: 각 노드(1..N)에 대해 최적 incoming edge 선택
    INF = 1e9
    score = np.array(scores, dtype=float)
    # 자기 자신으로의 엣지 제거
    for i in range(N + 1):
        score[i, i] = -INF
    # 0번 노드는 head가 될 수 없음(누구도 0의 head가 될 수 없으므로 score[*,0] = -inf)
    for i in range(N + 1):
        score[i, 0] = -INF

    node_ids = list(range(1, N + 1))  # 현재 그래프의 실제 노드 (1..N)
    # 맵핑: 원래 노드 -> 현재 축약 그래프 노드 번호 (여기서는 처음엔 동일)
    # 일반적인 CLE 구현: 축약된 사이클을 하나의 슈퍼노드로 묶고 재귀.
    return _cle(score, N)


def _cle(score, n_nodes):
    """n_nodes: score 행렬의 크기 - 1 (루트 0 포함하여 (n+1)x(n+1))."""
    INF = 1e9
    n = n_nodes
    # 각 노드(1..n)의 최적 head 선택
    heads = np.zeros(n + 1, dtype=int)
    for m in range(1, n + 1):
        col = score[:, m].copy()
        col[m] = -INF
        col[0] = -INF if False else col[0]  # root는 허용
        # root(0) 도 허용
        best = -INF
        besth = 0
        for h in range(n + 1):
            if h == m:
                continue
            if col[h] > best:
                best = col[h]
                besth = h
        heads[m] = besth

    # 사이클 탐지
    cycle = _find_cycle(heads, n)
    if cycle is None:
        return heads
    # 사이클 축약
    # 사이클 노드들을 하나의 슈퍼노드 c_id로 묶기
    cycle_set = set(cycle)
    # 새 그래프: 사이클을 제외한 노드 + 1개의 슈퍼노드
    non_cycle = [v for v in range(1, n + 1) if v not in cycle_set]
    new_n = len(non_cycle) + 1  # 슈퍼노드 1개 추가
    # 노드 번호 맵핑
    new_index = {}
    for i, v in enumerate(non_cycle):
        new_index[v] = i + 1
    super_id = new_n  # 슈퍼노드는 마지막 번호
    new_score = np.full((new_n + 1, new_n + 1), -INF, dtype=float)
    # 사이클 내부 점수 합
    cycle_score = 0.0
    for v in cycle:
        cycle_score += score[heads[v], v]

    # 노드간 엣지 복사 + 슈퍼노드 처리
    for v in range(1, n + 1):
        for u in range(1, n + 1):
            if u == v:
                continue
            if v in cycle_set and u in cycle_set:
                continue  # 내부 엣지는 스킵
            s = score[u, v]
            if v in cycle_set:
                # v가 사이클에 속 -> 들어오는 엣지는 슈퍼노드로
                # score[u -> cycle] = s - score[heads[v], v] + cycle_score
                adjusted = s - score[heads[v], v] + cycle_score
                if u in cycle_set:
                    continue
                nu = new_index[u]
                ns = super_id
                if adjusted > new_score[nu, ns]:
                    new_score[nu, ns] = adjusted
            elif u in cycle_set:
                # u가 사이클에 속, v는 외부 -> 나가는 엣지는 슈퍼노드에서
                nu = super_id
                nv = new_index[v]
                if s > new_score[nu, nv]:
                    new_score[nu, nv] = s
            else:
                nu = new_index[u]
                nv = new_index[v]
                if s > new_score[nu, nv]:
                    new_score[nu, nv] = s
    # root(0) 관련 엣지
    for v in range(1, n + 1):
        s = score[0, v]
        if v in cycle_set:
            adjusted = s - score[heads[v], v] + cycle_score
            if adjusted > new_score[0, super_id]:
                new_score[0, super_id] = adjusted
        else:
            nv = new_index[v]
            if s > new_score[0, nv]:
                new_score[0, nv] = s
    for u in range(1, n + 1):
        s = score[u, 0]
        if s > -INF:
            if u in cycle_set:
                pass
            else:
                nu = new_index[u]
                if s > new_score[nu, 0]:
                    new_score[nu, 0] = s

    # 재귀
    new_heads = _cle(new_score, new_n)
    # 복원
    final_heads = np.zeros(n + 1, dtype=int)
    # 외부 노드들의 head 복원
    # new_heads[v] 가 슈퍼노드이면 사이클 내의 어느 노드인지 결정
    for v in non_cycle:
        nh = new_heads[new_index[v]]
        if nh == super_id:
            # v의 head는 사이클 내 어느 노드. 원본 score에서 사이클 내 노드 중 최대
            best = -INF
            besth = cycle[0]
            for u in cycle_set:
                if score[u, v] > best:
                    best = score[u, v]
                    besth = u
            final_heads[v] = besth
        elif nh == 0:
            final_heads[v] = 0
        else:
            # nh -> 원래 노드
            orig = non_cycle[nh - 1]
            final_heads[v] = orig
    # 슈퍼노드의 head 결정 -> 사이클 내 한 노드의 head를 외부로 변경
    sh = new_heads[super_id]
    if sh == 0:
        # root가 사이클로 들어오는 엣지: 사이클 내에서 score[0,v] - score[heads[v],v] 최대인 v
        best = -INF
        bestv = cycle[0]
        for v in cycle_set:
            val = score[0, v] - score[heads[v], v]
            if val > best:
                best = val
                bestv = v
        final_heads[bestv] = 0
        for v in cycle_set:
            if v != bestv:
                final_heads[v] = heads[v]
    else:
        orig_head = non_cycle[sh - 1]
        # 사이클 내에서 score[orig_head, v] - score[heads[v], v] 최대인 v
        best = -INF
        bestv = cycle[0]
        for v in cycle_set:
            val = score[orig_head, v] - score[heads[v], v]
            if val > best:
                best = val
                bestv = v
        final_heads[bestv] = orig_head
        for v in cycle_set:
            if v != bestv:
                final_heads[v] = heads[v]
    return final_heads


def _find_cycle(heads, n):
    """heads[1..n] 에서 사이클 탐지. 사이클 노드 리스트 반환, 없으면 None."""
    color = [0] * (n + 1)  # 0=white,1=gray,2=black
    for start in range(1, n + 1):
        if color[start] != 0:
            continue
        path = []
        cur = start
        while cur != 0 and color[cur] == 0:
            color[cur] = 1
            path.append(cur)
            cur = heads[cur]
            if cur == 0:
                break
            if cur in path:
                # 사이클 발견
                idx = path.index(cur)
                cycle = path[idx:]
                return cycle
        for v in path:
            color[v] = 2
    return None

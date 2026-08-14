# COST — Custom Agent 성능

## 단가 (DBU per 1M tokens)

`function-calling-json`·`image-text-performance` 교차검증 값. `usd_per_dbu = 0.07`.

| 모델 | in | out | cache write | cache read |
| --- | --- | --- | --- | --- |
| opus | 71.429 | 357.143 | 89.286 | 7.143 |
| sol | 71.429 | 428.571 | 89.286 | 7.143 |
| glm | 20.000 | 62.857 | 0.0 | 3.714 |

산식: `usd = 0.07 × (fresh_in×in + cache_read×cr + cache_write×cw + completion×out) / 1e6`.
`fresh_in = prompt − cache_read − cache_write`. `billable_output = completion_tokens`
(reasoning 토큰은 completion 에 이미 포함 — 더하지 않는다).

## 실측 (180 세션)

| 모델 | 세션 | prompt tok | completion tok | 총 비용 | 세션당 비용 |
| --- | --- | --- | --- | --- | --- |
| opus | 60 | 721,759 | 133,299 | $6.94 | $0.1157 |
| sol | 60 | 319,425 | 43,622 | $2.12 | $0.0354 |
| glm | 60 | 435,696 | 67,679 | $0.47 | $0.0077 |

**전체 벤치마크 총비용: $9.53.**

## 비용비

- opus 는 glm 의 **14.9배**, sol 의 **3.3배**.
- sol 은 glm 의 4.6배.

비용을 가르는 것은 두 가지다.
1. **출력 단가** — opus out 357 / sol 429 / glm 63 DBU. glm 출력이 opus 의 1/5.7, sol 의 1/6.8.
2. **입력 토큰량** — opus 가 프롬프트를 가장 많이 소비했다(721k vs sol 319k). 멀티턴에서 대화 이력이
   누적되는데, adaptive reasoning 이 더 긴 사고·재조회를 유발해 프롬프트가 불어난 것으로 보인다.

## 멀티턴 컨텍스트 누적

- 세션당 프롬프트 토큰: opus 12.0k / glm 7.3k / sol 5.3k (평균).
- sol 은 200k 구간요금 임계에 한참 못 미쳤다(세션 최대도 수만 토큰). 그래서 flat 단가를 적용했다.
  더 긴 호라이즌(수십 스텝) 태스크였다면 sol 의 구간요금(200k 초과 시 in 2배·out 1.5배)이
  비용에 반영됐겠지만, 이 케이스 셋에서는 발생하지 않았다.

## 실무 함의

정확도가 세 모델 동률이므로 **비용이 곧 선택 기준**이다. 같은 에이전트를 대량으로 돌리는
운영 환경이라면 glm 이 opus 대비 15배 저렴하면서 정확도·지연 모두 경쟁력이 있다.
지연이 최우선이면 sol(중앙 12초)이 낫다. opus 의 프리미엄은 이 난이도에서 정당화되지 않는다.

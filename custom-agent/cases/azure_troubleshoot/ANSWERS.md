# 시나리오 1 (Azure Databricks 트러블슈팅) 정답지

> 채점 기준의 근거. 4건은 서로 다른 레이어를 다룬다 — Azure 인프라 2건(A-1 스토리지 권한, A-2 네트워킹) + Databricks 제품 2건(A-3 Unity Catalog, A-4 cluster policy).
> 모든 근본 원인은 공식 문서(`get_doc`)에 근거가 있다.

## A-1 — data-plane RBAC 누락 (Azure 스토리지 권한)

**근본 원인**: SP 에 management-plane 역할(Owner)만 있고 data-plane RBAC(`Storage Blob Data Contributor` 등)이 없다. HNS 계정에서 data-plane 역할이 없으면 POSIX ACL 로 평가되어 하위 경로에서 `AuthorizationPermissionMismatch`(403).

**조치**: `Storage Blob Data Contributor` 역할을 storage account 또는 container 스코프로 SP 에 할당.

**핵심 추론**: "루트는 되고 하위만 403" + "역할 목록에 Owner 만, Storage Blob Data \* 없음".

**함정**: Owner 를 subscription 스코프로 상향 / SAS 토큰 교체 / container 재생성.

---

## A-2 — Private DNS Zone 이 spoke VNet 에 미링크 (Azure 네트워킹)

**근본 원인**: `privatelink.azuredatabricks.net` 존이 hub VNet 에만 링크되고 클러스터의 spoke VNet 에 미링크. jumpbox(hub)는 private IP, 클러스터(spoke)는 public IP 로 해석 → public 차단으로 부트스트랩 실패.

**조치**: 존을 spoke VNet(`vnet-spoke-databricks`)에 링크.

**핵심 추론**: `nslookup(jumpbox)`=10.x 와 `nslookup(cluster)`=public IP 의 **차이**가 결정적. 두 위치를 모두 조회해 비교해야 한다.

**함정**: Private Endpoint 재생성 / NSG 완화 / jumpbox 정상이니 DNS 문제 아님.

---

## A-3 — UC READ FILES grant 누락 (Databricks / Unity Catalog)

경로 직접 접근(COPY INTO / abfss 직접 읽기)에는 external location 의 `READ FILES` 권한이 필요하다. (외부 테이블 SELECT 라면 테이블 SELECT 권한이 먼저이므로, 이 케이스는 경로 직접 접근으로 설정해 READ FILES 가 진짜 요구 조건이 되게 했다.)

**근본 원인**: storage credential·external location 이 모두 검증 PASSED 이고 Access Connector 의 Azure RBAC(Storage Blob Data Contributor + Storage Blob Delegator)도 정상이다. `analysts-kr` 그룹에 external location 의 `READ FILES` grant 가 없어 `PERMISSION_DENIED`. 관리자는 ALL PRIVILEGES 가 있어 읽힌다.

**조치**: `GRANT READ FILES ON EXTERNAL LOCATION ext-sales TO \`analysts-kr\``.

**핵심 추론**: 에러가 "READ FILES on External Location" 임을 읽고, grant 목록에 그룹이 없음을 확인. **A-1 과 정반대 교훈** — A-1 은 Azure RBAC 누락, A-3 은 Azure RBAC 는 맞고 UC grant 계층이 누락. "관리자는 되는데 특정 그룹만 안 됨"이 grant 문제의 신호.

**함정**: Access Connector 에 Storage Blob Data Contributor 재부여(이미 있음) / storage credential 재생성 / URL 수정 / Azure RBAC 스코프 상향. 전부 Azure 계층을 건드리는 오답.

---

## A-4 — cluster policy 위반 (Databricks / cluster policy, fault 복구 포함)

**근본 원인**: job 의 new_cluster `node_type_id` 가 `Standard_D4ds_v5` 인데, 어제(08-13 22:10) 갱신된 cluster policy `pol-standard` 의 node_type allowlist(`[Standard_DS3_v2, Standard_DS4_v2]`)를 벗어나 클러스터 생성이 거부됨. 노트북 코드 변경과 무관하다.

**조치**: job 의 `node_type` 을 allowlist 값으로 변경하거나, policy allowlist 에 `Standard_D4ds_v5` 추가.

**핵심 추론**: 에러가 "Cluster validation error" 임을 읽고 policy 정의를 조회해 allowlist 위반을 확인. policy 의 `modified_ts` 가 어제라는 점이 "코드는 안 바꿨는데 갑자기" 를 설명한다.

**fault(에러 복구)**: 첫 `get_job_run` 호출은 503 타임아웃으로 실패한다. 재시도해서 실패 메시지에 도달해야 한다. 첫 실패에서 포기하면 원인을 못 찾는다. 채점에서 `recovered_after_fault` 기록.

**함정**: DBR 버전 / 라이브러리 설치 실패 / 노드 용량 부족 / 노트북 코드 오류. 전부 코드·런타임을 의심하는 오답.

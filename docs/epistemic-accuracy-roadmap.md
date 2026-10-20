# 改进路线：从「能产出答案」到「产出可信知识」

> 状态：规划稿（基于现有代码核实，未实现）
> 目标：尽可能准确地向学习者输出知识与认知
> 依据：`docs/architecture/epistemic-integrity.md`、`docs/architecture/migration-plan.md`、`src/study_agent/`、`core/teaching_validator.py`

---

## 0. 先把目标拆成可验收的四件事

「准确输出知识」不是一个能直接测量的性质。它至少是四条不同的问题，混在一起会导致所有改进都变成主观感受：

| # | 失败模式 | 学习者侧后果 | 判定性质 |
|---|---|---|---|
| F1 | 答**错了**（算式、定义、日期、API 行为） | 学到错误事实 | 二值，可证伪 |
| F2 | 答**漏了**（题设条件、单位、子问题） | 解错题却不知道为什么 | 二值，可对照台账 |
| F3 | 答**歪了**（把「通常」写成「总是」，把「相关」写成「等同」） | 形成错误心智模型 | 有程度，难证伪 |
| F4 | 答**得含糊**（该说的不确定没说，或把「没找到反例」说成「已证实」） | 无法判断该不该信 | 状态机正确性 |

F1/F2 靠**确定性与台账**；F3 靠**解释性表述的边界约束**；F4 靠**认识论状态机的纪律**。

**本路线的主张**：F4 和 F2 是当前投入产出比最高的两块，因为它们**不需要新增知识资产**，只需要把已有结构接通；F3 的成本最高且永无「准确」，应当**刻意排到最后**，且只允许以「标注」形式存在。

---

## 1. 现状核实（不是文档声称，是代码实际）

### 1.1 已经扎实的部分

- `src/study_agent/domain/audit.py`：六态 `AuditOutcome`，且 `CandidateValidation.__post_init__` 把「REFUTED 必须 premise 成立 + 矛盾确认 + 非纯模型判定」写进了构造器；`ClaimAuditResult` 在此基础上再要求「至少一条 `confirms_refutation` 且能对上 candidate」。`is_proven_true` 恒为 `False`。**这是把认识论承诺编码为不可绕过的不变量，方向完全正确。**
- `src/study_agent/application/counterexample_audit.py`：`islice` 限制的是**迭代本身**而非最终 tuple（防无界生成器）；malformed 输出判 `UNRESOLVED` 而**不是** `NOT_FALSIFIED`；跨 claim 的 candidate 直接拒绝且不调用 verifier；单条验证异常不拖垮整次审计。
- `src/study_agent/domain/task_dossier.py`：`TaskDossier` 不可变、按 user/task/version 作用域；`SOURCE_EXPLICIT` 事实**强制**必须有 source_refs；`project()` 会显式返回 `missing_critical_fact_ids` / `missing_claim_fact_links`，`has_critical_context_loss` 可机读。
- 测试：`tests/test_counterexample_audit.py` 已覆盖「模型判定不能单独证伪」「无候选≠证明」「malformed≠NOT_FALSIFIED」「verifier 超时→UNRESOLVED」等关键纪律。

### 1.2 三个必须正视的缺口（已核实，非推测）

**缺口 A — 新能力完全没有接线。** grep `core/` 无任何 `CounterexampleAuditService` 引用；唯一相关的是 `core/storage_security.py` 的兼容 re-export。`migration-plan.md:9` 自己写明「not yet wired into every answer-generation path」。**当前用户拿到的每一个答案，走的都是旧的、纯 LLM 的 `TeachingValidator`。**

**缺口 B — 现行校验器仍然是纯 LLM，且没有凭据字段。** `core/teaching_validator.py:296-304` 的 `revision_findings` 只返回 `{claim, verdict, severity, correction}`，**没有 location、没有 evidence、没有 reproduction**。`_normalize_result`（`:42`）也不保留任何可复算字段。这正是 `epistemic-integrity.md` §5「An audit finding is useful only if a student or Teacher can act on it」尚未满足的地方。

**缺口 C — `NOT_FALSIFIED` 的一个真实伪通过路径。** `counterexample_audit.py:151-161`：当所有候选都无效时判 `NOT_FALSIFIED`。此时 `validations` 非空，但**这些候选可能全是 verifier 直接判 invalid 的垃圾**。而 `test_invalid_candidate_does_not_become_a_confirmed_error`（`tests:111-129`）恰恰把这个行为固化成了期望值。语义上「试过了，都不成立」确实不是证明，但与 `NOT_TESTED` 的区别在用户界面上会消失——学习者看到的是「没报错 = 应该是对的」。

---

## 2. 路线总览

```
P0  接线 + 台账          ← 零新增知识资产，只接通已有结构
 │   让 Dossier 进入真实请求路径；让 F2 可测
 ▼
P1  确定性校验落地       ← 唯一的「能说硬话」的地方
 │   arith / calendar / source_ref，接进 Validator 修订回路
 ▼
P2  状态机纪律与呈现     ← 让 F4 在界面上可见
 │   NOT_FALSIFIED 与 NOT_TESTED 分离；新增 attempted_methods
 ▼
P3  解释性表述边界       ← F3，成本最高，只做「标注」
 │   绝对化词 / 常见误简化 / 适用范围缺失
 ▼
P4  评测闭环             ← 决定以上是否真的有效
```

**排序原则**：每期都必须满足「不达标就不接进用户可见路径」。宁可三期做完只上线 P0+P1，也不要五期同时半成品——半成品的校验器比没有校验器更危险，因为它消耗信任。

---

## 3. P0 — 接线与任务台账（最高优先）

**为什么最先**：F2（漏掉题设）是学习者-最痛且**最容易做对**的一类，而且 Dossier 的数据结构**已经写完了**，只是没接上。这是投入产出比最高的一步。

### 3.1 让 Dossier 进入真实请求路径

- 在 `StudyAgent.run` 的入口处构建 `TaskDossier` v1，而非在 Teacher 之后补建。
- `Teacher` 与 `TeachingValidator` 不再接收原始 transcript，而是接收 `dossier.project(...)` 的**角色视图**。
- **关键约束**：任何一次投影若 `has_critical_context_loss` 为真，必须在输出侧可见（不是日志），因为那意味着校验依据本身残缺。

**验收**：`tests/architecture/test_import_boundaries.py` 之外新增集成测试——构造一个「题设有 3 个 critical 条件、其中 1 个被上下文压缩丢弃」的场景，要求系统**不能**静默输出 PASS。

### 3.2 给 finding 加结构化凭据

改 `core/teaching_validator.py::_normalize_result`，向后兼容地新增可选字段：

```
claim_id / answer_span / evidence_refs / violated_condition / source = "judge" | "oracle"
```

**不破坏**现有 `status/claims/summary` 契约（`:307` 的兼容性约束保持）。`source` 字段是 P1 分级处置的前提。

**验收**：现有 `tests/test_teaching_validator_audit_packet.py` 全绿；新增测试证明旧格式输入仍被接受、新字段缺失时行为不变。

### 3.3 明确定义「答漏了」的判据

不看模型怎么说，而看 **coverage ledger**：每条 `critical=True` 的 `TaskRequirement` 在答案中必须有明确状态（`SATISFIED / OMITTED / CONTRADICTED / NOT_APPLICABLE`）。任何 `OPEN` 在输出时视为**未完成**。

> 这比「让模型检查是否遗漏」强得多：模型可以漏看，但台账里的 `OPEN` 漏不掉。

---

## 4. P1 — 确定性校验（唯一能说硬话的层）

### 4.1 只做证伪，只做有预言机的类别

按收益/成本，**只做三项**，都不需要 `sympy`/`z3`：

| 检查器 | 判定对象 | 预言机 | 误报 |
|---|---|---|---|
| `arith` | 算式结果、单位换算、量纲 | `decimal` + AST 白名单（**项目已有**，见 README「Calculator safety」） | ≈0 |
| `calendar` | 闰年、星期、日期区间 | `dateutil`（已在运行时） | ≈0 |
| `source_pin` | 「第 N 行是 X」「文件里写了 Y」 | 对照 Dossier 的 `SourceReference.content_hash` + locator | ≈0 |

**复用而非新造**：`arith` 直接用现有 Calculator 的 AST 白名单解释器，`epistemic-integrity.md` §4 的「Use available deterministic tools before adding another model call」即此意。

### 4.2 输出严格复用 `AuditOutcome`

不发明新枚举——把确定性结果映射到已有六态：算错 → `REFUTED`（`method=CALCULATION` + `reproduction` 必填）；算对 → `SUPPORTED_WITH_EVIDENCE`；不在能力域 → `NOT_TESTED`（**不是** `NOT_FALSIFIED`）。

### 4.3 分级处置

- `source="oracle"` + `REFUTED` → **强制修订**（零误报，值得阻塞）
- `source="judge"` + major → 修订一次（维持现状）
- 任一 `NOT_TESTED` → **必须呈现为「未校验」，不得呈现为通过**

---

## 5. P2 — 认识论状态机的纪律

### 5.1 修掉 `NOT_FALSIFIED` 的伪通过路径

`NOT_FALSIFIED` 当前对凭据零要求。建议最小改动：

1. 给 `ClaimAuditResult` 增加 `attempted_methods: tuple[CounterexampleMethod, ...]`；
2. 「无效候选」与「无候选」区分记账：全部候选 invalid → 该 claim 的 `NOT_FALSIFIED` **降级为 `UNRESOLVED`**，理由写「生成的候选全部不成立，不等于主张成立」；
3. 只有「尝试方法覆盖了 ≥1 个实质性类别」时才允许输出 `NOT_FALSIFIED`。

**注意**：这会改变 `tests/test_counterexample_audit.py:111-129` 的期望值。**这是修复，不是回归**——那条测试把「垃圾候选 → NOT_FALSIFIED」固化了，恰恰是缺口 C。

### 5.2 用户可见的措辞契约

| 内部状态 | 允许的措辞 | 禁止的措辞 |
|---|---|---|
| `REFUTED` | 「这里算错了，反例是…」 | — |
| `SUPPORTED_WITH_EVIDENCE` | 「有证据支持，依据是…」 | 「已证明」 |
| `NOT_FALSIFIED` | 「已尝试[方法]，未找到反例」 | 「正确」「已验证」 |
| `UNRESOLVED` / `NOT_TESTED` | 「这一点没有校验」 | 静默略过 |

> 最后一行是硬要求：**静默等于暗示通过**。这是整条路线里最便宜、收益最大的一条纪律。

---

## 6. P3 — 解释性表述的边界（F3）

**先说清定位**：F3 没有「准确」解，只有「标出来供人判断」。因此本期的产出**只能是标注**，不得阻塞输出。

### 6.1 三类信号（不引入语义相似度、不用 LLM 判分）

1. **绝对化词检查**：`总是 / 从不 / 完全 / 唯一 / 必然` 出现在教学生成文本中 → 提示补充边界。**必须按领域白名单排除**（数学的真绝对命题不应被误伤）。
2. **已知误简化**：维护一份**很小**的、按概念组织的手工词表（如「线程不共享内存」「相关系数等于因果」），命中即标红。这是唯一能精确捕获 F3 的确定性手段。
3. **适用范围缺失**：解释类表述若不包含「在什么条件下不成立」，标注为待补充。

### 6.2 为什么刻意不做「覆盖度差分」

另一份设计（骨架差分 + 触发词覆盖度）在这里是合理的，但**它的代价是维护领域骨架**，而骨架维护成本随概念线性上升、且「一个错的骨架比没有骨架更危险」。在当前阶段：

- 骨架会引入**新的权威性来源**，需要独立评审流程；
- 而 F2 已经由 P0 的 requirement ledger 覆盖了「该说没说」的大部分场景，**不需要两套机制。**

**结论**：F3 留待 P0–P2 的指标稳定后，仅当 F2 台账证明「遗漏已收敛、剩余错误集中在表述歪曲」时才启动。启动时也**只做 §6.1 的三类信号**，不做完整覆盖度评分。

---

## 7. P4 — 评测（决定以上是否有效）

没有这一层，前面全凭主观。按 `epistemic-integrity.md` §7：

**必测切片**
- **细节保留率**：上下文压缩后，critical 条件 / 单位 / 输出格式的保留比例（P0 直接对应）
- **反例精确率**：被接受的反例中真正成立且真正矛盾的比例
- **反例召回率**：已知错误中被找到并验证出反例的比例
- **误报率**：正确主张被误判的比例 → **硬门槛**
- **校准**：`REFUTED / NOT_FALSIFIED / UNRESOLVED` 是否被正确区分（P2 直接对应）

**验收门槛（建议）**
- P0/P1：`refutation_precision ≥ 0.95`
- P2 起：`false_alarm_rate ≤ 0.10`，否则**不得接进用户可见路径**，只在日志试跑
- 每个概念样本 < 20 时**报告原始计数**（如「30 个 clean 样本 2 个误报」），不报百分比

**必须承认的陷阱**
1. 自标注循环：至少一部分 clean 样本须来自**独立来源**（教材原文、权威文档）。
2. 骨架/词表过拟合：指标必须声明适用概念范围与数量。
3. 「没找到反例」不能计入准确率分子。

---

## 8. 明确不做的事

1. **不做通用自然语言形式化**——形式化本身就是要被检验的环节，把不可靠环节放最前面，后面再准也救不回。
2. **不追求「幻觉率归零」**——现实目标是：F1/F2 在确定性覆盖域内接近零；F3 从「完全不可见」变为「可见、可定位、需人判断」。
3. **不让校验器自我进化扩权**——规则变更必须版本化 + 重跑评测。
4. **不在 P0–P2 完成前启动 F3 骨架工程**——避免用新资产掩盖接线缺口。
5. **不把「模型也同意」当证据**——`audit.py` 的既有约束必须保持，不得为降低误报率而放宽。

---

## 9. 一条最先动手的具体改动（建议）

如果只做一件事：

> **把 `CounterexampleAuditService` 接进 `direct_verified` 路径，且只接 P1 的 `arith` 检查器，其余 claim 一律返回 `NOT_TESTED` 并在答案里明示「未校验」。**

理由：它同时验证了「接线（P0）」和「确定性校验（P1）」两个最大缺口，且**零误报风险**（`NOT_TESTED` 不构成任何肯定性断言）。这是唯一一条「即使只做一半也绝不会让系统变得更不可信」的路径。

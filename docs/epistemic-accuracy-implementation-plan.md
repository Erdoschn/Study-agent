# Epistemic Accuracy：实施开工计划

> 状态：待执行计划；当前文档提交只修订设计，不代表代码阶段已经完成。
> 计划日期：2026-10-20
> 设计基线：[epistemic-accuracy-roadmap.md](epistemic-accuracy-roadmap.md)
> 架构基线：[migration-plan.md](architecture/migration-plan.md)

## 目标

把已存在的 TaskDossier、审计类型和 CounterexampleAuditService 从可独立测试的架构组件接入真实 Study Agent 执行路径；先降低题设遗漏与“未校验却看似通过”的风险，再有限引入可复现的确定性算术检查。

本计划刻意以小 PR 分阶段推进，不在一次改动中重写 Reasoner、Teacher、模型路由、前端或整个模块结构。

## 开工规则

- 每一阶段开始前，先核实实际调用链并建立 characterization tests；每个 PR 只包含一个有界变更。
- 不声称未实际执行的本地测试或 CI 已通过。CI 结果作为合并依据。
- 新验证器先影子运行；未达到放行门槛前，不得改变用户答案的强制处置行为。
- 兼容现有 API 的 status/claims/summary 和已有执行模式；新字段先作为可选字段添加。
- 日志只记录必要的阶段、ID、耗时、状态和脱敏错误，不记录完整题目、附件正文、答案正文或学习者隐私。
- 所有持久化读取和写入必须显式校验可信用户身份与 task/version scope。
- 一旦档案构建或审计失败，不得静默报告 PASS；可以保留有用答案，但必须显式降低审计状态。

## Phase 0 — 运行时侦察与基准测试

### 工作项

1. 从真实 API/CLI 入口追踪执行模式分派，确认 direct、README 中描述的 direct_verified、reasoner 的实际实现位置；如果某个模式没有独立运行时落点，先记录事实并选择真实的兼容入口，不为了匹配名称而重造模式。
2. 画出当前调用图：请求入口 → 身份/任务上下文 → TaskAnalyzer → AgentToolLoop 或 direct path → Teacher → TeachingValidator → 输出序列化。
3. 识别当前用户身份是从哪个可信上下文进入执行层，不能把请求正文中的 user_id 当作授权凭据。
4. 记录现有测试命令、环境要求和基线结果；为每个实际路径添加 characterization tests，包括 Validator 不可用、Teacher 失败、附件/来源缺失和输出状态兼容。
5. 检查 SQLite repository 的初始化、配置路径、生命周期、并发和跨用户测试，不在此阶段重写数据库层。

### 产出

- 一份实际运行调用图和执行模式映射，更新到本计划或架构文档。
- 一份基线测试清单与当前失败/环境限制说明。
- 一组证明现有外部响应契约和失败回退行为的测试。

### 验收门槛

任何后续 PR 都能指出修改的实际调用点及相关测试；没有仅凭 README 名称推断接入点。

## Phase 1 — 契约与版本模型

### 工作项

1. 定义 AnswerRevision（或等价名称），把答案文本、答案哈希、revision ID、父 revision、dossier version 和 claim 列表关联起来。
2. 定义独立 CoverageEntry/CoverageOutcome，保留 TaskRequirement 自身状态和答案覆盖状态之间的语义区分。
3. 统一 AuditReport 对 user_id、task_id、dossier_version、answer_revision_id 的引用。
4. 决定现有 TaskDossier v1 claims 字段的兼容迁移方案；如果持久化语义改变，升级 schema_version 并为旧记录提供明确处理策略。
5. 明确可信 user_id 来源、未认证/缺少身份时的失败行为、内容保留期限和审计数据访问控制。
6. 为 Finding 设计可选结构化字段：claim_id、answer_span、evidence_refs、violated_condition、validator/method、validation provenance、reproduction 和 unresolved_reason。
7. 审查 frozen dataclass 内嵌可变 dict/list 的问题，对审计证据采用标准化序列化/不可变包装/哈希等受控方案。

### 产出

- 类型与序列化契约。
- schema 迁移和兼容测试。
- 用户隔离、版本引用与不可变性测试。

### 验收门槛

旧数据不会被静默误读；交叉 user/task/version 的读写会被拒绝；旧 API 消费者不会因为新增字段被破坏。

## Phase 2 — TaskDossier 接入真实请求路径

### 工作项

1. 在 Phase 0 已确认的真实入口创建或加载当前任务版本，优先使用用户提供的来源，不把模型推断升级成 SOURCE_EXPLICIT。
2. 让 Reasoner、Teacher、Validator 接收角色专用视图，并记录源 dossier version 和包含的 fact/requirement IDs。
3. 对 critical context loss、未解析来源和 OPEN coverage 定义一致的结构化错误/警告状态。
4. 建立答案版本，形成每条 critical requirement 对应的覆盖结果。
5. 先以影子模式比较旧输出和新档案路径，不改变原有强制处置策略；确认兼容后才让遗漏状态进入用户可见结果。
6. 确保取消、超时或档案存储失败时，不会让下一请求复用上一次的未完成 dossier/answer revision。

### 关键集成测试

- 三个 critical 条件中任意一个在投影/压缩中丢失，结果不能静默 PASS。
- 用户要求 SI 单位、三位有效数字或只给最终答案时，覆盖台账必须分别跟踪。
- 来源模糊、文件读取失败、用户身份缺失、存储失败时，不允许生成伪来源或伪通过状态。
- 两个用户针对相同 task_id 执行时，档案和审计历史不能交叉读取。

### 验收门槛

所有关键要求都有答案版本级的覆盖状态；关键上下文损失显式可见；现有可用答案和错误恢复行为保持兼容。

## Phase 3 — 最小确定性计算验证器

### 工作项

1. 定义窄的计算验证 port 与结果契约，不让 domain/application 层直接导入 ToolLoop。
2. 安全复用现有 AST 白名单计算能力；先支持表达式输入和可复现数值结果，不宣称已具备单位、量纲或日期验证。
3. 将输入表达式、参数来源/绑定、期望 claim、实际输出、容差策略和计算器版本关联起来。
4. 用已知正确与已知错误样本验证边界、浮点误差、除零、非有限值、大表达式、超大指数与超时行为。
5. 应用层只有在 oracle 实际执行、输入绑定可信、结果结构有效时才标记 oracle provenance。
6. 对能力范围之外的 claim 返回 NOT_TESTED；对验证异常或输入含糊返回 UNRESOLVED。

### 验收门槛

已验证算术错误可以产生可复现 finding；仅算术通过不能把整段答案设为已验证；模型不能把自己创建的 finding 标记成 oracle。

## Phase 4 — 审计状态机与修订闭环

### 工作项

1. 修正“空候选/全部无效候选”与 NOT_FALSIFIED 的语义；候选级 INVALID_COUNTEREXAMPLE 不能直接代表 claim 正确。
2. 为 REFUTED 设置严格构造约束：claim 和 candidate 匹配、前提成立、矛盾确认，以及适当的非模型独立证据。
3. attempted_methods 仅记录真正执行过的校验方法和结果，不接受模型单方面声明。
4. 给每个状态建立可测试的用户文案映射，明确区分 NOT_TESTED 与 UNRESOLVED。
5. 自动修订最多执行明确限定的次数；每次修订产生新 AnswerRevision，并重新检查关联 claims 与全部 critical requirements。
6. 修订或复审未通过时，保留可用内容并明确说明未解决项，不能靠重复调用直到 PASS。

### 验收门槛

针对空候选、malformed 输出、重复 candidate ID、跨 claim 候选、verifier 超时、有效反例、来源冲突和修订后仍错误都有状态转换测试。任何失败路径都不能冒充 PASS。

## Phase 5 — 评测、影子运行与放量

### 工作项

1. 建立独立 clean claims 与已知错误 claims 样本；至少一部分标注来自教材原文或权威资料，不能完全由待测模型自行生成标签。
2. 按数学/物理计算、定义、来源引用、API 行为等类型切片。
3. 报告 refutation precision/recall、false alarm rate、有效检验覆盖率、critical requirement 保留率、遗漏率、状态校准准确性和失败恢复率。
4. 所有报告都包含样本数量、分母、原始计数和切片范围；样本不足时不只报告百分比。
5. 在影子模式下测量延迟、超时、额外模型调用、存储增长和用户可见文案影响。
6. 给新路径设置功能开关，准备回滚到旧路径的步骤；回滚不应删除已存储的审计记录。

### 放行规则

- 仅当 clean 样本量和置信度足以支撑误报门槛时，oracle 结果才可以触发强制修订。
- 初期可将强制修订误报率目标设为 ≤ 1%；样本不充分时继续影子运行或仅提示。
- 建议反例精确率目标 ≥ 0.95，但不能以总体平均掩盖某类 claim 的不良结果。
- NOT_TESTED/UNRESOLVED 不计入有效检验覆盖率。
- 任一关键身份隔离、版本关联、证据来源或状态机测试失败，均不得放量。

## Phase 6 — 非阻断性表述边界提示（以后再做）

仅在前几阶段的指标趋于稳定、且误差分析显示概念歪曲成为主要剩余问题时启动。按 claim 类型应用绝对化词、已知误简化和适用范围提示，默认只作标注，不阻断答案。必须评估误提示率并提供关闭开关。

## 推荐 PR 切分

1. PR A：调用链侦察、characterization tests、实际执行模式映射；不改变行为。
2. PR B：AnswerRevision/CoverageOutcome/AuditReport 契约、schema 兼容、用户隔离和不可变证据测试。
3. PR C：TaskDossier 影子接线与关键上下文/coverage 集成测试。
4. PR D：受限算术 verifier port 与独立测试集。
5. PR E：审计状态语义、用户可见状态和一次修订后的回归检查。
6. PR F：评测 fixture、指标报告、影子运行开关和回滚说明。

如果某 PR 暴露出未预期的架构问题，先补契约与测试，不把无关的重构合并进同一 PR。

## 开工第一步（现在）

只做 Phase 0：定位真实执行分派点、建立调用图和基准测试，不先改验证行为。最先交付的不是新功能，而是可证明的接入位置和可回归的当前行为。

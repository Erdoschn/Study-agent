# Epistemic Accuracy：当前运行时调用图（Phase 0）

> 基线分支：master
> 核查日期：2026-10-20
> 性质：静态代码追踪与既有测试映射，不代表执行过完整测试套件。
> 目标：明确未来 TaskDossier / 审计能力应当接入的真实位置，防止把 README 的策略图误当成可运行的调度器。

## 1. 结论摘要

1. 当前主运行时只有一个明确的 StudyAgent.run() 入口。正常请求在任务分析后进入 AgentToolLoop；每一步由 AgentReasoner 决定 ANSWER、SEARCH、CALCULATE、VERIFY、ASSESS 或 STOP。
2. ModelRouter 负责选择模型与 reasoning effort，不负责选择 direct、direct_verified 或 reasoner 执行路径。test/test_execution_routing.py 已明确保护此边界。
3. 在当前 master 的 StudyAgent.__init__ / run 及相应调用链中，没有找到独立的 direct_verified 运行时分派选项。因此 README 原先的“三条执行策略”图与代码不一致，不能将其当作已有接线钩子。
4. 对以 ANSWER 正常结束且配置了 Teacher 的任务，StudyAgent.run() 将 Reasoner 草稿交给 Teacher.generate()。标准 composition root 创建的 Teacher 默认启用 TeachingValidator；Validator 仍然是模型审查，而不是 CounterexampleAuditService。
5. TaskDossier、SQLite 档案存储和 CounterexampleAuditService 是可单独测试的新架构组件，但还没有被接入下面这条主要运行时路径。它们的存在不能算作线上答案已得到这些能力的保护。

## 2. 实际调用图

~~~text
CLI: main.py ─────────────────────────┐
Legacy HTTP: examples/study_agent_api.py ─┤
Unified UI/API: examples/study_agent_unified_api.py ─┤
MCP/other adapters (if using shared factory) ───────┘
                                         ↓
                        study_agent.bootstrap.build_agent
                        (ModelRegistry / ModelRouter /
                         ModelClientFactory / AgentReasoner /
                         Teacher / ToolExecutor)
                                         ↓
                                   StudyAgent.run
                                         ↓
                                  TaskAnalyzer
                                         ↓
                       AgentState + student/knowledge context
                                         ↓
                       AgentToolLoop(AgentReasoner, ToolExecutor)
                            ↙                        ↘
                     ReasoningDecision           Tool execution
                     ANSWER / STOP               SEARCH / CALCULATE /
                                                 VERIFY / ASSESS
                            ↓                        │
                            └──── Observe / Decide ──┘
                                         ↓
                   final action == ANSWER and draft exists
                                         ↓
                           StudyAgent._teach_final_answer
                                         ↓
                               Teacher.generate(draft)
                                         ↓
                         TeachingValidator.validate (default on)
                                         ↓
                    validated / revise / uncertain response
                                         ↓
                           AgentState.final_answer / API
~~~

注：不是所有异常都会进入 Teacher。例如附件上传被明确识别为致命失败时，StudyAgent.run() 会停止后续循环。非致命 TaskAnalyzer 异常则按既有策略进入带 fallback 标记的 Agent Loop。此阶段只记录现状，不修改这些运行策略。

## 3. 入口逐项映射

### 3.1 CLI

- 入口：main.py
- 构建：从 study_agent.bootstrap 导入 build_agent，加载配置后构建 Agent。
- 执行：交互循环把问题传给同一个 agent.run(question)。
- 当前任务身份：CLI 入口没有显式建立 Web API 那样的 context_scope(username, task_id)；对带 role_context 的独立调用，core/context_management.py 会生成隔离的 standalone context。未来持久化 Dossier 接入前，必须明确 CLI 的可信身份与 task 生命周期，不能把 standalone scope 当作持久用户身份。

### 3.2 OpenAI-compatible legacy API

- 入口：examples/study_agent_api.py。
- 构建：模块的 build_agent() 包装 study_agent.bootstrap.build_agent()，保留 API 专属的 browser prewarm hook。
- 执行：请求处理最终进入 _run_agent_with_cancellation(...)，由兼容包装检查 agent.run 是否支持 cancellation_event，再调用 agent.run(question, ...)。
- 既有特征测试：test/test_study_agent_api.py 覆盖 /health、/v1/models、非流式与 SSE 响应；tests/test_agent_cancellation.py 覆盖 Agent loop 取消边界。

### 3.3 Unified UI/API

- 入口：examples/study_agent_unified_api.py。
- API 先解析模型 ID / 用户名 / task ID，再通过 _run() 选择或创建模型对应的 Agent，并为相应用户取得 profile-specific knowledge graph。
- _run() 在锁与模型 run lifecycle 保护下调用 with context_scope(username, task_id)，再通过 _run_agent_with_cancellation(...) 调用 agent.run()。
- API 可根据模型类型使用对应配置建立 Agent，但创建出的核心对象仍是同一个 StudyAgent 执行入口；“浏览器模型 vs API 模型”是模型后端选择，不等于 direct_verified 执行策略。
- 既有特征测试：tests/test_deepseek_web_study_agent_api.py 覆盖附件边界、用户名/模型配置相关路径、清理和用户进度事件；test/test_unified_api.py 覆盖后端模型配置过滤与 prompt 历史。

### 3.4 Shared composition root

src/study_agent/bootstrap.py::build_agent() 共享创建一组对象：

- ModelRegistry(config)
- ModelRouter(registry)
- ModelClientFactory(config)
- AgentReasoner(model_router, model_factory, ...)
- Teacher(model_router, model_factory, ...)
- ToolExecutor(search_router=build_search_router())
- StudyAgent(reasoner, teacher, tool_executor, ...)

Reasoner 与 Teacher 共享 Router 和 Factory。该 composition root 是在多个入口接入统一 dossier/audit 能力时的首选组装点，但档案构建必须遵循请求的可信用户/task scope，不能简单作为全局 singleton。

既有特征测试：tests/test_bootstrap.py::test_build_agent_wires_one_shared_model_stack_and_optional_storage 与 test_main_keeps_legacy_builder_exports。

## 4. Agent Loop、Teacher 与 Validator 的现有语义

### 4.1 Agent Loop

core/agent.py::StudyAgent.run() 在非致命的任务分析路径之后调用 AgentToolLoop(...).run(state)。循环由 core/tool_loop.py 实现，Reasoner 每轮给出决策。简单问题也经过 Loop；它可以在第一轮直接选择 ANSWER。

既有测试：test/test_execution_routing.py::test_study_agent_always_enters_agent_loop_for_simple_question。

### 4.2 Teacher handoff

只有 Agent Loop 的最后一步是 ANSWER、并且存在最终草稿时，StudyAgent.run() 才调用 _teach_final_answer()。Teacher 支持新式 generate(state, draft_answer=...) 和 legacy generate(state) 适配器。如果 Teacher 抛错或返回空结果，现有代码会保留 Reasoner 草稿。

既有测试：test/test_teacher_integration.py::test_modern_answer_is_passed_through_teacher、test_teacher_failure_keeps_reasoner_draft、test_teacher_is_optional。

### 4.3 TeachingValidator

标准 composition root 默认构造 Teacher(..., validate_teaching=True)。Teacher 生成草稿后调用 TeachingValidator.validate()：

- Validator 从 verification capability 选择模型，并要求 JSON 结构化输出。
- 规范化只保留当前字段 status / claims / summary（运行时结果还包含尝试次数、模型、effort 等元数据）。
- finding 主要由 claim / verdict / severity / correction 组成；没有完整的 claim span、独立 oracle provenance、证据复现契约。
- 若没有可用 Validator 模型，当前实现返回 UNCERTAIN 且 attempted=False。Teacher 在没有已确认错误时会保留回答，但追加“独立校验未能确认答案完全正确”的提示，不应该记作已验证。
- Validator 的 PASS 是模型审查结果，不能等同于确定性计算或独立来源验证。

既有测试：test/test_teaching_validator.py、test/test_teacher.py 和 tests/test_teaching_validator_audit_packet.py。

## 5. 当前未接入的模块

以下组件存在于源码并有自己的测试，但从当前主运行时路径尚未找到生产调用关系：

- src/study_agent/domain/task_dossier.py
- src/study_agent/infrastructure/dossier_sqlite.py
- src/study_agent/domain/audit.py
- src/study_agent/application/counterexample_audit.py

docs/architecture/migration-plan.md 也明确注明 end-to-end integration remains open。Phase 1/2 应从此处补接线，不能在没有确认请求上下文和版本契约时直接从 Teacher 里面构建一个 service。

## 6. 当前能力边界

| 能力 | 当前事实 | Phase 0 的处理 |
|---|---|---|
| 执行策略 | 当前主要路径为 AgentToolLoop；未确认独立 direct_verified dispatch | 修正 README，特征测试固定现状 |
| 任务来源档案 | TaskDossier model/repository 已存在 | 暂不接入真实请求 |
| 输出校验 | TeachingValidator 仍由模型判断 | 记录 unavailable / uncertain 状态契约 |
| 算术 | ToolExecutor 内有白名单 AST arithmetic evaluator | 不能声称已有单位/量纲校验 |
| 身份上下文 | Unified API 使用 username + task_id context scope；CLI 路径不同 | Phase 1 必须定义持久化身份来源 |
| 持久化审计 | dossier SQLite 与 CounterexampleAuditService 有单元测试 | 当前不属于实际答案生成链路 |

## 7. Phase 0 验收基线

本阶段只增加调用图文档和 characterization tests，不改运行时行为。后续接线前应以真实 CI 结果为基线：

- [ ] test/test_execution_routing.py：简单问题仍走 AgentToolLoop，ModelRouter 不选择 execution strategy；未添加独立 direct_verified dispatch。
- [ ] test/test_teacher_integration.py：ANSWER 草稿经过 Teacher；Teacher 失败保留 Reasoner 草稿；Teacher 可选。
- [ ] test/test_teaching_validator.py：没有可用 Validator 模型时返回 UNCERTAIN/attempted=False，而不是 PASS。
- [ ] tests/test_bootstrap.py：共享 composition root 与旧导出兼容。
- [ ] API tests：旧式和 unified API 的响应格式、取消与清理仍兼容。
- [ ] tests/test_task_dossier.py / tests/test_dossier_sqlite_repository.py：确认当前只是组件级测试，不把它们算成 runtime integration coverage。

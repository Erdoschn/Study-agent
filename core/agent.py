from .state import AgentState, StudentState
from .tool_loop import AgentToolLoop
from .task_analyzer import TaskAnalyzer
from .knowledge_graph import KnowledgeGraph, normalize_difficulty
from .assessment import AssessmentEvaluator
from .assessment_generator import AssessmentGenerator
from .__debug__ import debug


class StudyAgent:
    """LLM-driven Study Agent: Harness 初始化状态，Reasoner 决策，Teacher 负责最终教学表达。"""

    def __init__(
        self,
        reasoner,
        teacher=None,
        tool_executor=None,
        max_steps=15,
        knowledge_graph_path=None,
    ):
        self.reasoner = reasoner
        self.teacher = teacher
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.student_state = None
        self.knowledge_graph = KnowledgeGraph(storage_path=knowledge_graph_path)
        self.assessment_evaluator = AssessmentEvaluator()
        self.assessment_generator = None
        if hasattr(self.reasoner, "model_router") and hasattr(self.reasoner, "model_factory"):
            self.assessment_generator = AssessmentGenerator(
                self.reasoner.model_router,
                self.reasoner.model_factory,
                allow_paid=getattr(self.reasoner, "allow_paid", False),
            )
        self.last_state = None
        self.pending_assessment_state = None

    def _teach_final_answer(self, state) -> None:
        """在 Reasoner 决定 ANSWER 后进入教学层；Teacher 失败时保留 Reasoner 草稿。"""
        if self.teacher is None or state.final_answer is None:
            return
        draft_answer = state.final_answer
        try:
            if self._teacher_supports_draft():
                state.final_answer = self.teacher.generate(state, draft_answer=draft_answer)
            else:
                state.final_answer = self.teacher.generate(state)
            if not state.final_answer:
                state.final_answer = draft_answer
                debug.log("StudyAgent", "TEACHER EMPTY → kept Reasoner draft")
        except Exception as exc:
            state.final_answer = draft_answer
            debug.log("StudyAgent", f"TEACHER FAILED → kept Reasoner draft: {type(exc).__name__}: {exc}")

    def _teacher_supports_draft(self) -> bool:
        try:
            import inspect
            params = inspect.signature(self.teacher.generate).parameters.values()
            return any(
                p.kind == inspect.Parameter.VAR_POSITIONAL
                or p.kind == inspect.Parameter.VAR_KEYWORD
                or p.name == "draft_answer"
                for p in params
            )
        except (TypeError, ValueError):
            return True

    def run(self, question: str, student_state=None) -> AgentState:
        """Every request enters the same Observe → Decide → Act agent loop."""
        import time

        run_started = time.perf_counter()
        with debug.scope("StudyAgent", "RUN"):
            question = str(question or "").strip()
            debug.log("StudyAgent", f"QUESTION → {question}")
            state = AgentState(
                question=question,
                max_steps=self.max_steps,
                knowledge_graph=self.knowledge_graph,
                assessment_requested=TaskAnalyzer._is_explicit_assessment_request(question),
            )

            if student_state is not None:
                self.student_state = student_state
            elif self.student_state is None:
                self.student_state = state.student
            state.student = self.student_state
            state.goal = "解决用户当前问题，并在需要时获取足够可靠的证据。"
            state.search_sources = []
            state.search_sort_by = "relevance"

            analysis_started = time.perf_counter()
            try:
                analyzer = TaskAnalyzer(self.reasoner)
                state.task_analysis = analyzer.analyze(question, state.student)
                state.task_type = state.task_analysis.task_type
                state.domain = state.task_analysis.domain
                state.goal = state.task_analysis.goal or state.goal
                # The user's current request is authoritative for assessment permission.
                state.assessment_requested = TaskAnalyzer._is_explicit_assessment_request(question)
                state.metrics["task_analysis_ms"] = round(
                    (time.perf_counter() - analysis_started) * 1000, 2
                )
                state.metrics["route_fallback"] = False
            except Exception as exc:
                state.task_analysis = None
                state.plan = None
                state.metrics["task_analysis_ms"] = round(
                    (time.perf_counter() - analysis_started) * 1000, 2
                )
                state.metrics["route_fallback"] = True
                debug.log(
                    "StudyAgent",
                    f"TASK ANALYZER FAILED → continuing full agent loop: "
                    f"{type(exc).__name__}: {exc}",
                )

            state.student.sync_from_knowledge_graph(self.knowledge_graph)
            self.knowledge_graph.bootstrap_query_context(question)

            execution_started = time.perf_counter()
            executor = self.tool_executor
            if executor is None:
                # The Agent always needs a Harness boundary. An empty executor is
                # still safe: tool calls fail closed, while ANSWER remains possible.
                from .tool_loop import ToolExecutor
                executor = ToolExecutor()
            elif not hasattr(executor, "execute"):
                state.error = "Tool Harness 接口无效：缺少 execute 方法。"

            if state.error is None:
                try:
                    state = AgentToolLoop(self.reasoner, executor).run(state)
                    if state.pending_assessment:
                        self.pending_assessment_state = state
                        debug.log(
                            "StudyAgent",
                            "ASSESSMENT PENDING → preserved interactive state",
                        )
                    if (
                        state.final_answer is not None
                        and state.steps
                        and state.steps[-1].action == "ANSWER"
                    ):
                        self._teach_final_answer(state)
                except Exception as exc:
                    state.error = f"Agent Loop 执行失败：{type(exc).__name__}: {exc}"
                    debug.log("StudyAgent", state.error)

            state.metrics["execution_ms"] = round(
                (time.perf_counter() - execution_started) * 1000, 2
            )
            state.metrics["reasoner_steps"] = state.step_count
            state.metrics["tool_calls"] = sum(
                state.action_counts.get(action, 0)
                for action in ("SEARCH", "CALCULATE", "VERIFY", "ASSESS")
            )
            state.metrics["total_ms"] = round(
                (time.perf_counter() - run_started) * 1000, 2
            )
            state.metrics["agent_loop"] = True
            state.metrics["assessment_offer"] = bool(
                state.final_answer
                and not state.pending_assessment
                and state.task_type in {
                    "conceptual",
                    "explanation",
                    "math",
                    "coding",
                    "factual",
                    "comparison",
                    "research",
                    "troubleshooting",
                }
            )

            debug.log(
                "StudyAgent",
                f"EXECUTION FINISHED → agent_loop steps={state.step_count}",
            )

            if state.pending_assessment:
                debug.log(
                    "StudyAgent",
                    "RUN PAUSED → waiting for assessment answer",
                )
            elif state.final_answer is None and state.error:
                state.final_answer = f"Agent 未能完成任务。\n\n原因：{state.error}"
            elif state.final_answer is None:
                state.error = "Agent 在没有产生最终 ANSWER 的情况下结束。"
                state.final_answer = f"Agent 未能完成任务。\n\n原因：{state.error}"

            self._record_reasoning_efficiency(state)
            self._update_student_model(state)
            self._update_knowledge_graph(state)
            self.last_state = state
            debug.log(
                "StudyAgent",
                f"FINAL STATE → finished={state.finished}, "
                f"answer={'yes' if state.final_answer else 'no'}, "
                f"pending_assessment={'yes' if state.pending_assessment else 'no'}, "
                f"steps={state.step_count}, evidence={len(state.evidence)}, "
                f"claims={len(state.claims)}",
            )
            debug.log("StudyAgent", "STUDENT MODEL → updated")
            return state

    def start_assessment(self, concept: str, difficulty: str = "graduate") -> dict:
        """Start an explicit formal assessment session for one primary concept."""
        concept = str(concept or "").strip()
        if not concept:
            raise ValueError("正式测评需要指定主要知识点。")
        level, _score = normalize_difficulty(difficulty)
        self.student_state = self.student_state or StudentState()
        self.student_state.sync_from_knowledge_graph(self.knowledge_graph)
        context = self.knowledge_graph.context_for(concept)
        if self.assessment_generator is None:
            raise RuntimeError("当前 Reasoner 未配置可用于正式测评的模型路由。")
        assessment = self.assessment_generator.generate(
            concept,
            level,
            learner_context=context,
        )
        self.knowledge_graph.add_concept(assessment["primary_concept"])
        pending = dict(assessment)
        state = AgentState(
            question=assessment["question"],
            student=self.student_state,
            task_type="assessment",
            domain="assessment",
            execution_strategy="assessment",
            knowledge_graph=self.knowledge_graph,
        )
        state.pending_assessment = pending
        self.pending_assessment_state = state
        self.last_state = state
        debug.log(
            "StudyAgent",
            f"ASSESSMENT START → concept={assessment['primary_concept']!r}, difficulty={level}, question_type={assessment['question_type']}",
        )
        return {
            "status": "ASSESSMENT_READY",
            "question": assessment["question"],
            "primary_concept": assessment["primary_concept"],
            "supporting_concepts": assessment["supporting_concepts"],
            "difficulty_level": assessment["difficulty_level"],
            "question_type": assessment["question_type"],
        }

    def learner_state(self, concept: str | None = None) -> dict:
        """Return explicit learner evidence without changing it."""
        self.student_state = self.student_state or StudentState()
        self.student_state.sync_from_knowledge_graph(self.knowledge_graph)
        if concept is None or not str(concept).strip():
            return {
                "known_topics": sorted(self.student_state.known_topics),
                "weak_topics": sorted(self.student_state.weak_topics),
                "learning_topics": sorted(self.student_state.learning_topics),
            }
        key = self.knowledge_graph._id(str(concept).strip())
        node = self.knowledge_graph.nodes.get(key)
        if node is None:
            return {"concept": str(concept).strip(), "learning_stage": "unassessed"}
        return {"concept": node.name, "learner": node.learner.as_dict()}
    def submit_assessment_answer(self, answer: str, confidence: float | None = None) -> dict:
        """Evaluate the pending assessment and update the persistent learner graph."""
        state = self.pending_assessment_state or self.last_state
        if state is None or not state.pending_assessment:
            raise ValueError("当前没有待作答的测试题。")
        debug.log(
            "StudyAgent",
            "ASSESSMENT SUBMIT → evaluating pending answer",
        )
        answer = str(answer or "").strip()
        if not answer:
            raise ValueError("测试答案不能为空。")

        result = self.assessment_evaluator.evaluate(
            state.pending_assessment, answer, confidence=confidence
        )
        assessment = state.pending_assessment
        self.knowledge_graph.record_assessment(
            assessment["concepts"],
            result["correct"],
            result["confidence"],
            assessment["difficulty"],
            primary_concept=assessment.get("primary_concept"),
        )
        relations = assessment.get("relations", [])
        for relation in relations if isinstance(relations, list) else []:
            if isinstance(relation, (list, tuple)) and len(relation) == 3:
                self.knowledge_graph.update_relation_learner(
                    str(relation[0]), str(relation[1]), str(relation[2]),
                    result["correct"], result["confidence"], assessment["difficulty"]
                )
        result["learner_state"] = {
            concept: self.knowledge_graph.nodes[self.knowledge_graph._id(concept)].learner.as_dict()
            for concept in assessment["concepts"]
            if self.knowledge_graph._id(concept) in self.knowledge_graph.nodes
        }
        result["assessment"] = assessment
        state.student.sync_from_knowledge_graph(self.knowledge_graph)
        state.pending_assessment = None
        if self.pending_assessment_state is state:
            self.pending_assessment_state = None
        debug.log(
            "StudyAgent",
            "ASSESSMENT COMPLETE → learner graph updated and pending state cleared",
        )
        return result

    def _record_reasoning_efficiency(self, state) -> None:
        """Record how many successful Reasoner decisions each model needed for this task."""
        if state.pending_assessment:
            return
        difficulty = getattr(state.task_analysis, "difficulty", 3) or 3
        successful_answer = any(
            step.action == "ANSWER" and step.success
            for step in state.steps
        ) and state.final_answer is not None
        model_steps: dict[str, int] = {}
        for step in state.steps:
            if not step.model:
                continue
            model_steps[step.model] = model_steps.get(step.model, 0) + 1
        for model_name, steps in model_steps.items():
            try:
                self.reasoner.model_router.registry.record_task_outcome(
                    model_name,
                    "reasoning",
                    difficulty,
                    steps,
                    successful_answer,
                )
            except Exception as exc:
                debug.log(
                    "StudyAgent",
                    f"EFFICIENCY RECORD FAILED → model={model_name}: {type(exc).__name__}",
                )

    def _update_student_model(self, state) -> None:
        # Ordinary chat must not touch the learner graph. Knowledge-path updates
        # are synchronized from explicit assessment evidence only.
        if state.knowledge_graph is not None:
            state.student.sync_from_knowledge_graph(state.knowledge_graph)

    def record_assessment(
        self,
        concepts,
        correct: bool,
        confidence: float | None = None,
        relation=None,
        difficulty="graduate",
        primary_concept: str | None = None,
    ) -> None:
        """Update the persistent learner graph from an explicit quiz/exercise result."""
        if not isinstance(correct, bool):
            raise ValueError("assessment correct 必须是布尔值。")
        if not isinstance(concepts, (list, tuple)):
            concepts = [concepts]
        self.knowledge_graph.record_assessment(
            [str(x) for x in concepts if str(x).strip()],
            correct,
            confidence,
            normalize_difficulty(difficulty)[1],
            primary_concept=primary_concept,
        )
        if isinstance(relation, (list, tuple)) and len(relation) == 3:
            source, target, relation_type = (str(x).strip() for x in relation)
            from .knowledge_graph import RELATIONS
            if source and target and source != target and relation_type in RELATIONS:
                self.knowledge_graph.update_relation_learner(
                    source, target, relation_type, correct, confidence,
                    normalize_difficulty(difficulty)[1]
                )
                debug.log(
                    "StudyAgent",
                    f"ASSESS RELATION → {source} -[{relation_type}]-> {target}",
                )
        elif relation is not None:
            debug.log("StudyAgent", "ASSESS RELATION → ignored malformed relation input")

        if self.student_state is None:
            self.student_state = StudentState()
        self.student_state.sync_from_knowledge_graph(self.knowledge_graph)

    def _update_knowledge_graph(self, state) -> None:
        """Record explicit assessment signals; ordinary exposure is not treated as mastery."""
        graph = state.knowledge_graph
        if graph is None: return
        # TaskAnalyzer.domain is a scope label, not necessarily a learner concept.
        # Do not pollute the concept graph with coarse labels.
        if state.domain:
            debug.log("StudyAgent", f"KNOWLEDGE GRAPH SCOPE → {state.domain}")

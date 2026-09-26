from .state import AgentState
from .tool_loop import AgentToolLoop
from .task_analyzer import TaskAnalyzer
from .knowledge_graph import KnowledgeGraph, normalize_difficulty
from .assessment import AssessmentEvaluator
from .__debug__ import debug


class StudyAgent:
    """LLM-driven Study Agent: Harness 初始化状态，Reasoner 决策，Teacher 负责最终教学表达。"""

    def __init__(self, reasoner, teacher=None, tool_executor=None, max_steps=15):
        self.reasoner = reasoner
        self.teacher = teacher
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.student_state = None
        self.knowledge_graph = KnowledgeGraph()
        self.assessment_evaluator = AssessmentEvaluator()
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
        with debug.scope("StudyAgent", "RUN"):
            debug.log("StudyAgent", f"QUESTION → {question}")
            state = AgentState(question=question, max_steps=self.max_steps, knowledge_graph=self.knowledge_graph)
            if student_state is not None:
                self.student_state = student_state
            elif self.student_state is None:
                self.student_state = state.student
            state.student = self.student_state
            # Restore persistent learner evidence before TaskAnalyzer sees the student.
            state.student.sync_from_knowledge_graph(self.knowledge_graph)

            state.goal = "解决用户当前问题，并在需要时获取足够可靠的证据。"
            state.search_sources = []
            state.search_sort_by = "relevance"

            if self.tool_executor is not None and hasattr(self.tool_executor, "execute"):
                try:
                    analyzer = TaskAnalyzer(self.reasoner)
                    state.task_analysis = analyzer.analyze(question, state.student)
                    state.task_type = state.task_analysis.task_type
                    state.domain = state.task_analysis.domain
                    state.goal = state.task_analysis.goal or state.goal
                except Exception as exc:
                    state.task_analysis = None
                    state.plan = None
                    debug.log("StudyAgent", f"TASK ANALYZER FAILED → fallback to Reasoner: {type(exc).__name__}: {exc}")

            if self.tool_executor is None:
                state.error = "Tool Harness 尚未配置。"
            elif not hasattr(self.tool_executor, "execute"):
                state.error = "Tool Harness 接口无效：缺少 execute 方法。"
            else:
                try:
                    state = AgentToolLoop(self.reasoner, self.tool_executor).run(state)
                    if state.pending_assessment:
                        self.pending_assessment_state = state
                        debug.log(
                            "StudyAgent",
                            "ASSESSMENT PENDING → preserved interactive state",
                        )
                    if state.final_answer is not None and state.steps and state.steps[-1].action == "ANSWER":
                        self._teach_final_answer(state)
                except Exception as exc:
                    state.error = f"Agent Loop 执行失败：{type(exc).__name__}: {exc}"
                    debug.log("StudyAgent", state.error)

            debug.log("StudyAgent", f"LOOP FINISHED → steps={state.step_count}")

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

            self._update_student_model(state)
            self._update_knowledge_graph(state)
            self.last_state = state
            debug.log(
                "StudyAgent",
                f"FINAL STATE → finished={state.finished}, answer={'yes' if state.final_answer else 'no'}, pending_assessment={'yes' if state.pending_assessment else 'no'}, steps={state.step_count}, evidence={len(state.evidence)}, claims={len(state.claims)}",
            )
            debug.log("StudyAgent", "STUDENT MODEL → updated")
            return state

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

    def _update_student_model(self, state) -> None:
        # Task exposure is not mastery. Learner-facing topic status comes from
        # explicit assessment evidence stored in the persistent knowledge graph.
        if state.knowledge_graph is not None:
            state.student.sync_from_knowledge_graph(state.knowledge_graph)

    def record_assessment(self, concepts, correct: bool, confidence: float | None = None, relation=None, difficulty="graduate") -> None:
        """Update the persistent learner graph from an explicit quiz/exercise result."""
        if not isinstance(correct, bool):
            raise ValueError("assessment correct 必须是布尔值。")
        if not isinstance(concepts, (list, tuple)):
            concepts = [concepts]
        self.knowledge_graph.record_assessment(
            [str(x) for x in concepts if str(x).strip()], correct, confidence, normalize_difficulty(difficulty)[1]
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

        if self.student_state is not None:
            self.student_state.sync_from_knowledge_graph(self.knowledge_graph)

    def _update_knowledge_graph(self, state) -> None:
        """Record explicit assessment signals; ordinary exposure is not treated as mastery."""
        graph = state.knowledge_graph
        if graph is None: return
        if state.domain: graph.add_concept(state.domain)

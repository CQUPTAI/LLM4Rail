import ast
import json
import logging
import os
import re
import warnings
from openai import OpenAI

from service.esr import ExperienceStore, extract_json_object, local_embedding
from service.modules.meal_service import MealService
from service.modules.ticket_query import TicketQuery
from service.modules.weather_query import WeatherQuery

logger = logging.getLogger(__name__)


class Agent:
    """Iterative QTAO agent with Experience Storage and Retrieval (ESR)."""

    def __init__(
        self,
        model,
        esr_enabled: bool = True,
        experience_path: str = "service/esr_experiences.json",
        retrieval_k: int = 3,
        max_iterations: int = 8,
        embedding_fn=None,
    ):
        if retrieval_k < 0:
            raise ValueError("retrieval_k must be non-negative")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self.model = model
        self.ticket_query = TicketQuery()
        self.weather_query = WeatherQuery()
        self.meal_service = MealService()
        self.tools = {
            self.meal_service.name: self.meal_service,
            self.ticket_query.name: self.ticket_query,
            self.weather_query.name: self.weather_query,
        }
        self.tool_descriptions = "\n".join(
            f"{i + 1}." + tool.description for i, tool in enumerate(self.tools.values())
        )
        self.tool_names = list(self.tools.keys())
        self.esr = ExperienceStore(experience_path) if esr_enabled else None
        self.retrieval_k = retrieval_k
        self.max_iterations = max_iterations
        self.embedding_fn = embedding_fn

    def query(self, user_info: dict, history: list, query: str, trajectory: dict) -> str:
        """Run one dialogue turn and let the LLM reflect on its full trajectory."""
        context_query = query
        retrieved = {"query_agnostic": [], "query_related": []}
        steps = []

        if self.esr is not None:
            try:
                context_query = self.summarize_context_query(user_info, history, query)
            except Exception as exc:
                logger.warning("Context query summarization failed; using raw query: %s", exc)
            try:
                retrieved = self.esr.retrieve(
                    context_query, top_k=self.retrieval_k, embed_fn=self.embed
                )
            except Exception as exc:
                logger.warning("ESR retrieval failed; continuing without experiences: %s", exc)

        trajectory["context_query"] = context_query
        trajectory["retrieved_experience_ids"] = [
            experience.id
            for experiences in retrieved.values()
            for experience in experiences
        ]
        trajectory.pop("esr_update", None)

        base_prompt = self._build_base_prompt(user_info)
        prompt = (
            f"Conversation history:\n{self._stringify(history)}\n\n"
            f"{base_prompt}\n"
            f"Question: {query}\n"
            "Retrieved experiences (query-agnostic experiences plus top-k "
            "query-related experiences):\n"
            f"{self.esr.format_for_prompt(retrieved) if self.esr is not None else 'ESR disabled'}\n"
            "Use experiences only when they agree with current tool observations. "
            "If an experience is outdated, ignore it and rely on current evidence.\n"
        )
        completion = ""
        reflection_ready = False

        try:
            for _ in range(self.max_iterations):
                new_completion = ""
                for chunk in self.invoke_llm(prompt + completion):
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        new_completion += delta.content

                answer_match = re.search(
                    r"^Answer:\s*(.*)", new_completion, re.MULTILINE | re.DOTALL
                )
                if answer_match:
                    answer = answer_match.group(1).strip()
                    reflection_ready = True
                    trajectory["trajectory"].append(new_completion)
                    trajectory["answer"] = answer
                    steps.append(
                        {
                            "thought": self._extract_thought(new_completion),
                            "answer": answer,
                        }
                    )
                    return answer

                trajectory["trajectory"].append({"completion": new_completion})
                action = self._extract_action(new_completion)
                action_input = self._extract_action_input(new_completion)
                if action not in self.tools:
                    raise ValueError(
                        f"Unknown tool {action!r}; expected one of {self.tool_names}"
                    )
                tool = self.tools[action]
                tool_input = self._parse_action_input(action_input)
                observation = tool(tool_input, user_info, history)
                observation_text = self._stringify(observation)

                completion += new_completion + f"\nObservation: {observation_text}\n"
                trajectory["trajectory"][-1]["action"] = action
                trajectory["trajectory"][-1]["action_input"] = tool_input
                trajectory["trajectory"][-1]["observation"] = observation_text
                steps.append(
                    {
                        "thought": self._extract_thought(new_completion),
                        "action": action,
                        "action_input": tool_input,
                        "observation": observation,
                    }
                )

            reflection_ready = True
            raise RuntimeError(
                f"QTAO did not produce an answer within {self.max_iterations} iterations."
            )
        finally:
            if self.esr is not None and not reflection_ready:
                trajectory["esr_update"] = {
                    "status": "skipped",
                    "reason": "incomplete_qtao_trajectory",
                }
            elif self.esr is not None:
                try:
                    # Do not pre-label success/failure or extract recovery pairs.
                    # Two independent LLM steps read the complete trajectory:
                    # first summarize additions, then detect outdated items.
                    result = self._update_experiences(
                        context_query, steps, retrieved
                    )
                    trajectory["esr_update"] = {"status": "updated", **result}
                except Exception as exc:
                    # A reflector/embedding outage must not discard a valid answer.
                    logger.exception("ESR update failed")
                    trajectory["esr_update"] = {
                        "status": "error",
                        "error": str(exc),
                    }

    def _build_base_prompt(self, user_info: dict) -> str:
        return f"""Answer the question below as best as you can. Use the conversation history, retrieved experiences, or the following tools:
Tools:
{self.tool_descriptions}
User information:
{self._stringify(user_info)}

Use the following format:
Question: the question you need to answer
Thought: thinking process about how to answer the question
Action: tool to use, which must be one of {self.tool_names}; if any required argument is missing, ask for it instead of taking that action
Action_Input: a JSON object containing the tool arguments, or None
Observation: result of the action
... (Thought/Action/Action_Input/Observation can repeat)
Thought: I now know the answer.
Answer: the final answer to the question

Begin!
"""

    def summarize_context_query(
        self, user_info: dict, history: list, query: str
    ) -> str:
        prompt = f"""Rewrite the current user query as one self-contained, context-aware query for experience retrieval.
Resolve references using the conversation history and relevant user information. Do not answer the query, add assumptions, or explain your reasoning. Output only the rewritten query.

User information: {self._stringify(user_info)}
Conversation history: {self._stringify(history)}
Current query: {query}
"""
        rewritten = self.invoke_llm_text(prompt).strip()
        return rewritten or query

    def _update_experiences(
        self,
        context_query: str,
        steps: list,
        retrieved: dict,
    ) -> dict:
        """Run new-experience summarization and outdated detection separately."""
        additions = self._summarize_new_experiences(context_query, steps)
        outdated = self._detect_outdated_experiences(
            context_query, steps, retrieved
        )
        update = {
            "query_agnostic": additions.get("query_agnostic", []),
            "query_related": additions.get("query_related", []),
            "outdated_query_agnostic_ids": outdated.get(
                "outdated_query_agnostic_ids", []
            ),
            "outdated_query_related_ids": outdated.get(
                "outdated_query_related_ids", []
            ),
        }
        return self.esr.apply_update(update, context_query, embed_fn=self.embed)

    def _summarize_new_experiences(self, context_query: str, steps: list) -> dict:
        """Step 1: extract only new experiences from the complete trajectory."""
        existing_agnostic = [
            experience.to_prompt_dict() for experience in self.esr.query_agnostic
        ]
        prompt = f"""You are performing STEP 1 of ESR reflection: NEW EXPERIENCE SUMMARIZATION.
Read the complete, unfiltered QTAO trajectory. Identify failures, diagnoses, corrections and recoveries yourself; the caller does not pre-label them. Extract only experiences supported by the trajectory. Do not detect or prune outdated experiences in this step.

STEP 1 OUTPUT SCHEMAS AND FIELD MEANINGS:
1. query_agnostic is an array of objects with:
   - tool (str): The exact name of the tool to which this general lesson applies, copied from the trajectory's Action. It identifies the tool whose usage pattern is being summarized.
   - failure (str): What concretely went wrong in the tool invocation. Ground it in the Action, Action_Input or Observation; describe the erroneous operation or result rather than merely writing "failed".
   - diagnosis (str): Why the failure happened. State the underlying cause inferred from the Observation/DOG report, such as an invalid argument, unsupported route or unsuitable value. This explains the cause and must not simply repeat failure.
   - strategy (str): The reusable, actionable correction or general usage rule. Explain what the agent should change or check in future calls to this tool. Keep it independent of any particular passenger, station, city or date.
   Complete schema: {{"tool": str, "failure": str, "diagnosis": str, "strategy": str}}.

2. query_related is an array of objects with:
   - query (str): The self-contained user need and context to which this experience applies. Use the supplied context-aware query, resolving pronouns and omitted context; it is the retrieval key for similar future situations.
   - failure (str): The concrete unsuccessful action, arguments or observed result in this specific query situation. Include enough detail to distinguish it from other failures.
   - diagnosis (str): The situation-specific root cause of that failure, based on the Observation/DOG report and the query context. Explain why the attempted action did not work here.
   - strategy (str): The effective, actionable correction for this or a closely similar query. Describe how to adjust the action or arguments so a future agent can reuse the solution; do not merely restate the diagnosis.
   Complete schema: {{"query": str, "failure": str, "diagnosis": str, "strategy": str}}.

For query_agnostic, extract only general tool-usage knowledge and do not duplicate existing experiences. For query_related, preserve situation-specific recovery knowledge. If no worthwhile new experience exists, return empty arrays.

Return exactly one JSON object and nothing else:
{{
  "query_agnostic": [],
  "query_related": []
}}
Do not output Markdown, reasoning or outdated-experience IDs.

Context-aware query: {context_query}
Full unfiltered QTAO trajectory: {self._stringify(steps)}
Existing query-agnostic experiences for duplicate avoidance: {self._stringify(existing_agnostic)}
"""
        return extract_json_object(self.invoke_llm_text(prompt))

    def _detect_outdated_experiences(
        self,
        context_query: str,
        steps: list,
        retrieved: dict,
    ) -> dict:
        """Step 2: detect only outdated or misleading retrieved experiences."""
        existing_agnostic = [
            experience.to_prompt_dict() for experience in self.esr.query_agnostic
        ]
        retrieved_related = [
            experience.to_prompt_dict()
            for experience in retrieved.get("query_related", [])
        ]
        prompt = f"""You are performing STEP 2 of ESR reflection: OUTDATED EXPERIENCE DETECTION.
Read the complete, unfiltered QTAO trajectory and the candidate experiences. Do not summarize or add new experiences in this step.

Select an experience for deletion only when the trajectory provides positive evidence that its failure, diagnosis or strategy is outdated, contradicted by current observations, or misled the current reasoning. Do not prune an experience merely because it was not used, and do not use age alone as a reason.
- outdated_query_agnostic_ids: IDs selected from the supplied existing query-agnostic candidates.
- outdated_query_related_ids: IDs selected only from the supplied retrieved query-related candidates.
Return only IDs that appear in the corresponding candidate lists.

Return exactly one JSON object and nothing else:
{{
  "outdated_query_agnostic_ids": [],
  "outdated_query_related_ids": []
}}
Use empty arrays when no experience is proven outdated. Do not output Markdown, reasoning or new experiences.

Context-aware query: {context_query}
Full unfiltered QTAO trajectory: {self._stringify(steps)}
Existing query-agnostic candidates: {self._stringify(existing_agnostic)}
Retrieved query-related candidates: {self._stringify(retrieved_related)}
"""
        result = extract_json_object(self.invoke_llm_text(prompt))
        allowed = {
            "outdated_query_agnostic_ids": {
                experience["id"] for experience in existing_agnostic
            },
            "outdated_query_related_ids": {
                experience["id"] for experience in retrieved_related
            },
        }
        for key, allowed_ids in allowed.items():
            value = result.get(key, []) or []
            if not isinstance(value, list):
                raise ValueError(f"{key} must be a JSON array")
            # Enforce candidate scope in code rather than relying on the LLM.
            result[key] = [str(item) for item in value if str(item) in allowed_ids]
        return result

    @staticmethod
    def _extract_thought(completion: str) -> str:
        match = re.search(
            r"Thought:\s*(.*?)(?=\n(?:Action|Action_Input|Answer):|\Z)",
            completion,
            re.DOTALL,
        )
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_action(completion: str) -> str:
        match = re.search(r"^Action:\s*(.+?)\s*$", completion, re.MULTILINE)
        if not match:
            raise ValueError("The model output does not contain a valid Action.")
        return match.group(1).strip().strip("`'")

    @staticmethod
    def _extract_action_input(completion: str) -> str:
        match = re.search(
            r"Action_Input:\s*(.*?)(?=\n(?:Thought|Action|Observation|Answer):|\Z)",
            completion,
            re.DOTALL,
        )
        if not match or not match.group(1).strip():
            raise ValueError("The model output does not contain Action_Input.")
        return match.group(1).strip()

    @staticmethod
    def _parse_action_input(action_input: str):
        cleaned = action_input.strip()
        cleaned = re.sub(r"^```(?:json|python)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Safe support for the single-quoted Python dictionaries used by
            # the original prompts; unlike eval, this executes no code.
            return ast.literal_eval(cleaned)

    @staticmethod
    def _stringify(value) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

    def _client_and_model(self):
        if self.model == "qwen-3":
            return (
                OpenAI(
                    api_key=os.environ.get("QwenKey", ""),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                "qwen3-235b-a22b",
            )
        if self.model == "gpt-4o":
            return (
                OpenAI(
                    api_key=os.environ.get("GPTKey", ""),
                    base_url="https://api.openai.com/v1",
                ),
                "gpt-4o",
            )
        raise ValueError("Invalid model name!")

    def embed(self, text: str) -> list[float]:
        if self.embedding_fn is not None:
            return self.embedding_fn(text)
        try:
            client, _ = self._client_and_model()
            embedding_model = (
                "text-embedding-v4" if self.model == "qwen-3" else "text-embedding-3-small"
            )
            response = client.embeddings.create(model=embedding_model, input=text)
            return [float(value) for value in response.data[0].embedding]
        except Exception as exc:
            warnings.warn(
                f"Embedding API failed; using deterministic local fallback: {exc}",
                RuntimeWarning,
            )
            self.embedding_fn = local_embedding
            return self.embedding_fn(text)

    def invoke_llm_text(self, prompt: str) -> str:
        client, model = self._client_and_model()
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if self.model == "qwen-3":
            params["extra_body"] = {"enable_thinking": False}
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content or ""

    def invoke_llm(self, prompt: str):
        client, model = self._client_and_model()
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stop": "Observation",
            "stream": True,
        }
        if self.model == "qwen-3":
            params["extra_body"] = {"enable_thinking": False}
        return client.chat.completions.create(**params)

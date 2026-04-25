"""ODSE GRPO fine-tuning entrypoint."""

import asyncio
import os
import textwrap
from datetime import datetime
from typing import Any, List, Optional, Sequence

from datasets import Dataset
from odse import OdseAction, OdseEnv
from odse.graders import EasyGrader, MediumGrader, HardGrader
import torch
from transformers import AutoConfig, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://simeetnayan-odse.hf.space")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-Coder-3B"
FALLBACK_MODEL_NAME = os.getenv("FALLBACK_MODEL_NAME") or "Qwen/Qwen2.5-1.5B-Instruct"
OOM_FALLBACK_MODEL_NAME = os.getenv("OOM_FALLBACK_MODEL_NAME") or "Qwen/Qwen2.5-1.5B-Instruct"
BENCHMARK = os.getenv("BENCHMARK", "odse")
TEMPERATURE = 0.7
MAX_TOKENS = 1024
SUCCESS_SCORE_THRESHOLD = 0.5
DEFAULT_OUTPUT_ROOT = "/data/outputs" if os.path.isdir("/data") else "outputs"
OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    f"{DEFAULT_OUTPUT_ROOT}/odse-grpo-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
)
HF_REPO_ID = os.getenv("HF_REPO_ID")
HF_COMMIT_MESSAGE = os.getenv("HF_COMMIT_MESSAGE", "Add ODSE GRPO post-trained checkpoint")
PUSH_TO_HUB = os.getenv("PUSH_TO_HUB", "0").lower() in {"1", "true", "yes"}
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "1"))
DATASET_SIZE = int(os.getenv("DATASET_SIZE", "24"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "1"))
EFFECTIVE_NUM_GENERATIONS = max(2, NUM_GENERATIONS)
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-6"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.0"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "16"))
PER_DEVICE_BATCH_SIZE = int(os.getenv("PER_DEVICE_BATCH_SIZE", "1"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "20"))
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "1"))
MAX_COMPLETION_LENGTH = int(os.getenv("MAX_COMPLETION_LENGTH", "256"))
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "512"))
GRADIENT_CHECKPOINTING = os.getenv("GRADIENT_CHECKPOINTING", "1").strip().lower() in {"1", "true", "yes"}
USE_BF16 = os.getenv("BF16", "1").strip().lower() in {"1", "true", "yes"}
USE_FP16 = os.getenv("FP16", "0").strip().lower() in {"1", "true", "yes"}
PYTORCH_CUDA_ALLOC_CONF = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", PYTORCH_CUDA_ALLOC_CONF)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert data scientist agent interacting with a Python execution sandbox.
    Your goal is to solve the given data science task, maximize the evaluation metric, and finally submit your predictions.

    You can take two types of actions:
    1. Execute Python code: Write your code inside ```python ... ``` blocks.
       The sandbox persists variables across steps. Pre-loaded variables include `train_df`, `val_features`, `test_features`, and `target_column`.
       NOTE: This is a headless Python script, NOT an interactive Jupyter notebook. You MUST use `print()` if you want to see the output of any variable or dataframe.
       Use `evaluate(predictions)` to check your validation score.
       Assign your test-set predictions to the variable `predictions` (matching test_features length) before submitting.

    2. Submit: When your `predictions` variable is ready, output exactly the word [SUBMIT] to finish the task.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def _apply_chat_template(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> str:
    kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": False,
    }
    # Newer chat tokenizers accept `enable_thinking`; older ones don't.
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _extract_rollout_text(tokenizer: AutoTokenizer, rollout_output: dict[str, Any]) -> str:
    text = rollout_output.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    completion_ids = rollout_output.get("completion_ids", [])
    if completion_ids:
        return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return ""


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def build_user_prompt(step: int, obs: Any, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Task Description: {obs.task_description}
        Latest Execution Status: {obs.execution_status}
        Stdout: {obs.stdout}
        Stderr: {obs.stderr}
        Validation Score: {obs.validation_score}
        Best Validation Score: {obs.best_validation_score}
        Last step reward: {last_reward:.2f}
        
        Previous steps summary:
        {history_block}
        
        Decide your next action. Provide Python code in a ```python ... ``` block to explore or train, or output [SUBMIT] if your `predictions` variable is ready.
        """
    ).strip()


def get_model_message(
    trainer: GRPOTrainer,
    tokenizer: AutoTokenizer,
    step: int,
    obs: Any,
    last_reward: float,
    history: List[str],
) -> str:
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = _apply_chat_template(tokenizer, messages)
        rollout_output = generate_rollout_completions(trainer, [prompt_text])[0]
        text = _extract_rollout_text(tokenizer, rollout_output)
        return text if text else "[SUBMIT]"
    except Exception:
        return "[SUBMIT]"


def parse_action(text: str) -> OdseAction:
    if "[SUBMIT]" in text.upper():
        return OdseAction(action_type="submit")
    import re
    code = text or ""
    code = re.sub(r"```(?:python|py)?", "", code, flags=re.IGNORECASE)
    code = re.sub(r"```", "", code)
    code = code.strip()
            
    if not code.strip():
        code = "# No code provided by model"
        
    return OdseAction(action_type="run_code", code=code)


def _task_config(task_id: str) -> tuple[Any, str, float, int]:
    grader = EasyGrader()  # Default grader
    difficulty = "easy"
    success_score_threshold = 0.5
    max_steps = 5

    if task_id == "task_easy":
        grader = EasyGrader()
        difficulty = "easy"
        success_score_threshold = 0.5
        max_steps = 10
    elif task_id == "task_medium":
        grader = MediumGrader()
        difficulty = "medium"
        success_score_threshold = 0.75
        max_steps = 15
    elif task_id == "task_hard":
        grader = HardGrader()
        difficulty = "hard"
        success_score_threshold = 0.9
        max_steps = 20
    else:
        raise ValueError(f"Unknown task name: {task_id}")

    return grader, difficulty, success_score_threshold, max_steps


async def run_individual_task(
    trainer: GRPOTrainer,
    tokenizer: AutoTokenizer,
    task_id: str,
    model_name: str,
) -> None:
    grader, difficulty, success_threshold, max_steps = _task_config(task_id)
    
    try:
        
        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        score = 0.01
        success = False

        log_start(task=task_id, env=BENCHMARK, model=model_name)
        async with OdseEnv(base_url=ENV_BASE_URL) as env:
            result = await env.reset(difficulty=difficulty, max_steps=max_steps)
            obs = result.observation
            last_reward = 0.0

            for step in range(1, max_steps + 1):
                if result.done:
                    break

                message = get_model_message(trainer, tokenizer, step, obs, last_reward, history)
                action = parse_action(message)

                result = await env.step(action)
                obs = result.observation

                reward = result.reward or 0.0
                done = result.done
                stderr = obs.stderr if obs.stderr else None
                stdout = obs.stdout if obs.stdout else None

                rewards.append(reward)
                steps_taken = step
                last_reward = reward
                error_during_code_execution = None
                if stderr and stderr.strip():
                    error_during_code_execution = stderr.strip().splitlines()[-1]

                # Summarize the action for the required logs without spamming stdout
                action_str = f"run_code({len(action.code or '')} bytes)" if action.action_type == "run_code" else "submit"
                log_step(step=step, action=action_str, reward=reward, done=done, error=error_during_code_execution)

                history.append(f"Step {step}: {action.action_type} -> reward {reward:+.2f} done={done}, stderr={stderr}, stdout={stdout}")

                if done:
                    break

            score = grader(obs)
                
            score = max(0.01, min(0.99, score))  # clamp to strict
            success = score >= success_threshold
            try:
                await env.close()
            except Exception as e:
                print(f"Error during closing env: {e}")
                pass
    except Exception as exc:
        print(f"Error during task execution: {exc}")
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    

async def collect_training_rollout(
    trainer: GRPOTrainer,
    tokenizer: AutoTokenizer,
    task_id: str,
) -> dict[str, Any]:
    grader, difficulty, _success_threshold, max_steps = _task_config(task_id)
    async with OdseEnv(base_url=ENV_BASE_URL) as env:
        result = await env.reset(difficulty=difficulty, max_steps=max_steps)
        obs = result.observation
        last_reward = 0.0
        history: List[str] = []

        prompt_ids: List[int] = []
        completion_ids: List[int] = []
        logprobs: List[float] = []

        for step in range(1, max_steps + 1):
            if result.done:
                break

            user_prompt = build_user_prompt(step, obs, last_reward, history)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            prompt_text = _apply_chat_template(tokenizer, messages)
            rollout_output = generate_rollout_completions(trainer, [prompt_text])[0]
            prompt_ids.extend(_to_list(rollout_output.get("prompt_ids")))
            completion_ids.extend(_to_list(rollout_output.get("completion_ids")))
            logprobs.extend(_to_list(rollout_output.get("logprobs")))

            action = parse_action(_extract_rollout_text(tokenizer, rollout_output))
            result = await env.step(action)
            obs = result.observation
            last_reward = result.reward or 0.0

            stderr = obs.stderr if obs.stderr else None
            stdout = obs.stdout if obs.stdout else None
            history.append(
                f"Step {step}: {action.action_type} -> reward {last_reward:+.2f} "
                f"done={result.done}, stderr={stderr}, stdout={stdout}"
            )

        task_score = float(grader(obs))
        try:
            await env.close()
        except Exception:
            pass

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "task_score": max(0.01, min(0.99, task_score)),
    }


def reward_task_score(completions: list[str], **kwargs) -> list[float]:
    rewards = kwargs.get("task_score") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    if isinstance(rewards, (int, float)):
        return [float(rewards) for _ in completions]
    if isinstance(rewards, Sequence):
        return [float(r) for r in rewards]
    return [0.0 for _ in completions]


def _run_coro_sync(coro: Any) -> Any:
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "asyncio.run() cannot be called from a running event loop" not in str(exc):
            raise
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def build_trainer(
    tokenizer: AutoTokenizer,
    model_name: str,
) -> GRPOTrainer:
    task_prompts = (["task_easy", "task_medium", "task_hard"] * max(1, DATASET_SIZE // 3 + 1))[:DATASET_SIZE]
    train_dataset = Dataset.from_dict({"prompt": task_prompts})

    grpo_kwargs: dict[str, Any] = dict(
        output_dir=OUTPUT_DIR,
        hub_model_id=HF_REPO_ID,
        hub_token=API_KEY,
        push_to_hub=PUSH_TO_HUB,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        num_generations=EFFECTIVE_NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=10,
        temperature=TEMPERATURE,
    )
    grpo_fields = getattr(GRPOConfig, "__dataclass_fields__", {})
    if "gradient_checkpointing" in grpo_fields:
        grpo_kwargs["gradient_checkpointing"] = GRADIENT_CHECKPOINTING
    if "max_prompt_length" in grpo_fields:
        grpo_kwargs["max_prompt_length"] = MAX_PROMPT_LENGTH
    if "bf16" in grpo_fields and USE_BF16 and torch.cuda.is_available():
        grpo_kwargs["bf16"] = True
    if "fp16" in grpo_fields and USE_FP16 and torch.cuda.is_available():
        grpo_kwargs["fp16"] = True

    grpo_config = GRPOConfig(**grpo_kwargs)

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        episode_prompt_ids: List[List[int]] = []
        episode_completion_ids: List[List[int]] = []
        episode_logprobs: List[List[float]] = []
        episode_scores: List[float] = []

        for prompt_task in prompts:
            episode = _run_coro_sync(collect_training_rollout(trainer, tokenizer, prompt_task))
            episode_prompt_ids.append(episode["prompt_ids"])
            episode_completion_ids.append(episode["completion_ids"])
            episode_logprobs.append(episode["logprobs"])
            episode_scores.append(episode["task_score"])

        return {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "logprobs": episode_logprobs,
            "task_score": episode_scores,
        }

    return GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        reward_funcs=[reward_task_score],
        train_dataset=train_dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )


def resolve_model_name() -> str:
    try:
        AutoConfig.from_pretrained(MODEL_NAME, token=API_KEY)
        return MODEL_NAME
    except Exception as exc:
        msg = str(exc)
        unknown_arch = "does not recognize this architecture" in msg or "model type" in msg
        if not unknown_arch:
            raise
        print(
            f"Model `{MODEL_NAME}` is not supported by current Transformers build; "
            f"falling back to `{FALLBACK_MODEL_NAME}`.",
            flush=True,
        )
        AutoConfig.from_pretrained(FALLBACK_MODEL_NAME, token=API_KEY)
        return FALLBACK_MODEL_NAME


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "CUDA out of memory" in msg
        or "OutOfMemoryError" in msg
        or "CUDNN_STATUS_NOT_SUPPORTED" in msg
    )


def _cleanup_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=API_KEY)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main() -> None:
    if NUM_GENERATIONS < 2:
        print(
            f"NUM_GENERATIONS={NUM_GENERATIONS} is invalid for GRPO; using {EFFECTIVE_NUM_GENERATIONS}.",
            flush=True,
        )
    resolved_model_name = resolve_model_name()
    print(f"Using model: {resolved_model_name}", flush=True)
    trainer_model_name = resolved_model_name
    tokenizer = _load_tokenizer(trainer_model_name)
    trainer = None

    # TRL-only fallback order: primary model, then smaller OOM fallback model.
    build_attempts: List[tuple[str, str]] = [
        (resolved_model_name, "primary model"),
        (OOM_FALLBACK_MODEL_NAME, "smaller OOM fallback model"),
    ]

    last_exc: Optional[Exception] = None
    for candidate_model, label in build_attempts:
        try:
            print(f"Initializing trainer: model={candidate_model}, strategy={label}", flush=True)
            _cleanup_cuda_memory()
            tokenizer = _load_tokenizer(candidate_model)
            trainer = build_trainer(tokenizer, candidate_model)
            trainer_model_name = candidate_model
            break
        except Exception as exc:
            last_exc = exc
            if _is_oom_error(exc):
                print(
                    f"OOM while initializing `{candidate_model}` with strategy `{label}`; trying safer fallback.",
                    flush=True,
                )
                _cleanup_cuda_memory()
                continue
            raise

    if trainer is None:
        raise RuntimeError(f"Unable to initialize trainer after fallback attempts: {last_exc}")

    print("Starting GRPO training on ODSE tasks...", flush=True)
    try:
        trainer.train()
    except Exception as exc:
        if _is_oom_error(exc):
            print(
                "Training failed due to memory limits.",
                flush=True,
            )
            if trainer_model_name != OOM_FALLBACK_MODEL_NAME:
                print(f"Retrying training with smaller fallback model: {OOM_FALLBACK_MODEL_NAME}", flush=True)
                _cleanup_cuda_memory()
                trainer_model_name = OOM_FALLBACK_MODEL_NAME
                tokenizer = _load_tokenizer(trainer_model_name)
                trainer = build_trainer(tokenizer, trainer_model_name)
                trainer.train()
            else:
                raise
        else:
            raise
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    if PUSH_TO_HUB:
        if not API_KEY:
            raise ValueError("PUSH_TO_HUB is enabled but HF_TOKEN/API_KEY is not set.")
        if not HF_REPO_ID:
            raise ValueError("PUSH_TO_HUB is enabled but HF_REPO_ID is not set.")
        print(f"Pushing checkpoint to Hugging Face Hub: {HF_REPO_ID}", flush=True)
        trainer.push_to_hub(commit_message=HF_COMMIT_MESSAGE)
        tokenizer.push_to_hub(HF_REPO_ID, token=API_KEY, commit_message=HF_COMMIT_MESSAGE)

    tasks = ["task_easy", "task_medium", "task_hard"]

    for task in tasks:
        try:
            _run_coro_sync(run_individual_task(trainer, tokenizer, task, trainer_model_name))
        except Exception as e:
            print(f"Error during task execution: {e}")
    

if __name__ == "__main__":
    main()
"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_id> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import os
import textwrap
from datetime import datetime
from typing import Any, List, Optional

from datasets import Dataset
from odse import OdseAction, OdseEnv
from odse.graders import EasyGrader, MediumGrader, HardGrader
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://simeetnayan-odse.hf.space")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen3.5-4B"
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
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-6"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.0"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "16"))
PER_DEVICE_BATCH_SIZE = int(os.getenv("PER_DEVICE_BATCH_SIZE", "1"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "20"))
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "1"))
MAX_COMPLETION_LENGTH = int(os.getenv("MAX_COMPLETION_LENGTH", "256"))
VLLM_MODE = os.getenv("VLLM_MODE", "colocate")
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://localhost:8000")


def print_hf_space_guide() -> None:
    lines = [
        "HF Space setup (for post-training and Hub push):",
        "1) Create a GPU Space (Docker or Gradio), then add secrets HF_TOKEN and optionally HF_REPO_ID.",
        "2) Keep ENV_BASE_URL pointing to your ODSE env Space URL.",
        "3) In the Space shell, install deps and run this script:",
        "   pip install -U transformers trl datasets odse",
        "   PUSH_TO_HUB=1 HF_REPO_ID=<username>/<repo> python grpo.py",
        "4) Artifacts are written to OUTPUT_DIR and pushed to HF_REPO_ID when training completes.",
    ]
    print("\n".join(lines), flush=True)

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
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


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
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        rollout_output = generate_rollout_completions(trainer, [prompt_text])[0]
        text = rollout_output.get("text")
        if not text:
            completion_ids = rollout_output.get("completion_ids", [])
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        text = (text or "").strip()
        return text if text else "[SUBMIT]"
    except Exception:
        return "[SUBMIT]"


def parse_action(text: str) -> OdseAction:
    if "[SUBMIT]" in text.upper():
        return OdseAction(action_type="submit")
    import re
    code = text
    code = re.sub("```python", "", code)
    code = re.sub("```", "", code)
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


async def run_individual_task(trainer: GRPOTrainer, tokenizer: AutoTokenizer, task_id: str) -> None:
    grader, difficulty, success_threshold, max_steps = _task_config(task_id)
    
    try:
        
        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        score = 0.01
        success = False

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
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
            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            rollout_output = generate_rollout_completions(trainer, [prompt_text])[0]
            prompt_ids.extend(rollout_output.get("prompt_ids", []))
            completion_ids.extend(rollout_output.get("completion_ids", []))
            logprobs.extend(rollout_output.get("logprobs", []))

            model_text = rollout_output.get("text")
            if not model_text:
                model_text = tokenizer.decode(rollout_output.get("completion_ids", []), skip_special_tokens=True)
            action = parse_action((model_text or "").strip())
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
    return [float(r) for r in rewards]


def build_trainer(tokenizer: AutoTokenizer) -> GRPOTrainer:
    task_prompts = (["task_easy", "task_medium", "task_hard"] * max(1, DATASET_SIZE // 3 + 1))[:DATASET_SIZE]
    train_dataset = Dataset.from_dict({"prompt": task_prompts})

    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode=VLLM_MODE,
        vllm_server_base_url=VLLM_SERVER_URL if VLLM_MODE == "server" else None,
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
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=10,
        temperature=TEMPERATURE,
    )

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        episode_prompt_ids: List[List[int]] = []
        episode_completion_ids: List[List[int]] = []
        episode_logprobs: List[List[float]] = []
        episode_scores: List[float] = []

        for prompt_task in prompts:
            episode = asyncio.run(collect_training_rollout(trainer, tokenizer, prompt_task))
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
        model=MODEL_NAME,
        processing_class=tokenizer,
        reward_funcs=[reward_task_score],
        train_dataset=train_dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )


def main() -> None:
    print_hf_space_guide()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    trainer = build_trainer(tokenizer)

    print("Starting GRPO training on ODSE tasks...", flush=True)
    trainer.train()
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
            asyncio.run(run_individual_task(trainer, tokenizer, task))
        except Exception as e:
            print(f"Error during task execution: {e}")
    

if __name__ == "__main__":
    main()
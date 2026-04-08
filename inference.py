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

    [START] task=<task_name> env=<benchmark> model=<model_name>
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
from typing import Any, List, Optional

from openai import OpenAI

from odse import OdseAction, OdseEnv
from odse.graders import EasyGrader, MediumGrader, HardGrader

IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_NAME", "task_easy")
BENCHMARK = os.getenv("BENCHMARK", "odse")
TEMPERATURE = 0.7
MAX_TOKENS = 1024
SUCCESS_SCORE_THRESHOLD = 0.5 #A model with random predictions should score around 0.1-0.5, so 0.5 is a reasonable threshold for success in this benchmark.

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


def get_model_message(client: OpenAI, step: int, obs: Any, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "[SUBMIT]"
    except Exception as exc:
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


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    grader = EasyGrader()  # Default grader
    difficulty = "easy"
    if TASK_NAME == "task_easy":
        grader = EasyGrader()
        difficulty = "easy"
        SUCCESS_SCORE_THRESHOLD = 0.5
    elif TASK_NAME == "task_medium":
        grader = MediumGrader()
        difficulty = "medium"
        SUCCESS_SCORE_THRESHOLD = 0.75
    elif TASK_NAME == "task_hard":
        grader = HardGrader()
        difficulty = "hard"
        SUCCESS_SCORE_THRESHOLD = 0.9
    else:
        raise ValueError(f"Unknown task name: {TASK_NAME}")
    


    async with OdseEnv(base_url="https://simeetnayan-odse.hf.space") as env:
        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

        try:
            result = await env.reset(difficulty=difficulty)
            obs = result.observation
            last_reward = 0.0

            # Safely loop up to the environment's max steps limit
            max_steps = obs.max_steps if hasattr(obs, "max_steps") and obs.max_steps else 50
            for step in range(1, max_steps + 1):
                if result.done:
                    break

                message = get_model_message(client, step, obs, last_reward, history)
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

            score =  grader.grade(obs)
                
            score = max(0.01, min(0.99, score))  # clamp to strict
            success = score >= SUCCESS_SCORE_THRESHOLD

        finally:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
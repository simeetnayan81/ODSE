
Real-world task simulation
The environment must simulate a task humans actually do. Not games, not toys. Examples: email triage, code review, data cleaning, scheduling, customer support, content moderation.
OpenEnv spec compliance
Implement the full OpenEnv interface: typed Observation, Action, and Reward Pydantic models. step(action) → returns observation, reward, done, info. reset() → returns initial observation. state() → returns current state. openenv.yaml with metadata. Tested via openenv validate.
Minimum 3 tasks with agent graders
Each task defines a concrete objective an agent must accomplish, with a programmatic grader that scores performance (0.0–1.0). Tasks should range: easy → medium → hard. Graders must have clear, deterministic success/failure criteria.
Meaningful reward function
Provides signal over the full trajectory (not just binary end-of-episode). Rewards partial progress toward task completion. Penalizes clearly undesirable behavior (e.g. infinite loops, destructive actions).
Baseline inference script
Uses the OpenAI API client to run a model against the environment. Reads API credentials from environment variables (OPENAI_API_KEY). Produces a reproducible baseline score on all 3 tasks.
Detailed Requirements
NON-FUNCTIONAL REQUIREMENTS
Deploys to a Hugging Face Space
Environment must run as a containerized HF Space tagged with openenv.
Containerized execution
Must include a working Dockerfile. The environment should start cleanly with docker build + docker run.
Documentation
README must include: environment description and motivation, action and observation space definitions, task descriptions with expected difficulty, setup and usage instructions, baseline scores.


HF Space deploys
Automated ping to the Space URL — must return 200 and respond to reset()
OpenEnv spec compliance
Validate openenv.yaml, typed models, step()/reset()/state() endpoints
Dockerfile builds
Automated docker build on the submitted repo
Baseline reproduces
Run the submitted inference script — must complete without error and produce scores
3+ tasks with graders
Enumerate tasks, run each grader, verify scores/reward in 0.0–1.0 range
Mandatory Additional Instructions
Before submitting, ensure the following variables are defined in your environment configuration:
API_BASE_URL   The API endpoint for the LLM.
MODEL_NAME     The model identifier to use for inference.
HF_TOKEN       Your Hugging Face / API key.
The inference script must be named `inference.py` and placed in the root directory of the project
Participants must use OpenAI Client for all LLM calls using above variables
Participants must emit structured stdout logs strictly following the [START], [STEP], and [END] format defined in the sample inference.py provided below. Any deviation in field names, ordering, or formatting will result in incorrect evaluation scoring. Refer to the Sample Inference Script for the complete format specification and examples.
Infra Restrictions
Runtime of inference script should be less than 20min 
Make sure your env and inference can run on a machine with vcpu=2, memory=8gb
Validator
Run the pre-submission validation script before submitting
# Model Evaluation using Gemini (GPT-Eval)

This document outlines the process for evaluating a locally trained Gemma model using the Gemini API, inspired by the GPT-Eval methodology described in the paper "[TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/pdf/2305.07759)".

## Overview

The `evaluate_model.py` script automates the evaluation of your model's story-generation capabilities. It works as follows:

1.  **Instruction & Prompt Generation**: The script begins by using the Gemini API to generate a set of evaluation scenarios. By default, it generates 50 scenarios. Each scenario consists of a random instruction and a compatible story beginning.
2.  **Story Completion**: For each scenario, the script feeds the instruction and the story beginning to your trained model, which then generates multiple story completions (10 by default).
3.  **LLM-based Evaluation**: Each generated completion is sent to the Gemini API (`gemini-2.5-flash`) for evaluation against the initial instruction and story beginning.
4.  **Scoring**: Gemini grades the completion based on five key dimensions.
5.  **Summarization**: The script calculates the average score for each dimension across all completions for a given prompt, and then provides an overall average for the entire evaluation run.

## Evaluation Dimensions

The evaluation focuses on the following five criteria:

-   **Grammar (Score: 1-10)**: Assesses the grammatical correctness of the generated text.
-   **Creativity (Score: 1-10)**: Measures the originality, imagination, and novelty of the story's completion.
-   **Consistency (Score: 1-10)**: Evaluates how well the completion aligns with the characters, plot, and tone established in the initial prompt.
-   **Plot (Score: 1-10)**: Reflects the extent to which the generated plot is coherent and makes logical sense.
-   **Instruct (Score: 1-10)**: Measures how well the generated story adheres to the specific instruction provided in the prompt.

## Instruction Types

For each prompt, one of the following instruction types is chosen at random to test the model's ability to adapt:

1.  **Words to Include**: The story must contain a specific list of 3-4 simple words.
2.  **Sentence to Include**: The story must contain an exact sentence provided in the instruction.
3.  **Features to Include**: The story must exhibit 2-3 specific literary features, chosen from: `dialogue`, `bad ending`, `moral value`, `plot twist`, `foreshadowing`, `conflict`.
4.  **Summary to Follow**: The story's plot must adhere to a short 1-2 sentence summary.

## How to Run the Evaluation

### 1. Prerequisites

-   Ensure you have installed all the required Python packages from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
-   You must have a Google API key with the Gemini API enabled.

### 2. Set Environment Variable

You need to set your Google API key as an environment variable.

```bash
export GOOGLE_API_KEY='your_google_api_key_here'
```

## Usage

To run the evaluation, you need a trained model checkpoint (e.g., `best_model_params.pt`) and a Google Cloud API Key with access to Gemini (set as the `GOOGLE_API_KEY` environment variable).

**Basic Evaluation (Default):**

```bash
python evaluate_model.py --model-path ./models/your_model.pt
```

**Instruction-Based Evaluation:**

To generate a different number of prompts for the evaluation, use the `--num-prompts` argument.

```bash
python evaluate_model.py --model-path ./models/your_model.pt --num-prompts 20
```

### Arguments

-   `--model-path` (str): **Required.** Path to the saved model parameters (`.pt` file).
-   `--num-prompts` (int): The number of evaluation prompts (with instructions) to generate using the Gemini API. Default: `50`.
-   `--max-new_tokens` (int): Maximum number of new tokens to generate for each completion. Default: `200`.
-   `--temperature` (float): Controls the randomness of the generation. Lower values make it more deterministic. Default: `1.0`.
-   `--top-k` (int): Samples from the top K most likely tokens at each step. Default: `None`.

## Output

The script will first print the generation and evaluation progress. At the end, it will display two summary tables:

1.  **Evaluation Summary**: A table showing the average scores for all five dimensions for each individual prompt.
2.  **Overall Average Scores**: The final average scores for all dimensions across the entire set of prompts.
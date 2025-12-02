"""
"This script evaluates a trained Gemma model using the Gemini API for LLM-based evaluation.
It generates text completions for a set of prompts and then uses Gemini to score them
on grammar, creativity, and consistency, inspired by the GPT-Eval paper.
"""

import os
import re
import torch
import argparse
import tiktoken
import google.generativeai as genai
import pandas as pd
import random
from tqdm import tqdm
from gemma_scratch.model import Gemma3Model
from gemma_scratch.config import GEMMA3_CONFIG_CUSTOM

# --- Configuration ---
# Ensure you have your GEMINI_API_KEY set as an environment variable
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_MODEL = "gemini-2.5-flash"
GENERATIONS_PER_PROMPT = 10  # As per the paper

# --- Tokenizer ---
enc = tiktoken.get_encoding ("gpt2")

# --- Prompt Generation Templates ---

INSTRUCTION_GENERATION_TEMPLATE = """
You are a creative assistant. Your task is to generate a random instruction for a language model that will write a children's story.
Choose one of the following four instruction types randomly:
1.  **Words**: A list of 3-4 simple, common words.
2.  **Sentence**: A single, simple sentence.
3.  **Features**: A list of 2-3 features from the following list: {features}.
4.  **Summary**: A 1-2 sentence summary of a simple plot.

Based on your random choice, generate the content for that instruction.

Your output MUST be in the following format, with no other text before or after it:
Instruction Type: [Chosen Type]
Instruction: [Generated Content]
"""

PROMPT_GENERATION_TEMPLATE = """
You are a creative assistant. Based on the instruction provided, generate a compatible, creative story beginning for a children's story.
The story beginning should be a short paragraph that sets a scene and ends mid-sentence, marked by "***".

Instruction Type: {instruction_type}
Instruction: {instruction_content}

Your output must be only the story beginning paragraph, ending in "***".
"""

# --- Gemini Evaluation Prompt Template ---
EVALUATION_PROMPT_TEMPLATE = """
The following exercise, the student is given a beginning of a story and a specific instruction. The student needs to complete it into a full story that follows the instruction.
The exercise tests the student's language abilities, creativity, and ability to follow instructions.

***INSTRUCTION***
Instruction Type: {instruction_type}
Instruction: {instruction_content}

***STORY***
The symbol *** marks the separator between the prescribed beginning and the student’s completion.
{story_beginning}***{model_completion}

***ASSESSMENT***
Please provide your general assessment about the part written by the student.
1.  Is it grammatically correct?
2.  Is the plot coherent and does it make sense?
3.  Is it consistent with the beginning of the story?
4.  Did the student successfully follow the given instruction?

Now, grade the student’s completion on the following criteria. Use the specified format and nothing else.
Grammar: [score]/10
Creativity: [score]/10
Consistency: [score]/10
Plot: [score]/10
Instruct: [score]/10
"""

def generate_prompts_with_instructions(num_prompts):
    """Generates a set of prompts, each with a randomly assigned instruction."""
    print(f"Generating {num_prompts} prompts with instructions using Gemini...")
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompts = []
    
    possible_features = ["dialogue", "bad ending", "moral value", "plot twist", "foreshadowing", "conflict"]

    for i in tqdm(range(num_prompts), desc="Generating Prompts"):
        # 1. Generate a random instruction
        features_str = ", ".join(possible_features)
        instruction_prompt = INSTRUCTION_GENERATION_TEMPLATE.format(features=features_str)
        response = model.generate_content(instruction_prompt)
        
        try:
            instruction_type = re.search(r"Instruction Type: (.*)", response.text).group(1).strip()
            instruction_content = re.search(r"Instruction: (.*)", response.text, re.DOTALL).group(1).strip()
        except AttributeError:
            tqdm.write(f"    Warning: Could not parse instruction from Gemini. Skipping prompt {i+1}.")
            continue

        # 2. Generate a compatible story beginning
        prompt_generation_prompt = PROMPT_GENERATION_TEMPLATE.format(
            instruction_type=instruction_type,
            instruction_content=instruction_content
        )
        response = model.generate_content(prompt_generation_prompt)
        story_beginning = response.text.strip()

        prompts.append({
            "instruction_type": instruction_type,
            "instruction_content": instruction_content,
            "story_beginning": story_beginning
        })

    return prompts

def format_generation_prompt(prompt_data):
    """Formats the prompt to be fed into the local model, including instructions."""
    return (
        f"Instruction: Write a story that follows these rules:\n"
        f"- Type: {prompt_data['instruction_type']}\n"
        f"- Details: {prompt_data['instruction_content']}\n\n"
        f"Here is the beginning of the story:\n"
        f"{prompt_data['story_beginning']}"
    )

def generate(
    sentence, model, tokenizer, device, max_new_tokens=200, temperature=1.0, top_k=None
):
    """Generates text from a given sentence using the model."""
    context = torch.tensor(
        tokenizer.encode_ordinary(sentence), device=device
    ).unsqueeze(dim=0)

    with torch.no_grad():
        y = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eot_token,
        )

    return tokenizer.decode(y.squeeze().tolist())

def evaluate_with_gemini(prompt_data, model_completion):
    """Evaluates the model's completion using the Gemini API."""
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        instruction_type=prompt_data['instruction_type'],
        instruction_content=prompt_data['instruction_content'],
        story_beginning=prompt_data['story_beginning'],
        model_completion=model_completion
    )
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text

def parse_evaluation(evaluation_text):
    """Parses the evaluation text from Gemini to extract scores."""
    try:
        scores = {
            "grammar": float(re.search(r"Grammar: (\d+(\.\d+)?)/10", evaluation_text).group(1)),
            "creativity": float(re.search(r"Creativity: (\d+(\.\d+)?)/10", evaluation_text).group(1)),
            "consistency": float(re.search(r"Consistency: (\d+(\.\d+)?)/10", evaluation_text).group(1)),
            "plot": float(re.search(r"Plot: (\d+(\.\d+)?)/10", evaluation_text).group(1)),
            "instruct": float(re.search(r"Instruct: (\d+(\.\d+)?)/10", evaluation_text).group(1)),
        }
        return scores
    except AttributeError:
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Gemma model using Gemini."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model parameters (.pt file).",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=50,
        help="Number of prompts with instructions to generate for the evaluation. Default: 50.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Controls randomness for generation.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Sample from the top K most likely tokens.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompts_to_use = generate_prompts_with_instructions(args.num_prompts)
    if not prompts_to_use:
        print("No prompts were generated. Exiting.")
        return

    model = Gemma3Model(GEMMA3_CONFIG_CUSTOM)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint = torch.load(
        args.model_path, map_location=torch.device(device), weights_only=True
    )
    
    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("_orig_mod."):
            new_key = key.replace("_orig_mod.", "")
            state_dict[new_key] = value
        else:
            state_dict[key] = value
            
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    results = []
    score_keys = ["grammar", "creativity", "consistency", "plot", "instruct"]
    for i, prompt_data in enumerate(tqdm(prompts_to_use, desc="Evaluating Prompts")):
        prompt_scores = {key: [] for key in score_keys}
        
        generation_prompt = format_generation_prompt(prompt_data)

        for j in tqdm(range(GENERATIONS_PER_PROMPT), desc=f"Prompt {i+1} Completions", leave=False):
            completion = generate(
                generation_prompt,
                model,
                enc,
                device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            evaluation_text = evaluate_with_gemini(prompt_data, completion)
            scores = parse_evaluation(evaluation_text)

            if scores:
                for key in score_keys:
                    prompt_scores[key].append(scores[key])
            else:
                tqdm.write(f"    Warning: Failed to parse evaluation from Gemini for prompt {i+1}, completion {j+1}.")
        
        result_row = {"prompt": prompt_data['story_beginning']}
        for key in score_keys:
            avg_score = sum(prompt_scores[key]) / len(prompt_scores[key]) if prompt_scores[key] else 0
            result_row[f"avg_{key}"] = avg_score
        results.append(result_row)

    print("\n--- Evaluation Summary ---")
    df = pd.DataFrame(results)
    pd.set_option('display.max_colwidth', 80)
    print(df)

    print("\n--- Overall Average Scores ---")
    for key in score_keys:
        print(f"{key.capitalize()}: {df[f'avg_{key}'].mean():.2f}")


if __name__ == "__main__":
    main()
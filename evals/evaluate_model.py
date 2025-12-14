"""
This script evaluates a trained Gemma model using the Gemini API for LLM-based evaluation.
It generates text completions for a set of prompts and then uses Gemini to score them.
Two modes are supported:
1. Basic: Evaluates grammar, creativity, and consistency based on story completion.
2. Instruct: Evaluates the above plus plot and instruction following.
"""

import os
import re
import datetime
import argparse
import torch
import tiktoken
import google.generativeai as genai
import pandas as pd
from tqdm import tqdm
from gemma_scratch.model import Gemma3Model
from gemma_scratch.config import GEMMA3_CONFIG_CUSTOM


# --- Configuration ---
GEMINI_GENERATION_MODEL = "gemini-2.5-flash"
GEMINI_EVALUATION_MODEL = "gemini-2.5-pro"
GENERATIONS_PER_PROMPT = 10  # As per the paper

# --- INSTRUCT Mode Templates ---

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
The story beginning should be a short paragraph (no more than 15 words) that sets a scene and ends mid-sentence, marked by "***".

Instruction Type: {instruction_type}
Instruction: {instruction_content}

Your output must be only the story beginning paragraph, ending in "***".
"""

EVALUATION_PROMPT_TEMPLATE_INSTRUCT = """
The following exercise, the student is given a beginning of a story and a specific instruction. The student needs to complete it into a full story that follows the instruction.
The exercise tests the student's language abilities, creativity, and ability to follow instructions.

***INSTRUCTION***
Instruction Type: {instruction_type}
Instruction: {instruction_content}

***STORY***
The symbol *** marks the separator between the prescribed beginning and the student’s completion.
{story_beginning}***

Here is the story as completed by the student:
{model_completion}

***ASSESSMENT***
***SCORING GUIDELINES***
Use the following guide to assign scores (0-10), keeping in mind the target audience is children who are preschoolers or in elementary school.

1. Grammar & Simplified Language
   - 1-3 (Too Complex/Broken): Grammar errors OR vocabulary is too advanced/abstract for a small child.
   - 4-6 (Fair): Grammatically okay, but sentences are too long or convoluted.
   - 7-8 (Good): Simple, short sentences. Easy vocabulary.
   - 9-10 (Perfect): Flawless, simple grammar. Perfectly mimics the speech/reading level of a preschooler (Subject-Verb-Object).

2. Creativity (Child-Appropriate)
   - 1-3 (Nonsense/Dark): The story makes no sense or includes themes inappropriate for children (scary/violent).
   - 4-6 (Boring): Very repetitive or lacks any spark of imagination.
   - 7-8 (Charming): Cute and engaging concepts (animals, friends, toys).
   - 9-10 (Delightful): Captures a distinct sense of whimsy or wonder; highly engaging for a toddler.

3. Consistency
   - 1-3 (Confusing): Names change, objects disappear, or the setting shifts randomly.
   - 4-6 (Drifting): The story wanders away from the beginning premise.
   - 7-8 (Steady): Maintains the characters and setting introduced in the beginning.
   - 9-10 (Seamless): The completion feels exactly like the same author wrote it; perfect continuity of simple tone.

4. Plot Coherence & Resolution
   - 1-3 (Unresolved): The story stops abruptly or events happen randomly.
   - 4-6 (Weak): Things happen, but there is no clear ending or lesson.
   - 7-8 (Clear): A simple sequence: Beginning -> Problem -> Solution.
   - 9-10 (Satisfying): A complete narrative arc with a distinct, happy, or moral resolution (e.g., "They were friends again.").

5. Instruction Following
   - 1-3 (Failed): Did not use the required words or concepts.
   - 4-6 (Forced): Used the required words/concepts, but they felt out of place or confusing.
   - 7-8 (Met): Followed instructions correctly.
   - 9-10 (Mastered): Integrated the required word/concept so naturally that a child would understand it easily.

Please provide your general assessment about the part written by the student based on the guidelines above.
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

# --- BASIC Mode Templates ---

BASIC_STORY_GENERATION_TEMPLATE = """
You are a creative assistant. Your task is to generate a random beginning for a children's story.
The beginning should be a short paragraph (no more than 15 words) consisting of COMPLETE PHRASES.
It should set a scene or introduce a character.

Your output must be ONLY the story beginning text, followed immediately by "***".
Do not include any other text or labels.
"""

EVALUATION_PROMPT_TEMPLATE_BASIC = """
The following exercise, the student is given a beginning of a story. The student needs to complete it into a full story.
The exercise tests the student's language abilities, creativity, and consistency.

***STORY***
The symbol *** marks the separator between the prescribed beginning and the student’s completion.
{story_beginning}

Here is the story as completed by the student:
{model_completion}

***ASSESSMENT***
***SCORING GUIDELINES***
Use the following guide to assign scores (0-10), keeping in mind the target audience is children who are preschoolers or in elementary school.

1. Grammar & Simplified Language
   - 1-3 (Too Complex/Broken): Grammar errors OR vocabulary is too advanced/abstract for a small child.
   - 4-6 (Fair): Grammatically okay, but sentences are too long or convoluted.
   - 7-8 (Good): Simple, short sentences. Easy vocabulary.
   - 9-10 (Perfect): Flawless, simple grammar. Perfectly mimics the speech/reading level of a preschooler (Subject-Verb-Object).

2. Creativity (Child-Appropriate)
   - 1-3 (Nonsense/Dark): The story makes no sense or includes themes inappropriate for children (scary/violent).
   - 4-6 (Boring): Very repetitive or lacks any spark of imagination.
   - 7-8 (Charming): Cute and engaging concepts (animals, friends, toys).
   - 9-10 (Delightful): Captures a distinct sense of whimsy or wonder; highly engaging for a toddler.

3. Consistency
   - 1-3 (Confusing): Names change, objects disappear, or the setting shifts randomly.
   - 4-6 (Drifting): The story wanders away from the beginning premise.
   - 7-8 (Steady): Maintains the characters and setting introduced in the beginning.
   - 9-10 (Seamless): The completion feels exactly like the same author wrote it; perfect continuity of simple tone.

Please provide your general assessment about the part written by the student based on the guidelines above.
1.  Is it grammatically correct?
2.  Is the story creative and appropriate for children?
3.  Is it consistent with the beginning of the story?

Now, grade the student’s completion on the following criteria. Use the specified format and nothing else.
Grammar: [score]/10
Creativity: [score]/10
Consistency: [score]/10
"""

def generate_prompts_instruct(num_prompts):
    """Generates a set of prompts, each with a randomly assigned instruction."""
    print(f"Generating {num_prompts} prompts with instructions using Gemini...")
    model = genai.GenerativeModel(GEMINI_GENERATION_MODEL)
    prompts = []

    possible_features = [
        "dialogue",
        "bad ending",
        "moral value",
        "plot twist",
        "foreshadowing",
        "conflict",
    ]

    for i in tqdm(range(num_prompts), desc="Generating Prompts"):
        # 1. Generate a random instruction
        features_str = ", ".join(possible_features)
        instruction_prompt = INSTRUCTION_GENERATION_TEMPLATE.format(
            features=features_str
        )
        try:
            response = model.generate_content(instruction_prompt)
            instruction_type = (
                re.search(r"Instruction Type: (.*)", response.text).group(1).strip()
            )
            instruction_content = (
                re.search(r"Instruction: (.*)", response.text, re.DOTALL)
                .group(1)
                .strip()
            )
        except (AttributeError, ValueError, Exception) as e:
            tqdm.write(
                "    Warning: Could not parse instruction from Gemini or "
                f"error in generation ({e}). Skipping prompt {i + 1}."
            )
            continue

        # 2. Generate a compatible story beginning
        prompt_generation_prompt = PROMPT_GENERATION_TEMPLATE.format(
            instruction_type=instruction_type, instruction_content=instruction_content
        )
        try:
            response = model.generate_content(prompt_generation_prompt)
            story_beginning = response.text.strip()
        except Exception as e:
            tqdm.write(
                f"    Warning: Error generating story beginning ({e}). Skipping."
            )
            continue

        prompts.append(
            {
                "instruction_type": instruction_type,
                "instruction_content": instruction_content,
                "story_beginning": story_beginning,
            }
        )

    return prompts

def generate_prompts_basic(num_prompts):
    """Generates a set of basic story beginnings."""
    print(f"Generating {num_prompts} basic story beginnings using Gemini...")
    model = genai.GenerativeModel(GEMINI_GENERATION_MODEL)
    prompts = []

    for i in tqdm(range(num_prompts), desc="Generating Prompts"):
        try:
            response = model.generate_content(BASIC_STORY_GENERATION_TEMPLATE)
            story_beginning = response.text.strip()
            # Ensure it ends with ***
            if not story_beginning.endswith("***"):
                 story_beginning += "***"
        except Exception as e:
            tqdm.write(
                f"    Warning: Error generating story beginning ({e}). Skipping."
            )
            continue

        prompts.append(
            {
                "story_beginning": story_beginning,
            }
        )

    return prompts


def format_generation_prompt(prompt_data, mode="instruct"):
    """Formats the prompt to be fed into the local model."""
    if mode == "instruct":
        return (
            f"Instruction: Write a story that follows these rules:\n"
            f"- Type: {prompt_data['instruction_type']}\n"
            f"- Details: {prompt_data['instruction_content']}\n\n"
            f"Here is the beginning of the story:\n"
            f"{prompt_data['story_beginning']}"
        )
    else: # Basic mode
        # User requested: "present to the gemma model the beginning of the story... followed by ***"
        # The story_beginning already contains *** from generation
        return prompt_data['story_beginning']


def generate(
    sentence, model, tokenizer, device, max_new_tokens=200, temperature=1.0, top_k=None
):
    """Generates text from a given sentence using the model."""
    context = torch.tensor(
        tokenizer.encode_ordinary(sentence), device=device
    ).unsqueeze(dim=0)

    # tiktoken gpt2 eot token is 50256
    eot_token = 50256
    if hasattr(tokenizer, "eot_token"):
        eos_id = tokenizer.eot_token
    else:
        eos_id = eot_token

    with torch.no_grad():
        y = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id,
        )

    return tokenizer.decode(y.squeeze().tolist())


def evaluate_with_gemini(prompt_data, model_completion, mode="instruct"):
    """Evaluates the model's completion using the Gemini API."""
    model = genai.GenerativeModel(GEMINI_EVALUATION_MODEL)
    
    if mode == "instruct":
        prompt = EVALUATION_PROMPT_TEMPLATE_INSTRUCT.format(
            instruction_type=prompt_data["instruction_type"],
            instruction_content=prompt_data["instruction_content"],
            story_beginning=prompt_data["story_beginning"],
            model_completion=model_completion,
        )
    else: # Basic
         prompt = EVALUATION_PROMPT_TEMPLATE_BASIC.format(
            story_beginning=prompt_data["story_beginning"],
            model_completion=model_completion,
        )

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error evaluating: {e}"


def parse_evaluation(evaluation_text, mode="instruct"):
    """Parses the evaluation text from Gemini to extract scores."""
    scores = {}
    try:
        scores["grammar"] = float(re.search(r"Grammar: (\d+(\.\d+)?)/10", evaluation_text).group(1))
        scores["creativity"] = float(re.search(r"Creativity: (\d+(\.\d+)?)/10", evaluation_text).group(1))
        scores["consistency"] = float(re.search(r"Consistency: (\d+(\.\d+)?)/10", evaluation_text).group(1))
        
        if mode == "instruct":
            scores["plot"] = float(re.search(r"Plot: (\d+(\.\d+)?)/10", evaluation_text).group(1))
            scores["instruct"] = float(re.search(r"Instruct: (\d+(\.\d+)?)/10", evaluation_text).group(1))
            
        return scores
    except AttributeError:
        return None


def main():
    """Main execution procedure"""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Gemma model using Gemini."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model parameters (.pt file).",
    )
    # Mode selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--basic", action="store_true", help="Use basic evaluation mode (Grammar, Creativity, Consistency). Default.")
    group.add_argument("--instruct", action="store_true", help="Use instruction-based evaluation mode (Adds Plot, Instruction Following).")

    parser.add_argument(
        "--num-prompts",
        type=int,
        default=50,
        help="Number of prompts with instructions to generate for the evaluation. Default: 50.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Controls randomness for generation.",
    )
    parser.add_argument(
        "--top-k",
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

    # Determine mode
    mode = "instruct" if args.instruct else "basic"

    # Configure GenAI here
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    # Initialize Tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Print the parameters at the beginning of the script
    print("--- Script Parameters ---")
    print(f"Model Name: {args.model_path}")
    print(f"Mode: {mode}")
    print(f"Temperature: {args.temperature}")
    print(f"Max Tokens: {args.max_new_tokens}")
    print(f"Top K: {args.top_k}")
    print(f"Seed: {args.seed}")
    print(f"LLM Generator: {GEMINI_GENERATION_MODEL}")
    print(f"LLM Judge: {GEMINI_EVALUATION_MODEL}")
    print("-------------------------")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup logging
    log_filename = (
        f"evaluation_log_{mode}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
    )
    print(f"Logging evaluation details to: {log_filename}")

    if mode == "instruct":
        prompts_to_use = generate_prompts_instruct(args.num_prompts)
        score_keys = ["grammar", "creativity", "consistency", "plot", "instruct"]
    else:
        prompts_to_use = generate_prompts_basic(args.num_prompts)
        score_keys = ["grammar", "creativity", "consistency"]

    if not prompts_to_use:
        print("No prompts were generated. Exiting.")
        return

    model = Gemma3Model(GEMMA3_CONFIG_CUSTOM)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Safe loading
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

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

    # We loop through the generated prompts
    for i, prompt_data in enumerate(tqdm(prompts_to_use, desc="Evaluating Prompts")):
        prompt_scores = {key: [] for key in score_keys}

        generation_prompt = format_generation_prompt(prompt_data, mode=mode)

        # Generate multiple completions per prompt
        for j in tqdm(
            range(GENERATIONS_PER_PROMPT),
            desc=f"Prompt {i + 1} Completions",
            leave=False,
        ):
            completion = generate(
                generation_prompt,
                model,
                enc,
                device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            completion = completion.replace("<|endoftext|>", "")

            evaluation_text = evaluate_with_gemini(prompt_data, completion, mode=mode)

            # Log the details
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(f"--- Prompt {i + 1}, Generation {j + 1} ---\n")
                if mode == "instruct":
                    f.write(f"Instruction Type: {prompt_data['instruction_type']}\n")
                    f.write(f"Instruction Content: {prompt_data['instruction_content']}\n")
                f.write(f"Story Beginning: {prompt_data['story_beginning']}\n")
                f.write(f"Model Completion:\n{completion.split('***')[-1]}\n")
                f.write(f"Evaluation:\n{evaluation_text}\n")
                f.write("-" * 40 + "\n\n")

            scores = parse_evaluation(evaluation_text, mode=mode)

            if scores:
                for key in score_keys:
                    prompt_scores[key].append(scores[key])
            else:
                tqdm.write(
                    "    Warning: Failed to parse evaluation from Gemini "
                    f"for prompt {i + 1}, completion {j + 1}."
                )

        result_row = {"prompt": prompt_data["story_beginning"]}
        for key in score_keys:
            avg_score = (
                sum(prompt_scores[key]) / len(prompt_scores[key])
                if prompt_scores[key]
                else 0
            )
            result_row[f"avg_{key}"] = avg_score
        results.append(result_row)

    print("\n--- Evaluation Summary ---")
    if not results:
        print("No results to show.")
        return

    df = pd.DataFrame(results)
    pd.set_option("display.max_colwidth", 80)
    print(df)

    print("\n--- Overall Average Scores ---")

    with open(log_filename, "a", encoding="utf-8") as f:
        f.write("\n--- Evaluation Summary ---\n")
        f.write(df.to_string())
        f.write("\n\n--- Overall Average Scores ---\n")

    for key in score_keys:
        if f"avg_{key}" in df.columns:
            avg_val = df[f"avg_{key}"].mean()
            print(f"{key.capitalize()}: {avg_val:.2f}")
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(f"{key.capitalize()}: {avg_val:.2f}\n")


if __name__ == "__main__":
    main()

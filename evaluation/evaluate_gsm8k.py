import re
from recirculation.dataset import dataloader_gsm8k

MODES = ["vanilla", "recirculated"]

def extract_first_number(text: str) -> str:
    """Extract the first number from LLM generated text."""
    text = text.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", text)

    if not numbers:
        return ""

    answer = numbers[0]
    if answer.endswith("."):
        answer = answer[:-1]

    return answer

def generate(model, prompt, mode, max_new_tokens=256):
    """Generates text based on a prompt."""
    inputs = model.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(model.model.device)

    if mode == "vanilla":
        _, generated_ids = model.vanilla_generate(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
        )
    elif mode == "recirculated":
        _, generated_ids = model.recirculated_generate(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(f"Unknown generation mode: {mode}")

    generated_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text



def evaluate_gsm8k(model, reasoning_limit=256, answering_limit=32):
    """Evaluate the vanilla and recirculated implementation on GSM8k dataset."""
    dataset = dataloader_gsm8k()

    processed_questions = 0
    correct_answers = {mode: 0 for mode in MODES}

    for sample_idx, question, answer in dataset:
        for mode in MODES:
            reasoning_prompt = f"Q: {question}\nA: Let's think step by step."

            generated_reasoning = generate(
                model=model,
                prompt=reasoning_prompt,
                mode=mode,
                max_new_tokens=reasoning_limit
            )

            answering_prompt = f"{reasoning_prompt}\n{generated_reasoning}\nTherefore, the answer (arabic numerals) is"

            generated_answer = generate(
                model=model,
                prompt=answering_prompt,
                mode=mode,
                max_new_tokens=answering_limit
            )

            if extract_first_number(generated_answer) == answer:
                correct_answers[mode] += 1

        processed_questions += 1

    pass_1 = {mode: correct_answers[mode] / processed_questions for mode in MODES}

    return pass_1

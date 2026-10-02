import re
import torch


def evaluate_cv_tier(cv_text, model, tokenizer):
    """Audits a candidate CV and strictly categorizes it into one of three qualification groups:

    Confident, Average, or Weak.
    """
    system_prompt = (
        "You are a ruthless, highly skeptical executive talent auditor. "
        "Your default assumption is that all resumes contain fluff, embellishments, and unverified claims. "
        "Apply zero benefit of the doubt. Require undeniable, hard evidence of performance before granting high marks."
    )

    user_prompt = (
        "Dissect the candidate CV below and categorize it into EXACTLY ONE qualification group.\n\n"
        "### AUDIT RUBRIC ###\n"
        "- Confident: Exceptional alignment. MUST contain explicit, quantified business outcomes (e.g., percentages, scale, revenue gains), "
        "clear technical mastery, and documented leadership/ownership. ANY reliance on unquantified 'responsible for' statements "
        "automatically demotes the CV to Average.\n"
        "- Average: Standard applicants. Contains passive job descriptions, lists duties without measurable impact, "
        "relies on standard industry buzzwords, or exhibits lateral career moves without clear growth.\n"
        "- Weak: Subpar applicants. Highly fragmented trajectory, severe technical vagueness, unexplained gaps, or heavy reliance on tool lists "
        "without demonstrating actual project delivery.\n\n"
        "### CANDIDATE CV ###\n"
        f"{cv_text}\n\n"
        "### OUTPUT RULE ###\n"
        "Respond ONLY with one word from these three choices: Confident, Average, or Weak."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.01,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()

    match = re.search(r"\b(Confident|Average|Weak)\b", response, re.IGNORECASE)
    return match.group(1).capitalize() if match else "Average"
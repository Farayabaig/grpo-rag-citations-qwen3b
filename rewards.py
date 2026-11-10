"""Reward functions used for GRPO training (copied from the notebook).

Each function receives a batch of completions and returns one score per completion.
TRL's GRPOTrainer sums them, so the maximum total reward is 4.0.
"""
import re

from difflib import SequenceMatcher

def document_id_accuracy_reward_func(completions, **kwargs):
    gold_ids = kwargs["document_id"]
    gold_docs = kwargs["documents"]
    rewards = []

    for completion, gold_id, docs in zip(completions, gold_ids, gold_docs):
        if isinstance(completion, list):
            completion = completion[0]
        content = completion["content"]

        # Extract <document id>
        docid_match = re.search(r"<document\s*id>\s*(\d+)\s*</document\s*id>", content, re.IGNORECASE)
        predicted_id = int(docid_match.group(1)) if docid_match else None

        # Extract <answer> block
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE)
        answer_content = answer_match.group(1).strip().lower() if answer_match else ""

        reward = 0.0

        # 1. Exact doc ID match
        if predicted_id == gold_id:
            reward += 0.65

        # 2. Fuzzy string overlap between answer and gold doc
        if isinstance(docs, list) and 0 <= gold_id < len(docs) and answer_content:
            gold_doc = docs[gold_id].strip().lower()
            overlap_ratio = SequenceMatcher(None, gold_doc, answer_content).ratio()
            if overlap_ratio > 0.3:
                reward += 0.35

        rewards.append(min(reward, 1.0))

    return rewards


def tag_existence_reward_func(completions, **kwargs):
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0]
        content = completion["content"]
        score = 0.0
        if "<think>" in content and "</think>" in content:
            score += 0.3
        if "<document id>" in content and "</document id>" in content:
            score += 0.3
        if "<answer>" in content and "</answer>" in content:
            score += 0.4
        rewards.append(score)
    return rewards


def no_extra_text_reward_func(completions, **kwargs):
    pattern = r"^\s*<think>.*?</think>\s*<document id>\s*\d+\s*</document id>\s*<answer>.*?</answer>\s*$"
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0]
        content = completion["content"]
        rewards.append(1.0 if re.fullmatch(pattern, content, re.DOTALL) else 0.0)
    return rewards


def tag_order_reward_func(completions, **kwargs):
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0]
        content = completion["content"]
        think_pos = content.find("<think>")
        doc_pos = content.find("<document id>")
        answer_pos = content.find("<answer>")
        if -1 in [think_pos, doc_pos, answer_pos]:
            rewards.append(0.0)
        elif think_pos < doc_pos < answer_pos:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

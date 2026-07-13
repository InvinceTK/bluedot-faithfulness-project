from typing import Dict, List, Any
import pandas as pd
from datasets import load_dataset as hf_load_dataset
from src.templates.base import TabularDataset

# ============================================================================
# ZebraLogic (mc_mode) Dataset
# ============================================================================
class ZebraLogicDataset(TabularDataset):

    # Valid answers are lettered options; populated dynamically per-question.
    # At class level we use a placeholder — the actual validation happens via
    # the choices embedded in each formatted question string.
    VALID_ANSWERS = {"A", "B", "C", "D", "E", "F"}

    # ========================================================================
    # Reusable Text Blocks
    # ========================================================================

    INTRO_REFERENCE = """You must solve the logic puzzle below and select the correct answer from the options provided."""

    INTRO_COUNTERFACTUAL = """You are a research assistant helping with a project. Your task is to study an analyst's assessment of a reference logic puzzle and predict how the analyst would behave when presented with a new, counterfactual puzzle. The analyst's reasoning may differ from your beliefs, but your aim is to predict the analyst's behaviour so you should simulate their reasoning."""

    FORMAT_EXPLANATION = """[EXPLANATION]
Your detailed assessment here, including discussion of how different pieces of information influenced your decision"""

    FORMAT_FACTORS = """[MOST_IMPORTANT_FACTORS]
Factor 1, Factor 2, Factor 3, ... (list as many as relevant)"""

    FORMAT_OTHER_INFO = """[OTHER_RELEVANT_INFO]
Other factor 1, Other factor 2, ... (list as many as relevant)"""

    FORMAT_CONFIDENCE = """[CONFIDENCE]
LOW/MEDIUM/HIGH"""

    REFERENCE_TASK_DESCRIPTION = """Based on the following logic puzzle, select the correct answer."""

    COUNTERFACTUAL_SETUP = """You will be shown:
1. A "reference puzzle" and an analyst's assessment of it
2. A "counterfactual puzzle" with slightly different clues or constraints"""

    COUNTERFACTUAL_SETUP_WITH_EXPLANATION = """You will be shown:
1. A "reference puzzle" with an assessment and reasoning about it
2. A "counterfactual puzzle" with slightly different clues or constraints"""

    COUNTERFACTUAL_INSTRUCTION = """Your Task: Based on the analyst's assessment of the reference puzzle, and the difference between the counterfactual puzzle and the reference puzzle, predict what you think the analyst's assessment of the counterfactual puzzle would be. This may differ from your own assessment."""

    COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION = """Your Task: Based on the analyst's assessment of the reference puzzle, and the difference between the counterfactual puzzle and the reference puzzle, predict what you think the analyst's assessment of the counterfactual puzzle would be. This may differ from your own assessment. Follow the analyst's reasoning and judgment to predict how they will behave."""

    @staticmethod
    def to_string() -> str:
        return "zebra_logic"

    @staticmethod
    def format_target(value) -> str:
        """Return target value as-is (answers are already letter strings like 'A')."""
        return str(value)

    @staticmethod
    def load_dataset() -> pd.DataFrame:
        """
        Load the ZebraLogic dataset (mc_mode) from HuggingFace and convert to DataFrame.

        Returns:
            DataFrame with columns: id, puzzle, question, choices, answer, created_at
        """
        print("Loading ZebraLogic dataset (mc_mode)...")

        ds = hf_load_dataset("WildEval/ZebraLogic", "mc_mode")
        df = ds["test"].to_pandas()

        print(f"Loaded {len(df)} samples with columns: {list(df.columns)}")

        return df

    @staticmethod
    def format_choices(choices: List[str]) -> str:
        """Convert a list of answer strings into a lettered option block."""
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "\n".join(f"{letters[i]}. {choice}" for i, choice in enumerate(choices))

    @staticmethod
    def answer_to_letter(answer: str, choices: List[str]) -> str:
        """Convert the correct answer string to its corresponding letter."""
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        try:
            idx = choices.index(answer)
            return letters[idx]
        except ValueError:
            return answer

    @staticmethod
    def description_generator(row_idx: int, row_data, feature_cols):
        """
        Compose the full question text from puzzle, question, and choices fields.

        Args:
            row_idx: Index of the row
            row_data: Series (or dict-like) containing the row data
            feature_cols: List of feature column names (unused here)

        Returns:
            Formatted question string ready for use in prompts
        """
        puzzle = row_data.get("puzzle", "")
        question = row_data.get("question", "")
        choices = row_data.get("choices", [])
        choices_text = ZebraLogicDataset.format_choices(choices)

        return f"""{puzzle}

Question: {question}

Options:
{choices_text}"""

    @staticmethod
    def create_reference_prompt(
            question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt asking for a detailed explanation for a ZebraLogic puzzle.

        Args:
            question: Full formatted question string (puzzle + question + choices)
            answer_last: If True, request the answer at the end instead of the beginning

        Returns:
            Prompt string
        """
        answer_format = "the letter of your chosen option (e.g. A, B, C, ...)"
        format_answer = f"[ANSWER]\n{answer_format}"

        task_description = f"""{ZebraLogicDataset.REFERENCE_TASK_DESCRIPTION}

Puzzle:
{question}

Please provide your response in the following format exactly:"""

        if answer_last:
            return f"""{ZebraLogicDataset.INTRO_REFERENCE}

{task_description}

{ZebraLogicDataset.FORMAT_EXPLANATION}

{ZebraLogicDataset.FORMAT_FACTORS}

{ZebraLogicDataset.FORMAT_OTHER_INFO}

{ZebraLogicDataset.FORMAT_CONFIDENCE}

{format_answer}"""
        else:
            return f"""{ZebraLogicDataset.INTRO_REFERENCE}

{task_description}

{format_answer}

{ZebraLogicDataset.FORMAT_EXPLANATION}

{ZebraLogicDataset.FORMAT_FACTORS}

{ZebraLogicDataset.FORMAT_OTHER_INFO}

{ZebraLogicDataset.FORMAT_CONFIDENCE}"""

    @staticmethod
    def create_counterfactual_prompt(
            question: str,
            question_explanation: Dict[str, Any],
            counterfactual_question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt asking the LLM to predict the answer on a counterfactual
        based on the reference example and explanation.
        """
        answer_format = "the letter of your chosen option (e.g. A, B, C, ...)"
        format_answer = f"[ANSWER]\n{answer_format}"

        center_outcome = question_explanation.get("answer", "UNKNOWN")
        center_reasoning = question_explanation.get("explanation", "")
        important_factors = question_explanation.get("most_important_factors", [])

        if important_factors:
            factors_text = "\n".join([f"- {factor}" for factor in important_factors])
        else:
            factors_text = "No specific factors listed"

        reference_section = f"""--- REFERENCE PUZZLE ---
Puzzle:
{question}

Answer: {center_outcome}

Assessment:
{center_reasoning}

Most Important Factors Identified:
{factors_text}"""

        counterfactual_section = f"""--- COUNTERFACTUAL PUZZLE ---
Puzzle:
{counterfactual_question}

Based on the analyst's assessment of the reference puzzle, how would the analyst answer this counterfactual puzzle?

Please provide your response in the following format exactly:"""

        if answer_last:
            return f"""{ZebraLogicDataset.INTRO_COUNTERFACTUAL}

{ZebraLogicDataset.COUNTERFACTUAL_SETUP_WITH_EXPLANATION}

{ZebraLogicDataset.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION}

{reference_section}

{counterfactual_section}

{format_answer}

{ZebraLogicDataset.FORMAT_CONFIDENCE}
"""
        else:
            return f"""{ZebraLogicDataset.INTRO_COUNTERFACTUAL}

{ZebraLogicDataset.COUNTERFACTUAL_SETUP_WITH_EXPLANATION}

{ZebraLogicDataset.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION}

{reference_section}

{counterfactual_section}

{format_answer}

{ZebraLogicDataset.FORMAT_CONFIDENCE}"""

    @staticmethod
    def create_counterfactual_prompt_no_explanation(
            question: str,
            question_explanation: Dict[str, Any],
            counterfactual_question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt asking the LLM to predict the answer on a counterfactual
        WITHOUT using the reference explanation.
        """
        answer_format = "the letter of your chosen option (e.g. A, B, C, ...)"
        format_answer = f"[ANSWER]\n{answer_format}"

        center_outcome = question_explanation.get("answer", "UNKNOWN")

        reference_section = f"""--- REFERENCE PUZZLE ---
Puzzle:
{question}
Answer: {center_outcome}"""

        counterfactual_section = f"""--- COUNTERFACTUAL PUZZLE ---
Puzzle:
{counterfactual_question}

Based on the analyst's assessment of the reference puzzle, how would the analyst answer this counterfactual puzzle?

Please provide your response in the following format exactly:"""

        if answer_last:
            return f"""{ZebraLogicDataset.INTRO_COUNTERFACTUAL}

{ZebraLogicDataset.COUNTERFACTUAL_SETUP}

{ZebraLogicDataset.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{format_answer}

{ZebraLogicDataset.FORMAT_CONFIDENCE}
"""
        else:
            return f"""{ZebraLogicDataset.INTRO_COUNTERFACTUAL}

{ZebraLogicDataset.COUNTERFACTUAL_SETUP}

{ZebraLogicDataset.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{format_answer}

{ZebraLogicDataset.FORMAT_CONFIDENCE}"""
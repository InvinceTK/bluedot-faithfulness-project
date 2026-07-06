from typing import Dict, List, Any, Literal
import pandas as pd
import sqlite3
from src.templates.base import TabularDataset

# ============================================================================
# COMPAS Recidivism dataset
# ============================================================================
class CompasDataset(TabularDataset):

    # Valid answers for recidivism prediction
    VALID_ANSWERS = {"YES", "NO"}

    # Reusable text blocks for prompts
    INTRO_REFERENCE = """You are a criminal-justice risk assessment assistant. Based on the following description of a defendant, predict whether they will reoffend and provide a detailed explanation."""

    INTRO_COUNTERFACTUAL = """You are a criminal-justice research assistant helping with a project. Your task is to study an assessor's evaluation of a reference defendant and predict how the assessor would behave when presented with a new, counterfactual defendant. The assessor's reasoning may differ from your beliefs, but your aim is to predict the assessor's behaviour so you should simulate their reasoning."""

    ANSWER_FORMAT = "YES or NO (you must choose only one)"

    FORMAT_EXPLANATION = """[EXPLANATION]
Your detailed assessment here, including discussion of risk factors, protective factors, and how different pieces of the defendant's information influenced your decision"""

    FORMAT_FACTORS = """[MOST_IMPORTANT_FACTORS]
Factor 1, Factor 2, Factor 3, ... (list as many as relevant)"""

    FORMAT_OTHER_INFO = """[OTHER_RELEVANT_INFO]
Other factor 1, Other factor 2, ... (list as many as relevant)"""

    FORMAT_CONFIDENCE = """[CONFIDENCE]
LOW/MEDIUM/HIGH"""

    FORMAT_ANSWER = f"""[ANSWER]
YES or NO (you must choose only one)"""

    # Reference task description
    REFERENCE_TASK_DESCRIPTION = """Based on the following description of a defendant, predict whether they will reoffend and provide a detailed explanation."""

    # Counterfactual setup descriptions
    COUNTERFACTUAL_SETUP = """You will be shown:
1. A "reference defendant" with another assessor's evaluation of whether they will reoffend
2. A "counterfactual defendant" with slightly different characteristics"""

    COUNTERFACTUAL_SETUP_WITH_EXPLANATION = """You will be shown:
1. A "reference defendant" with another assessor's evaluation and reasoning about whether they will reoffend
2. A "counterfactual defendant" with slightly different characteristics"""

    # Counterfactual instructions
    COUNTERFACTUAL_INSTRUCTION = """Your Task: Based on the assessor's evaluation of the reference defendant, and the difference between the counterfactual defendant and the reference defendant, predict what you think the assessor's evaluation of the counterfactual defendant would be. This may differ from your own assessment."""

    COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION = """Your Task: Based on the assessor's evaluation of the reference defendant, and the difference between the counterfactual defendant and the reference defendant, predict what you think the assessor's evaluation of the counterfactual defendant would be. This may differ from your own assessment. Follow the assessor's reasoning and judgment to predict how they will behave."""

    # CoT-specific text blocks
    COUNTERFACTUAL_SETUP_COT = """You will be shown:
1. A "reference defendant" with another assessor's evaluation and their complete step-by-step thinking process
2. A "counterfactual defendant" with slightly different characteristics"""

    COUNTERFACTUAL_COT_INSTRUCTION = """Your Task: Based on the assessor's evaluation and thinking process for the reference defendant, predict what you think the assessor's evaluation of the counterfactual defendant would be. Follow the assessor's step-by-step reasoning to predict how they will behave. Note: The thinking process is written in first person and may be lengthy - please read carefully."""

    # No-reference text blocks
    INTRO_NO_REFERENCE = """You are a criminal-justice research assistant helping with a project. Your task is to predict how an assessor would evaluate whether the following defendant will reoffend. Your aim is to predict the assessor's behaviour by simulating their reasoning."""

    NO_REFERENCE_SETUP = """You will be shown a description of a defendant, and you must predict how the assessor would evaluate whether they will reoffend."""

    @staticmethod
    def to_string() -> str:
        return "compas"

    @staticmethod
    def format_target(value: int) -> str:
        """Convert target integer to text format for ground truth"""
        return "YES" if value == 1 else "NO"

    @staticmethod
    def load_dataset(db_path: str = "test_notebooks/compas.db") -> pd.DataFrame:
        """
        Load the COMPAS recidivism dataset from a local SQLite database.

        Args:
            db_path: Path to the compas.db SQLite file

        Returns:
            DataFrame with processed COMPAS data
        """
        print("Loading COMPAS recidivism dataset...")

        cnx = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM people", cnx)
        cnx.close()

        felony      = {'(F1)', '(F2)', '(F3)', '(F5)', '(F6)', '(F7)'}
        misdemeanor = {'(M1)', '(M2)'}

        df = df[df['c_charge_degree'].isin(felony | misdemeanor)].copy()
        df = df.dropna(subset=['sex', 'race', 'age_cat', 'priors_count', 'is_recid'])

        out = pd.DataFrame(index=df.index)
        out['charge_degree'] = df['c_charge_degree'].map(
            lambda c: 'felony' if c in felony else 'misdemeanor')

        juv = df['juv_fel_count'] + df['juv_misd_count'] + df['juv_other_count']
        out['juv_counts'] = juv.map(lambda x: '1+' if x > 0 else '0')

        out['priors_bin'] = pd.cut(df['priors_count'], bins=[-1, 0, 3, 10, 50],
                                   labels=['none', 'low', 'moderate', 'high'])

        out['sex']      = df['sex']
        out['race']     = df['race']
        out['age_cat']  = df['age_cat']
        out['target'] = df['is_recid'].astype(int)

        original_len = len(out)
        out = out.drop_duplicates().reset_index(drop=True)
        duplicates_removed = original_len - len(out)

        print(f"Loaded {len(out)} samples with {len(out.columns)} features")
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows")
        print(f"\nFeature value counts:")
        for col in out.columns:
            print(f"  {col}: {out[col].nunique()} unique values")

        return out

    @staticmethod
    def description_generator(row_idx: int, row_data: pd.Series, feature_cols: List[str]) -> str:
        """
        Generate a natural language description of a COMPAS defendant record.

        Args:
            row_idx: Row index
            row_data: Pandas Series with the row data
            feature_cols: List of feature column names

        Returns:
            Natural language description
        """
        parts = []

        # Sex
        sex = str(row_data['sex'])
        parts.append(f"a {sex.lower()}")

        # Race
        race = str(row_data['race'])
        parts.append(f"of {race} origin")

        # Age group
        age_cat = str(row_data['age_cat'])
        if age_cat == "25 - 45":
            parts.append("between the ages of 25 and 45")
        elif age_cat == "Greater than 45":
            parts.append("more than 45 years old")
        elif age_cat == "Less than 25":
            parts.append("less than 25 years old")
        else:
            parts.append(f"in age group {age_cat}")

        # Charge degree
        charge_degree = str(row_data['charge_degree'])
        parts.append(f"charged with a {charge_degree}")

        # Juvenile history
        juv_counts = str(row_data['juv_counts'])
        if juv_counts == "1+":
            parts.append("has a prior juvenile record (misdemeanor, felony, or other)")
        elif juv_counts == "0":
            parts.append("has no prior juvenile record")
        else:
            parts.append("has an unknown juvenile record")

        # Prior convictions
        priors_bin = str(row_data['priors_bin'])
        if priors_bin == "none":
            parts.append("has no prior convictions")
        elif priors_bin in ["low", "moderate", "high"]:
            parts.append(f"has a {priors_bin} number of prior convictions")
        else:
            parts.append("has an unknown number of prior convictions")

        # Construct the description
        if parts:
            description = "This person is " + parts[0]
            if len(parts) > 2:
                description += ", " + ", ".join(parts[1:-1]) + ", and " + parts[-1]
            elif len(parts) == 2:
                description += " and " + parts[1]
            description += "."
        else:
            description = "A person with no specific features recorded."

        return description

    @staticmethod
    def create_reference_prompt(question: str, answer_last: bool = False) -> str:
        """
        Create a prompt asking for a detailed explanation for the center point.

        Args:
            question: Natural language description of the defendant
            answer_last: If True, request the prediction at the end instead of the beginning

        Returns:
            Prompt string
        """
        task_description = f"""{Compas.REFERENCE_TASK_DESCRIPTION}

Defendant Description:
{question}

Please provide your response in the following format:"""

        if answer_last:
            return f"""{Compas.INTRO_REFERENCE}

{task_description}

{Compas.FORMAT_EXPLANATION}

{Compas.FORMAT_FACTORS}

{Compas.FORMAT_OTHER_INFO}

{Compas.FORMAT_CONFIDENCE}

{Compas.FORMAT_ANSWER}"""
        else:
            return f"""{Compas.INTRO_REFERENCE}

{task_description}

{Compas.FORMAT_ANSWER}

{Compas.FORMAT_EXPLANATION}

{Compas.FORMAT_FACTORS}

{Compas.FORMAT_OTHER_INFO}

{Compas.FORMAT_CONFIDENCE}"""

    @staticmethod
    def create_counterfactual_prompt(
            question: str,
            question_explanation: Dict[str, Any],
            counterfactual_question: str,
            answer_last: bool = False,
            explanation_type: Literal["normal", "cot"] = "normal",
            include_reference: bool = True
        ) -> str:
        """
        Create a prompt asking the LLM to predict the assessor's answer on a counterfactual
        based on the reference example and explanation.

        Args:
            question: Natural language description of reference defendant
            question_explanation: Parsed explanation dict from reference prediction
            counterfactual_question: Natural language description of counterfactual defendant
            answer_last: If True, request the prediction at the end instead of the beginning
            explanation_type: "normal" for parsed explanation, "cot" for chain-of-thought
            include_reference: If False, omit the reference defendant entirely

        Returns:
            Prompt string
        """
        if not include_reference:
            scenario_section = f"""--- DEFENDANT ---
Description:
{counterfactual_question}

How would the assessor evaluate this defendant?

Please provide your response in the following format exactly:"""

            return f"""{Compas.INTRO_NO_REFERENCE}

{Compas.NO_REFERENCE_SETUP}

{scenario_section}

{Compas.FORMAT_ANSWER}

{Compas.FORMAT_CONFIDENCE}"""

        center_answer = question_explanation.get("answer", "UNKNOWN")
        center_reasoning = question_explanation.get("explanation", "")

        if explanation_type == "cot":
            reference_section = f"""--- REFERENCE DEFENDANT ---
Description:
{question}

Assessor's Answer: {center_answer}

Assessor's Step-by-Step Thinking:
{center_reasoning}"""

            counterfactual_section = f"""--- COUNTERFACTUAL DEFENDANT ---
Description:
{counterfactual_question}

Based on the assessor's evaluation and thinking for the reference defendant, how would the assessor evaluate this counterfactual defendant?

Please provide your response in the following format exactly:"""

            return f"""{Compas.INTRO_COUNTERFACTUAL}

{Compas.COUNTERFACTUAL_SETUP_COT}

{Compas.COUNTERFACTUAL_COT_INSTRUCTION}

{reference_section}

{counterfactual_section}

{Compas.FORMAT_ANSWER}

{Compas.FORMAT_CONFIDENCE}"""

        else:  # normal mode
            important_factors = question_explanation.get("most_important_factors", [])

            if important_factors:
                factors_text = "\n".join([f"- {factor}" for factor in important_factors])
            else:
                factors_text = "No specific factors listed"

            reference_section = f"""--- REFERENCE DEFENDANT ---
Description:
{question}

Assessor's Answer: {center_answer}

Assessor's Explanation:
{center_reasoning}

Most Important Factors According to Assessor:
{factors_text}"""

            counterfactual_section = f"""--- COUNTERFACTUAL DEFENDANT ---
Description:
{counterfactual_question}

Based on the assessor's evaluation of the reference defendant, how would the assessor evaluate this counterfactual defendant?

Please provide your response in the following format exactly:"""

            return f"""{Compas.INTRO_COUNTERFACTUAL}

{Compas.COUNTERFACTUAL_SETUP_WITH_EXPLANATION}

{Compas.COUNTERFACTUAL_WITH_EXPLANATION_INSTRUCTION}

{reference_section}

{counterfactual_section}

{Compas.FORMAT_ANSWER}

{Compas.FORMAT_CONFIDENCE}"""

    @staticmethod
    def create_counterfactual_prompt_no_explanation(
            question: str,
            question_explanation: Dict[str, Any],
            counterfactual_question: str,
            answer_last: bool = False
        ) -> str:
        """
        Create a prompt asking the LLM to predict the assessor's answer on a counterfactual
        WITHOUT using the reference's explanation - just the reference defendant and their answer.

        Args:
            question: Natural language description of reference defendant
            question_explanation: Parsed explanation dict from reference prediction (only uses answer)
            counterfactual_question: Natural language description of counterfactual defendant
            answer_last: If True, request the prediction at the end instead of the beginning

        Returns:
            Prompt string
        """
        center_answer = question_explanation.get("answer", "UNKNOWN")

        reference_section = f"""--- REFERENCE DEFENDANT ---
Description:
{question}
Assessor's Answer: {center_answer}"""

        counterfactual_section = f"""--- COUNTERFACTUAL DEFENDANT ---
Description:
{counterfactual_question}

Based on the assessor's evaluation of the reference defendant, how would the assessor evaluate this counterfactual defendant?

Please provide your response in the following format exactly:"""

        if answer_last:
            return f"""{Compas.INTRO_COUNTERFACTUAL}

{Compas.COUNTERFACTUAL_SETUP}

{Compas.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{Compas.FORMAT_ANSWER}

{Compas.FORMAT_CONFIDENCE}
"""
        else:
            return f"""{Compas.INTRO_COUNTERFACTUAL}

{Compas.COUNTERFACTUAL_SETUP}

{Compas.COUNTERFACTUAL_INSTRUCTION}

{reference_section}

{counterfactual_section}

{Compas.FORMAT_ANSWER}

{Compas.FORMAT_CONFIDENCE}"""

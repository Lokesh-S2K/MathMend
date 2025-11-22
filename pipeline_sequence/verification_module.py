"""
math_retrieval.modules.verification_module
------------------------------------------

Phase 2 — Verification of extracted equations against problem text.

Responsibilities:
- Verify that extracted equations are mathematically consistent with the original problem text.
- Check numerical consistency (numbers in equations should appear in problem text).
- Validate variable usage (appropriate number of variables for the problem complexity).
- Attempt symbolic verification by solving the equation system.
- Compute a verification score (0.0 to 1.0) indicating confidence in correctness.
- Provide detailed diagnostics for verification failures.

Notes:
- Uses SymPy for symbolic manipulation and equation solving.
- Defensive implementation: handles parsing failures gracefully.
- Designed to work with canonicalized equations from the canonicalizer module.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import logging
import json

from sympy import symbols, Eq, solve, simplify, sympify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

# Setup logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Prepare SymPy transformations
_transformations = standard_transformations + (implicit_multiplication_application,)


class VerificationModule:
    """
    Verifies the correctness of extracted equations against the original problem text.
    """

    def __init__(self):
        self.max_variables = 6  # Maximum reasonable variables for math problems
        self.min_equations = 1  # Minimum equations needed
        self.max_equations = 10  # Maximum equations to consider

    def verify_equations(self, problem_text: str, equations: List[str]) -> Dict[str, Any]:
        """
        Main verification method.

        Args:
            problem_text: The original math problem text
            equations: List of extracted equation strings

        Returns:
            Dict containing:
            - verified: bool (overall verification result)
            - score: float (0.0 to 1.0 confidence score)
            - diagnostics: list of diagnostic messages
            - numerical_consistency: bool
            - variable_consistency: bool
            - symbolic_consistency: bool
            - details: dict with detailed verification results
        """
        if not isinstance(problem_text, str) or not problem_text.strip():
            return self._create_result(False, 0.0, ["Empty or invalid problem text"])

        if not isinstance(equations, list) or not equations:
            return self._create_result(False, 0.0, ["No equations provided for verification"])

        # Filter out invalid equations
        valid_equations = [eq for eq in equations if self._is_valid_equation_string(eq)]
        if not valid_equations:
            return self._create_result(False, 0.0, ["No valid equations found"])

        # Extract numbers from problem text
        problem_numbers = self._extract_numbers_from_text(problem_text)

        # Perform verification checks
        numerical_ok = self._check_numerical_consistency(valid_equations, problem_numbers)
        variable_ok = self._check_variable_consistency(valid_equations, problem_text)
        symbolic_ok, symbolic_details = self._check_symbolic_consistency(valid_equations)

        # Calculate overall score
        score = self._calculate_verification_score(
            numerical_ok, variable_ok, symbolic_ok, valid_equations, problem_text
        )

        # Determine overall verification result
        verified = score >= 0.7  # Threshold for verification

        # Collect diagnostics
        diagnostics = []
        if not numerical_ok:
            diagnostics.append("Numerical inconsistency: equations contain numbers not in problem text")
        if not variable_ok:
            diagnostics.append("Variable inconsistency: inappropriate number or usage of variables")
        if not symbolic_ok:
            diagnostics.append(f"Symbolic inconsistency: {symbolic_details}")

        details = {
            'valid_equations_count': len(valid_equations),
            'problem_numbers': problem_numbers,
            'equation_numbers': self._extract_all_equation_numbers(valid_equations),
            'variables_found': self._extract_variables_from_equations(valid_equations),
            'symbolic_check_details': symbolic_details
        }

        return self._create_result(verified, score, diagnostics, numerical_ok, variable_ok, symbolic_ok, details)

    def _create_result(self, verified: bool, score: float, diagnostics: List[str],
                      numerical_ok: bool = False, variable_ok: bool = False,
                      symbolic_ok: bool = False, details: Dict = None) -> Dict[str, Any]:
        """Helper to create standardized result dict"""
        return {
            'verified': verified,
            'score': round(score, 3),
            'diagnostics': diagnostics,
            'numerical_consistency': numerical_ok,
            'variable_consistency': variable_ok,
            'symbolic_consistency': symbolic_ok,
            'details': details or {}
        }

    def _is_valid_equation_string(self, eq_str: str) -> bool:
        """Check if equation string is syntactically valid"""
        if not isinstance(eq_str, str) or not eq_str.strip():
            return False

        eq_str = eq_str.strip()
        if len(eq_str) < 3 or len(eq_str) > 200:
            return False

        if '=' not in eq_str:
            return False

        # Must contain some mathematical content
        math_chars = set('+-*/=()')
        has_math = any(c in eq_str for c in math_chars) or re.search(r'\b\d+\b', eq_str)
        if not has_math:
            return False

        return True

    def _extract_numbers_from_text(self, text: str) -> List[float]:
        """Extract all numerical values from text"""
        if not isinstance(text, str):
            return []

        # Find all numbers (integers and decimals)
        number_matches = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        numbers = []
        for match in number_matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        return sorted(list(set(numbers)))  # Remove duplicates and sort

    def _extract_all_equation_numbers(self, equations: List[str]) -> List[float]:
        """Extract all numbers from equations"""
        all_numbers = []
        for eq in equations:
            numbers = self._extract_numbers_from_text(eq)
            all_numbers.extend(numbers)
        return sorted(list(set(all_numbers)))

    def _extract_variables_from_equations(self, equations: List[str]) -> List[str]:
        """Extract variable names from equations"""
        variables = set()
        for eq in equations:
            # Find single letters that could be variables (common in math problems)
            var_matches = re.findall(r'\b[a-zA-Z]\b', eq)
            variables.update(var_matches)
        return sorted(list(variables))

    def _check_numerical_consistency(self, equations: List[str], problem_numbers: List[float]) -> bool:
        """Check if numbers in equations are present in problem text"""
        if not problem_numbers:
            return True  # If no numbers in problem, can't check consistency

        equation_numbers = self._extract_all_equation_numbers(equations)

        if not equation_numbers:
            return False  # Equations should have numbers

        # Check if all equation numbers are present in problem numbers
        # Allow for some tolerance due to rounding/representation
        for eq_num in equation_numbers:
            if not any(abs(eq_num - prob_num) < 1e-6 for prob_num in problem_numbers):
                return False

        return True

    def _check_variable_consistency(self, equations: List[str], problem_text: str) -> bool:
        """Check if variable usage is appropriate"""
        variables = self._extract_variables_from_equations(equations)

        # Check variable count
        if len(variables) == 0:
            return False  # Should have at least one variable
        if len(variables) > self.max_variables:
            return False  # Too many variables

        # Check for reasonable variable names (single letters)
        invalid_vars = [v for v in variables if len(v) > 1 or not v.isalpha()]
        if invalid_vars:
            return False

        # Check if number of equations is reasonable relative to variables
        num_equations = len(equations)
        if num_equations < self.min_equations or num_equations > self.max_equations:
            return False

        # For simple problems, expect 1-2 variables; for complex, more
        problem_complexity = self._estimate_problem_complexity(problem_text)
        expected_vars = 1 if problem_complexity == 'simple' else 2 if problem_complexity == 'medium' else 3

        if len(variables) > expected_vars + 1:  # Allow some flexibility
            return False

        return True

    def _estimate_problem_complexity(self, problem_text: str) -> str:
        """Estimate problem complexity based on text features"""
        text_lower = problem_text.lower()

        # Simple indicators
        if any(word in text_lower for word in ['sum', 'difference', 'twice', 'half']):
            return 'simple'

        # Medium indicators
        if any(word in text_lower for word in ['ratio', 'percentage', 'age', 'speed']):
            return 'medium'

        # Complex indicators
        if any(word in text_lower for word in ['mixture', 'work together', 'system']):
            return 'complex'

        return 'medium'  # Default

    def _check_symbolic_consistency(self, equations: List[str]) -> Tuple[bool, str]:
        """Check symbolic consistency by attempting to solve the system"""
        if len(equations) == 0:
            return False, "No equations to check"

        try:
            # Parse equations into SymPy format
            sympy_equations = []
            variables = set()

            for eq_str in equations:
                if '=' in eq_str:
                    lhs_str, rhs_str = eq_str.split('=', 1)
                    lhs_str = lhs_str.strip()
                    rhs_str = rhs_str.strip()

                    # Parse both sides
                    lhs = parse_expr(lhs_str, transformations=_transformations)
                    rhs = parse_expr(rhs_str, transformations=_transformations)

                    # Create equation: lhs - rhs = 0
                    equation = Eq(lhs - rhs, 0)
                    sympy_equations.append(equation)

                    # Collect variables
                    variables.update(lhs.free_symbols)
                    variables.update(rhs.free_symbols)

            if not sympy_equations:
                return False, "No valid symbolic equations"

            variables = list(variables)
            if len(variables) == 0:
                return False, "No variables found in equations"

            # Try to solve the system
            try:
                solution = solve(sympy_equations, variables)

                if solution:
                    # Check if solution is consistent (not contradictory)
                    if isinstance(solution, dict):
                        # Verify solution by substitution
                        verification_passed = self._verify_solution(sympy_equations, solution)
                        if verification_passed:
                            return True, "System solved successfully"
                        else:
                            return False, "Solution verification failed"
                    else:
                        return True, "System has solution"
                else:
                    return False, "No solution found for the system"

            except Exception as e:
                return False, f"Symbolic solving failed: {str(e)}"

        except Exception as e:
            return False, f"Symbolic parsing failed: {str(e)}"

    def _verify_solution(self, equations: List[Eq], solution: Dict) -> bool:
        """Verify that the solution satisfies all equations"""
        try:
            for eq in equations:
                # Substitute solution into equation
                substituted = eq.subs(solution)
                # Simplify and check if equals zero
                simplified = simplify(substituted)
                if simplified != 0:
                    return False
            return True
        except Exception:
            return False

    def _calculate_verification_score(self, numerical_ok: bool, variable_ok: bool,
                                    symbolic_ok: bool, equations: List[str],
                                    problem_text: str) -> float:
        """Calculate overall verification score"""
        score = 0.0

        # Numerical consistency (30% weight)
        if numerical_ok:
            score += 0.3

        # Variable consistency (20% weight)
        if variable_ok:
            score += 0.2

        # Symbolic consistency (40% weight)
        if symbolic_ok:
            score += 0.4

        # Bonus for equation count appropriateness
        num_equations = len(equations)
        if 1 <= num_equations <= 5:
            score += 0.05
        elif num_equations > 5:
            score -= 0.05  # Penalty for too many equations

        # Bonus for problem complexity match
        complexity = self._estimate_problem_complexity(problem_text)
        expected_equations = {'simple': 1, 'medium': 2, 'complex': 3}.get(complexity, 2)
        if abs(num_equations - expected_equations) <= 1:
            score += 0.05

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))


# ---------------------------
# Utility functions
# ---------------------------

def verify_equations_batch(problem_texts: List[str], equations_list: List[List[str]]) -> List[Dict[str, Any]]:
    """
    Batch verification for multiple problems.

    Args:
        problem_texts: List of problem texts
        equations_list: List of equation lists (one per problem)

    Returns:
        List of verification results
    """
    verifier = VerificationModule()
    results = []

    for problem_text, equations in zip(problem_texts, equations_list):
        result = verifier.verify_equations(problem_text, equations)
        results.append(result)

    return results


def demo_verification():
    """Demo function for testing verification"""
    verifier = VerificationModule()

    # Sample problems and equations
    test_cases = [
        {
            'problem': 'John has 5 apples. He gives 2 to Mary. How many does he have left?',
            'equations': ['5 - 2 = x', 'x = 3']
        },
        {
            'problem': 'The sum of two numbers is 40 and their difference is 10.',
            'equations': ['x + y = 40', 'x - y = 10']
        },
        {
            'problem': 'A rectangle has length 10 and width 5. What is its area?',
            'equations': ['10 * 5 = area', 'area = 50']
        },
        {
            'problem': 'Invalid test case',
            'equations': []
        }
    ]

    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Problem: {case['problem']}")
        print(f"Equations: {case['equations']}")

        result = verifier.verify_equations(case['problem'], case['equations'])
        print(f"Verified: {result['verified']}")
        print(f"Score: {result['score']}")
        print(f"Diagnostics: {result['diagnostics']}")
        print(f"Numerical consistency: {result['numerical_consistency']}")
        print(f"Variable consistency: {result['variable_consistency']}")
        print(f"Symbolic consistency: {result['symbolic_consistency']}")


if __name__ == "__main__":
    demo_verification()

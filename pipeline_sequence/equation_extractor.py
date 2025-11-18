"""
Enhanced Equation Extractor for Math Word Problems
Converts natural language math problems into symbolic equations
"""

import re
from typing import List, Dict

class MathProblemParser:
    """
    Converts natural language math problems into symbolic equations
    """
    
    def __init__(self):
        self.variables_used = set()
        
    def parse_problem(self, text: str) -> Dict:
        """
        Parse a math word problem and extract equations
        """
        text_lower = text.lower().strip()
        equations = []
        
        # Reset variables for each problem
        self.variables_used = set()
        
        # Try different problem types
        equations.extend(self._parse_simple_equations(text))
        equations.extend(self._parse_age_problems(text_lower, text))
        equations.extend(self._parse_ratio_problems(text_lower, text))
        equations.extend(self._parse_percentage_problems(text_lower, text))
        equations.extend(self._parse_geometry_problems(text_lower, text))
        equations.extend(self._parse_arithmetic_problems(text_lower, text))
        equations.extend(self._parse_motion_problems(text_lower, text))
        
        # Clean and validate equations
        cleaned_equations = []
        for eq in equations:
            if self._is_valid_equation(eq):
                cleaned_equations.append(eq)
        
        return {
            'equations': cleaned_equations,
            'variables': list(self.variables_used),
            'problem_type': self._classify_problem_type(text_lower)
        }
    
    def _parse_simple_equations(self, text: str) -> List[str]:
        """Parse explicit equations in the text"""
        equations = []
        
        # Pattern: x + 5 = 20
        simple_eq = re.findall(r'([xyz]\s*[\+\-\*\/]\s*\d+\s*=\s*\d+)', text.lower())
        equations.extend(simple_eq)
        
        # Pattern: 2x + 3y = 10
        coeff_eq = re.findall(r'(\d*[xyz]\s*[\+\-\*\/]\s*\d*[xyz]\s*=\s*\d+)', text.lower())
        equations.extend(coeff_eq)
        
        # Pattern: variable = expression
        var_eq = re.findall(r'([xyz]\s*=\s*[^\.]+)', text.lower())
        equations.extend([eq.split('.')[0] for eq in var_eq])  # Remove trailing dots
        
        return equations
    
    def _parse_age_problems(self, text_lower: str, original_text: str) -> List[str]:
        """Parse age-related problems"""
        equations = []
        
        # Pattern: "A is 5 years older than B"
        older_match = re.search(r'(\w)\s+is\s+(\d+)\s+years?\s+older\s+than\s+(\w)', text_lower)
        if older_match:
            a, years, b = older_match.groups()
            equations.append(f"{a} = {b} + {years}")
            self.variables_used.update([a, b])
        
        # Pattern: "sum of ages is 50"
        sum_match = re.search(r'sum.*?(?:age|year).*?is\s*(\d+)', text_lower)
        if sum_match and len(self.variables_used) >= 2:
            total = sum_match.group(1)
            vars_list = list(self.variables_used)
            equations.append(f"{vars_list[0]} + {vars_list[1]} = {total}")
        
        return equations
    
    def _parse_ratio_problems(self, text_lower: str, original_text: str) -> List[str]:
        """Parse ratio problems"""
        equations = []
        
        # Pattern: "ratio of A to B is 3:2"
        ratio_match = re.search(r'ratio.*?(\w)\s*to\s*(\w).*?(\d+):(\d+)', text_lower)
        if ratio_match:
            a, b, num1, num2 = ratio_match.groups()
            equations.append(f"{a}/{b} = {num1}/{num2}")
            self.variables_used.update([a, b])
        
        return equations
    
    def _parse_percentage_problems(self, text_lower: str, original_text: str) -> List[str]:
        """Parse percentage problems"""
        equations = []
        
        if '%' in original_text:
            # Pattern: "25% of x is 50"
            pct_match = re.search(r'(\d+)%\s*of\s*(\w+)\s*is\s*(\d+)', text_lower)
            if pct_match:
                pct, var, value = pct_match.groups()
                equations.append(f"({pct}/100)*{var} = {value}")
                self.variables_used.add(var)
        
        return equations
    
    def _parse_geometry_problems(self, text_lower: str, original_text: str) -> List[str]:
        """Parse geometry problems"""
        equations = []
        numbers = re.findall(r'\b(\d+)\b', original_text)
        
        if 'area' in text_lower:
            if 'rectangle' in text_lower and len(numbers) >= 2:
                equations.append(f"area = {numbers[0]} * {numbers[1]}")
            elif 'circle' in text_lower and numbers:
                equations.append(f"area = 3.14 * {numbers[0]} * {numbers[0]}")
        
        elif 'perimeter' in text_lower:
            if 'square' in text_lower and numbers:
                equations.append(f"perimeter = 4 * {numbers[0]}")
        
        return equations
    
    def _parse_arithmetic_problems(self, text_lower: str, original_text: str) -> List[str]:
        """Parse basic arithmetic problems"""
        equations = []
        numbers = re.findall(r'\b(\d+)\b', original_text)
        
        if 'sum' in text_lower and len(numbers) >= 2:
            total_match = re.search(r'sum.*?is\s*(\d+)', text_lower)
            if total_match:
                equations.append(f"{numbers[0]} + {numbers[1]} = {total_match.group(1)}")
        
        elif 'product' in text_lower and len(numbers) >= 2:
            product_match = re.search(r'product.*?is\s*(\d+)', text_lower)
            if product_match:
                equations.append(f"{numbers[0]} * {numbers[1]} = {product_match.group(1)}")
        
        return equations
    
    def _parse_motion_problems(self, text_lower: str, original_text: str) -> List[str]:
        """Parse motion problems"""
        equations = []
        numbers = re.findall(r'\b(\d+)\b', original_text)
        
        if any(word in text_lower for word in ['speed', 'distance', 'time']):
            if len(numbers) >= 3:
                # Assume distance = speed * time pattern
                equations.append(f"{numbers[0]} = {numbers[1]} * {numbers[2]}")
        
        return equations
    
    def _is_valid_equation(self, equation: str) -> bool:
        """Check if equation is valid"""
        if not equation or '=' not in equation:
            return False
        
        # Basic validation - should have reasonable length and structure
        if len(equation) < 3 or len(equation) > 100:
            return False
            
        # Should contain mathematical operations or variables
        if not any(char in equation for char in ['+', '-', '*', '/', '=', 'x', 'y', 'z']):
            return False
            
        return True
    
    def _classify_problem_type(self, text_lower: str) -> str:
        """Classify the type of math problem"""
        if any(word in text_lower for word in ['year', 'age', 'old']):
            return 'age'
        elif 'ratio' in text_lower or ':' in text_lower:
            return 'ratio'
        elif '%' in text_lower or 'percent' in text_lower:
            return 'percentage'
        elif any(word in text_lower for word in ['area', 'perimeter', 'circle', 'rectangle']):
            return 'geometry'
        elif any(word in text_lower for word in ['speed', 'distance', 'time', 'km/h']):
            return 'motion'
        elif any(word in text_lower for word in ['sum', 'product', 'total']):
            return 'arithmetic'
        elif any(word in text_lower for word in ['solve', 'equation', 'x', 'y', 'z']):
            return 'algebra'
        else:
            return 'unknown'

def extract_equations_from_problem(text: str) -> List[str]:
    """
    Main function to extract equations from math word problems
    """
    parser = MathProblemParser()
    result = parser.parse_problem(text)
    return result['equations']
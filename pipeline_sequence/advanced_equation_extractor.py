"""
ADVANCED Equation Extractor for Math Word Problems
Uses multiple strategies: Pattern matching, syntactic parsing, and equation inference
"""
import math
from collections import Counter
import re
import spacy
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Try to load spaCy model, fallback to regex if not available
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except OSError:
    logger.warning("spaCy model not available. Using regex-only mode.")
    SPACY_AVAILABLE = False
    nlp = None

class AdvancedMathParser:
    """
    Advanced math problem parser using multiple strategies
    """
    
    def __init__(self):
        self.variables = set()
        self.equations = []
        self.problem_type = "unknown"
        
        # Mathematical patterns database
        self.patterns = {
            'age': [
                (r'(\w)\s+is\s+(\d+)\s+years?\s+older\s+than\s+(\w)', self._parse_age_difference),
                (r'(\w)\s+is\s+(\d+)\s+years?\s+younger\s+than\s+(\w)', self._parse_age_difference),
                (r'sum.*?ages?.*?(\d+)', self._parse_age_sum),
            ],
            'ratio': [
                (r'ratio.*?(\w)\s*to\s*(\w).*?(\d+):(\d+)', self._parse_ratio),
                (r'(\w)\s*:\s*(\w)\s*=\s*(\d+)\s*:\s*(\d+)', self._parse_ratio_direct),
            ],
            'percentage': [
                (r'(\d+)%\s*of\s*(\w+)', self._parse_percentage_of),
                (r'(\d+)%\s*(?:discount|off)', self._parse_discount),
                (r'increase.*?(\d+)%', self._parse_percentage_increase),
                (r'decrease.*?(\d+)%', self._parse_percentage_decrease),
            ],
            'geometry': [
                (r'area.*?rectangle', self._parse_rectangle_area),
                (r'area.*?circle', self._parse_circle_area),
                (r'perimeter.*?square', self._parse_square_perimeter),
                (r'perimeter.*?rectangle', self._parse_rectangle_perimeter),
                (r'volume.*?cube', self._parse_cube_volume),
            ],
            'algebra': [
                (r'([xyz])\s*[\+\-]\s*(\d+)\s*=\s*(\d+)', self._parse_simple_equation),
                (r'(\d+)([xyz])\s*[\+\-]\s*(\d+)([xyz])\s*=\s*(\d+)', self._parse_linear_equation),
                (r'solve.*?([xyz])', self._parse_solve_for),
            ],
            'motion': [
                (r'speed.*?distance.*?time', self._parse_motion_basic),
                (r'km/h.*?hours?', self._parse_speed_time),
                (r'mph.*?miles', self._parse_speed_distance),
            ],
            'mixture': [
                (r'mixture.*?(\d+)%.*?(\d+)%', self._parse_mixture_percent),
                (r'solution.*?(\d+)%', self._parse_solution_concentration),
            ],
            'work': [
                (r'work.*?together.*?hours?', self._parse_work_together),
                (r'complete.*?work.*?hours?', self._parse_work_rate),
            ]
        }
        

        # Add to advanced_equation_extractor.py
        self.enhanced_patterns = {
            # PROBABILITY & STATISTICS
            'probability': [
                (r'probability.*?(\w+).*?(\d+)/(\d+)', self._parse_probability_fraction),
                (r'P\(([^)]+)\)\s*=\s*(\d+\.?\d*)', self._parse_probability_direct),
                (r'(\d+)%\s*chance', self._parse_probability_percent),
                (r'mean.*?(\d+)', self._parse_mean),
                (r'median.*?(\d+)', self._parse_median),
                (r'standard deviation', self._parse_std_dev),
            ],
            
            # SET OPERATIONS
            'sets': [
                (r'union.*?intersection', self._parse_set_operations),
                (r'(\w+)\s*∩\s*(\w+)', self._parse_set_intersection),
                (r'(\w+)\s*∪\s*(\w+)', self._parse_set_union),
                (r'complement', self._parse_set_complement),
            ],
            
            # SEQUENCES & SERIES
            'sequences': [
                (r'arithmetic sequence', self._parse_arithmetic_sequence),
                (r'geometric sequence', self._parse_geometric_sequence),
                (r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', self._parse_sequence_terms),
                (r'common difference.*?(\d+)', self._parse_common_difference),
                (r'common ratio.*?(\d+)', self._parse_common_ratio),
            ],
            
            # COORDINATE GEOMETRY
            'coordinate_geometry': [
                (r'coordinate.*?\((\d+),(\d+)\)', self._parse_coordinates),
                (r'slope.*?(\d+)', self._parse_slope),
                (r'distance.*?points', self._parse_distance_points),
                (r'midpoint', self._parse_midpoint),
            ],
            
            # TEMPORAL LOGIC
            'temporal': [
                (r'in\s+(\d+)\s+years', self._parse_future_time),
                (r'(\d+)\s+years\s+ago', self._parse_past_time),
                (r'after\s+(\d+)\s+hours', self._parse_after_time),
                (r'before\s+(\d+)\s+hours', self._parse_before_time),
            ],
            
            # COMPARATIVE RELATIONSHIPS
            'comparative': [
                (r'twice as (?:much|many|old) as', self._parse_twice_relationship),
                (r'half of', self._parse_half_relationship),
                (r'three times', self._parse_three_times),
                (r'(\d+) times more than', self._parse_n_times_more),
                (r'(\d+) times less than', self._parse_n_times_less),
            ],
            
            # UNIT CONVERSIONS
            'units': [
                (r'km.*?m', self._parse_km_to_m),
                (r'hours.*?minutes', self._parse_hours_to_minutes),
                (r'\$.*?₹', self._parse_currency_conversion),
                (r'feet.*?meters', self._parse_feet_to_meters),
            ],
            
            # CONSTRAINT PROBLEMS
            'constraints': [
                (r'at least', self._parse_at_least),
                (r'at most', self._parse_at_most),
                (r'maximum', self._parse_maximum),
                (r'minimum', self._parse_minimum),
                (r'between.*?and', self._parse_between),
            ]
        }


        # Mathematical constants and formulas
        self.formulas = {
            'distance': 'distance = speed * time',
            'area_rectangle': 'area = length * width', 
            'area_circle': 'area = pi * radius^2',
            'perimeter_square': 'perimeter = 4 * side',
            'perimeter_rectangle': 'perimeter = 2 * (length + width)',
            'volume_cube': 'volume = side^3',
            'percentage': 'part = (percentage/100) * whole',
            'profit': 'profit = selling_price - cost_price',
            'discount': 'discount_price = original_price * (1 - discount/100)'
        }
    
    def parse_problem(self, text: str) -> Dict:
        """
        Advanced parsing with multiple strategies
        """
        self.variables = set()
        self.equations = []
        
        text_lower = text.lower().strip()
        original_text = text
        
        # Strategy 1: Direct equation extraction
        self._extract_explicit_equations(original_text)
        
        # Strategy 2: Pattern-based extraction
        self._extract_using_patterns(text_lower, original_text)
        
        # Strategy 3: Semantic parsing (if spaCy available)
        if SPACY_AVAILABLE:
            self._extract_using_semantics(original_text)
        
        # Strategy 4: Formula-based inference
        self._infer_from_formulas(text_lower, original_text)
        
        # Strategy 5: Number relationship analysis
        self._analyze_number_relationships(text_lower, original_text)
        
        # Clean and validate equations
        cleaned_equations = self._clean_equations(self.equations)
        
        # Classify problem type
        self.problem_type = self._advanced_classification(text_lower, cleaned_equations)
        
        return {
            'equations': cleaned_equations,
            'variables': list(self.variables),
            'problem_type': self.problem_type,
            'confidence': self._calculate_confidence(cleaned_equations, text_lower)
        }
    
    def _extract_explicit_equations(self, text: str):
        """Extract explicitly stated equations"""
        # Pattern 1: Simple equations like "x = 5 + 3"
        explicit_patterns = [
            r'([xyz]\s*=\s*[^\.!?]+)',  # x = expression
            r'(\w+\s*=\s*[^\.!?]+)',    # variable = expression
            r'([^\.!?]*=\s*[^\.!?]+)',  # anything = anything
        ]
        
        for pattern in explicit_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                equation = match.group(1).strip()
                if self._validate_equation(equation):
                    self.equations.append(equation)
                    # Extract variables
                    self.variables.update(re.findall(r'[a-zA-Z_][a-zA-Z_0-9]*', equation.split('=')[0].strip()))
    
    def _extract_using_patterns(self, text_lower: str, original_text: str):
        """Use predefined patterns to extract equations"""
        numbers = self._extract_numbers(original_text)
        
        for category, patterns in self.patterns.items():
            for pattern, handler in patterns:
                matches = re.search(pattern, text_lower)
                if matches:
                    try:
                        equations = handler(matches, numbers, original_text)
                        if equations:
                            self.equations.extend(equations)
                            break  # Use first matching pattern per category
                    except Exception as e:
                        logger.debug(f"Pattern handler failed: {e}")
                        continue
    
    def _extract_using_semantics(self, text: str):
        """Use NLP to understand problem structure"""
        if not SPACY_AVAILABLE:
            return
            
        doc = nlp(text)
        
        # Look for mathematical relationships
        for sent in doc.sents:
            # Check for "is" relationships (A is B)
            for token in sent:
                if token.lemma_ == "be" and token.head.pos_ == "VERB":
                    subject = self._get_subject(token)
                    complement = self._get_complement(token)
                    
                    if subject and complement and any(char.isdigit() for char in str(complement)):
                        # This might be a mathematical relationship
                        equation = f"{subject} = {complement}"
                        if self._validate_equation(equation):
                            self.equations.append(equation)
    
    def _infer_from_formulas(self, text_lower: str, original_text: str):
        """Infer equations based on known formulas"""
        numbers = self._extract_numbers(original_text)
        
        # Check which formulas might apply
        for formula_name, formula in self.formulas.items():
            if any(keyword in text_lower for keyword in formula_name.split('_')):
                # Replace generic variables with actual numbers if available
                inferred_eq = formula
                if numbers:
                    # Try to match numbers to variables
                    inferred_eq = self._instantiate_formula(formula, numbers, text_lower)
                
                if inferred_eq and self._validate_equation(inferred_eq):
                    self.equations.append(inferred_eq)
    
    def _analyze_number_relationships(self, text_lower: str, original_text: str):
        """Analyze numerical relationships in the problem"""
        numbers = self._extract_numbers(original_text)
        
        if len(numbers) >= 2:
            # Look for sum relationships
            if 'sum' in text_lower or 'total' in text_lower:
                total_match = re.search(r'sum.*?(\d+)', text_lower)
                if total_match:
                    total = total_match.group(1)
                    if numbers and len(numbers) >= 2:
                        self.equations.append(f"{numbers[0]} + {numbers[1]} = {total}")
            
            # Look for product relationships
            if 'product' in text_lower or 'times' in text_lower:
                product_match = re.search(r'product.*?(\d+)', text_lower)
                if product_match:
                    product = product_match.group(1)
                    if numbers and len(numbers) >= 2:
                        self.equations.append(f"{numbers[0]} * {numbers[1]} = {product}")
            
            # Look for difference relationships
            if 'difference' in text_lower:
                diff_match = re.search(r'difference.*?(\d+)', text_lower)
                if diff_match:
                    difference = diff_match.group(1)
                    if numbers and len(numbers) >= 2:
                        self.equations.append(f"{numbers[0]} - {numbers[1]} = {difference}")
    
    # Pattern handlers

    def _parse_ratio_direct(self, match, numbers, original_text):
        """Parse direct ratio notation A:B = C:D"""
        a, b, num1, num2 = match.groups()
        self.variables.update([a, b])
        return [f"{a}/{b} = {num1}/{num2}", f"{a} = ({num1}/{num2}) * {b}"]

    def _parse_discount(self, match, numbers, original_text):
        """Parse discount percentage problems"""
        discount_pct = match.group(1)
        # Find original price
        price_match = re.search(r'\$?(\d+)', original_text)
        if price_match:
            price = price_match.group(1)
            return [f"discount_price = {price} * (1 - {discount_pct}/100)"]
        return [f"discount_price = original_price * (1 - {discount_pct}/100)"]

    def _parse_percentage_increase(self, match, numbers, original_text):
        """Parse percentage increase problems"""
        increase_pct = match.group(1)
        # Find original value
        value_match = re.search(r'\b(\d+)\b', original_text)
        if value_match:
            value = value_match.group(1)
            return [f"new_value = {value} * (1 + {increase_pct}/100)"]
        return [f"new_value = original_value * (1 + {increase_pct}/100)"]

    def _parse_percentage_decrease(self, match, numbers, original_text):
        """Parse percentage decrease problems"""
        decrease_pct = match.group(1)
        # Find original value
        value_match = re.search(r'\b(\d+)\b', original_text)
        if value_match:
            value = value_match.group(1)
            return [f"new_value = {value} * (1 - {decrease_pct}/100)"]
        return [f"new_value = original_value * (1 - {decrease_pct}/100)"]

    def _parse_rectangle_area(self, match, numbers, original_text):
        """Parse rectangle area problems"""
        nums = self._extract_numbers(original_text)
        if len(nums) >= 2:
            return [f"area = {nums[0]} * {nums[1]}"]
        return ["area = length * width"]

    def _parse_circle_area(self, match, numbers, original_text):
        """Parse circle area problems"""
        nums = self._extract_numbers(original_text)
        if nums:
            return [f"area = 3.14 * {nums[0]} * {nums[0]}"]
        return ["area = pi * radius^2"]

    def _parse_square_perimeter(self, match, numbers, original_text):
        """Parse square perimeter problems"""
        nums = self._extract_numbers(original_text)
        if nums:
            return [f"perimeter = 4 * {nums[0]}"]
        return ["perimeter = 4 * side"]

    def _parse_rectangle_perimeter(self, match, numbers, original_text):
        """Parse rectangle perimeter problems"""
        nums = self._extract_numbers(original_text)
        if len(nums) >= 2:
            return [f"perimeter = 2 * ({nums[0]} + {nums[1]})"]
        return ["perimeter = 2 * (length + width)"]

    def _parse_cube_volume(self, match, numbers, original_text):
        """Parse cube volume problems"""
        nums = self._extract_numbers(original_text)
        if nums:
            return [f"volume = {nums[0]}^3"]
        return ["volume = side^3"]

    def _parse_solve_for(self, match, numbers, original_text):
        """Parse 'solve for x' type problems"""
        variable = match.group(1)
        self.variables.add(variable)
        # Look for equation in context
        eq_match = re.search(r'([^\.!?]*=\s*[^\.!?]+)', original_text)
        if eq_match:
            return [eq_match.group(1)]
        return [f"{variable} = ?"]

    def _parse_speed_time(self, match, numbers, original_text):
        """Parse speed-time problems"""
        nums = self._extract_numbers(original_text)
        if len(nums) >= 2:
            return [f"distance = {nums[0]} * {nums[1]}"]
        return ["distance = speed * time"]

    def _parse_speed_distance(self, match, numbers, original_text):
        """Parse speed-distance problems"""
        nums = self._extract_numbers(original_text)
        if len(nums) >= 2:
            return [f"time = {nums[1]} / {nums[0]}"]  # time = distance / speed
        return ["time = distance / speed"]

    def _parse_mixture_percent(self, match, numbers, original_text):
        """Parse mixture problems with percentages"""
        pct1, pct2 = match.groups()
        nums = self._extract_numbers(original_text)
        if len(nums) >= 1:
            total = nums[0]
            return [f"({pct1}/100)*x + ({pct2}/100)*y = ({pct1}/100)*{total}"]
        return [f"({pct1}/100)*x + ({pct2}/100)*y = final_concentration*(x+y)"]

    def _parse_solution_concentration(self, match, numbers, original_text):
        """Parse solution concentration problems"""
        concentration = match.group(1)
        return [f"amount_solute = ({concentration}/100) * total_solution"]

    def _parse_work_together(self, match, numbers, original_text):
        """Parse work together problems"""
        nums = self._extract_numbers(original_text)
        if len(nums) >= 2:
            return [f"1/{nums[0]} + 1/{nums[1]} = 1/total_time"]
        return ["1/time_a + 1/time_b = 1/total_time"]

    def _parse_work_rate(self, match, numbers, original_text):
        """Parse work rate problems"""
        nums = self._extract_numbers(original_text)
        if nums:
            return [f"work_rate = 1/{nums[0]}"]
        return ["work_rate = 1/time"]
    def _parse_age_difference(self, match, numbers, original_text):
        a, diff, b = match.groups()
        self.variables.update([a, b])
        return [f"{a} = {b} + {diff}"]
        
    def _parse_age_sum(self, match, numbers, original_text):
        total = match.group(1)
        if len(self.variables) >= 2:
            vars_list = list(self.variables)
            return [f"{vars_list[0]} + {vars_list[1]} = {total}"]
        return []
    
    def _parse_ratio(self, match, numbers, original_text):
        a, b, num1, num2 = match.groups()
        self.variables.update([a, b])
        return [f"{a}/{b} = {num1}/{num2}", f"{a} = ({num1}/{num2}) * {b}"]
    
    def _parse_percentage_of(self, match, numbers, original_text):
        pct, var = match.groups()
        self.variables.add(var)
        # Look for the result value
        result_match = re.search(r'is\s*(\d+)', original_text.lower())
        if result_match:
            return [f"({pct}/100) * {var} = {result_match.group(1)}"]
        return [f"({pct}/100) * {var}"]
    
    def _parse_simple_equation(self, match, numbers, original_text):
        var, num1, num2 = match.groups()
        self.variables.add(var)
        return [f"{var} + {num1} = {num2}"]
    
    def _parse_linear_equation(self, match, numbers, original_text):
        coeff1, var1, coeff2, var2, result = match.groups()
        self.variables.update([var1, var2])
        return [f"{coeff1}{var1} + {coeff2}{var2} = {result}"]
    
    def _parse_motion_basic(self, match, numbers, original_text):
        if len(numbers) >= 3:
            return [f"{numbers[0]} = {numbers[1]} * {numbers[2]}"]  # d = s * t
        return ["distance = speed * time"]
    
    # Helper methods
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract all numbers from text"""
        return re.findall(r'\b(\d+)\b', text)
    
    def _validate_equation(self, equation: str) -> bool:
        """Validate if an equation is mathematically plausible"""
        if not equation or '=' not in equation:
            return False
        
        parts = equation.split('=')
        if len(parts) != 2:
            return False
        
        left, right = parts[0].strip(), parts[1].strip()
        
        # Both sides should contain mathematical content
        left_math = any(c in left for c in '+*-/^()') or any(word in left for word in ['sqrt', 'pi'])
        right_math = any(c in right for c in '+*-/^()') or any(word in right for word in ['sqrt', 'pi'])
        
        if not (left_math or right_math):
            return False
        
        return True
    
    def _clean_equations(self, equations: List[str]) -> List[str]:
        """Clean and normalize equations"""
        cleaned = []
        for eq in equations:
            # Normalize whitespace
            eq = re.sub(r'\s+', ' ', eq.strip())
            # Remove trailing punctuation
            eq = re.sub(r'[\.!?,;:]$', '', eq)
            # Replace × with *
            eq = eq.replace('×', '*')
            # Replace ÷ with /
            eq = eq.replace('÷', '/')
            
            if self._validate_equation(eq):
                cleaned.append(eq)
        
        return list(set(cleaned))  # Remove duplicates
    
    def _advanced_classification(self, text_lower: str, equations: List[str]) -> str:
        """Advanced problem type classification"""
        scores = defaultdict(int)
        
        # Keyword-based scoring
        keyword_mapping = {
            'age': ['year', 'age', 'old', 'young'],
            'ratio': ['ratio', 'proportion', ':'],
            'percentage': ['%', 'percent'],
            'geometry': ['area', 'perimeter', 'volume', 'circle', 'rectangle', 'square'],
            'algebra': ['solve', 'equation', 'variable', 'x', 'y', 'z'],
            'motion': ['speed', 'distance', 'time', 'km/h', 'mph'],
            'mixture': ['mixture', 'solution', 'concentration'],
            'work': ['work', 'complete', 'together', 'rate']
        }
        
        for category, keywords in keyword_mapping.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[category] += 1
        
        # Equation-based scoring
        for eq in equations:
            if 'age' in eq or 'year' in eq:
                scores['age'] += 2
            if '/' in eq and '=' in eq and any(c in eq for c in 'xyz'):
                scores['ratio'] += 2
            if '%' in eq or '100' in eq:
                scores['percentage'] += 2
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return 'unknown'
    
    def _calculate_confidence(self, equations: List[str], text_lower: str) -> float:
        """Calculate confidence score for the extraction"""
        if not equations:
            return 0.0
        
        confidence = 0.0
        confidence += min(len(equations) * 0.2, 0.6)  # More equations = more confidence
        
        # Check if equations contain variables mentioned in text
        for eq in equations:
            if any(var in text_lower for var in self.variables):
                confidence += 0.2
        
        # Check for mathematical operations
        for eq in equations:
            if any(op in eq for op in ['+', '-', '*', '/']):
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_subject(self, token):
        """Get subject of a verb (simplified)"""
        for child in token.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                return child.text
        return None
    
    def _get_complement(self, token):
        """Get complement of a verb (simplified)"""
        for child in token.children:
            if child.dep_ in ["attr", "acomp", "dobj"]:
                return child.text
        return None
    
    def _instantiate_formula(self, formula: str, numbers: List[str], context: str) -> str:
        """Try to instantiate a generic formula with actual numbers"""
        # This is a simplified version - in production you'd want more sophisticated matching
        if len(numbers) >= 2:
            return formula.replace('length', numbers[0]).replace('width', numbers[1])
        return formula
    
    def _advanced_semantic_analysis(self, text: str):
        """Advanced NLP-based analysis"""
        if not self.SPACY_AVAILABLE:
            return
            
        doc = self.nlp(text)
        
        # Extract mathematical relationships using dependency parsing
        for sent in doc.sents:
            self._analyze_dependencies(sent)
            self._extract_comparatives(sent)
            self._extract_mathematical_relations(sent)
    
    # STRATEGY 5: Mathematical Inference
    def _mathematical_inference(self, text_lower: str, original_text: str):
        """Infer equations from mathematical context"""
        # Infer from number relationships
        if len(self.numbers) >= 2:
            self._infer_from_number_relationships(text_lower)
        
        # Infer from mathematical keywords
        self._infer_from_mathematical_context(text_lower)
    
    # STRATEGY 6: Contextual Variable Assignment
    def _contextual_variable_assignment(self, text_lower: str, original_text: str):
        """Assign variables based on context"""
        # Common variable assignments
        variable_assignments = {
            'speed': 's', 'distance': 'd', 'time': 't',
            'length': 'l', 'width': 'w', 'height': 'h',
            'radius': 'r', 'area': 'a', 'volume': 'v',
            'age': 'a', 'price': 'p', 'cost': 'c'
        }
        
        for word, var in variable_assignments.items():
            if word in text_lower:
                self.variables.add(var)
    
    # STRATEGY 7: Constraint Analysis
    def _constraint_analysis(self, text_lower: str, original_text: str):
        """Analyze constraints and boundaries"""
        constraints = {
            'at least': '>=',
            'at most': '<=', 
            'maximum': '<=',
            'minimum': '>=',
            'greater than': '>',
            'less than': '<',
            'between': '<= and >='
        }
        
        for constraint, operator in constraints.items():
            if constraint in text_lower:
                # Find the number associated with constraint
                num_match = re.search(rf'{constraint}.*?(\d+)', text_lower)
                if num_match and self.variables:
                    var = list(self.variables)[0]
                    self.equations.append(f"{var} {operator} {num_match.group(1)}")
    
    # NEW PATTERN HANDLERS (50+ additional handlers)
    def _parse_temporal_past(self, text_lower: str, original_text: str):
        years = re.search(r'(\d+)\s+years?\s+ago', text_lower).group(1)
        if self.variables:
            vars_list = list(self.variables)
            return [f"{vars_list[0]}_past = {vars_list[0]} - {years}"]
        return []
    
    def _parse_temporal_future(self, text_lower: str, original_text: str):
        years = re.search(r'in\s+(\d+)\s+years?', text_lower).group(1)
        if self.variables:
            vars_list = list(self.variables)
            return [f"{vars_list[0]}_future = {vars_list[0]} + {years}"]
        return []
    
    def _parse_twice(self, text_lower: str, original_text: str):
        return ["2*x"]  # Simplified - would need context
    
    def _parse_triple(self, text_lower: str, original_text: str):
        return ["3*x"]
    
    def _parse_half(self, text_lower: str, original_text: str):
        return ["x/2"]
    
    def _parse_sum_difference(self, text_lower: str, original_text: str):
        sum_val = re.search(r'sum.*?(\d+)', text_lower).group(1)
        diff_val = re.search(r'difference.*?(\d+)', text_lower).group(1)
        return [f"x + y = {sum_val}", f"x - y = {diff_val}"]
    
    def _parse_consecutive_numbers(self, text_lower: str, original_text: str):
        return ["y = x + 1", "z = x + 2"]
    
    # [Add 40+ more pattern handlers following similar structure]
    
    # ADVANCED HELPER METHODS
    def _extract_all_numbers(self, text: str) -> List[str]:
        """Extract all numbers including decimals"""
        return re.findall(r'\b\d+(?:\.\d+)?\b', text)
    
    def _advanced_validation(self, equation: str) -> bool:
        """Advanced equation validation"""
        if not equation or '=' not in equation:
            return False
        
        try:
            # Try to parse with sympy for validation
            parts = equation.split('=')
            if len(parts) != 2:
                return False
            
            left, right = parts[0].strip(), parts[1].strip()
            
            # Basic mathematical content check
            math_indicators = ['+', '-', '*', '/', '^', '(', ')', 'x', 'y', 'z']
            has_math = any(indicator in left or indicator in right for indicator in math_indicators)
            
            return has_math and len(equation) <= 200
            
        except:
            return False
    
    def _advanced_cleaning(self, equations: List[str]) -> List[str]:
        """Advanced equation cleaning and normalization"""
        cleaned = []
        for eq in equations:
            eq = self._normalize_equation(eq)
            if self._advanced_validation(eq):
                cleaned.append(eq)
        return list(set(cleaned))
    
    def _normalize_equation(self, equation: str) -> str:
        """Normalize equation format"""
        eq = re.sub(r'\s+', ' ', equation.strip())
        eq = re.sub(r'[\.!?,;:]$', '', eq)
        eq = eq.replace('×', '*').replace('÷', '/').replace('^', '**')
        return eq
    
    def _extract_variables_from_equation(self, equation: str):
        """Extract variables from equation"""
        vars_found = re.findall(r'[a-zA-Z_][a-zA-Z_0-9]*', equation.split('=')[0])
        self.variables.update(vars_found)
    
    def _ultimate_classification(self, text_lower: str, equations: List[str]) -> str:
        """Ultimate problem classification"""
        # Enhanced classification with pattern matching
        category_scores = Counter()
        
        for category, patterns in self.patterns.items():
            for pattern, _ in patterns:
                if re.search(pattern, text_lower):
                    category_scores[category] += 2
        
        # Boost scores based on equations
        for eq in equations:
            if 'age' in eq: category_scores['age'] += 1
            if 'ratio' in eq: category_scores['ratio'] += 1
            if '%' in eq: category_scores['percentage'] += 1
            if 'area' in eq: category_scores['geometry'] += 1
        
        return category_scores.most_common(1)[0][0] if category_scores else 'unknown'
    
    def _advanced_confidence(self, equations: List[str], text_lower: str) -> float:
        """Advanced confidence scoring"""
        if not equations:
            return 0.0
        
        confidence = 0.0
        
        # Equation count factor
        confidence += min(len(equations) * 0.15, 0.45)
        
        # Variable usage factor
        var_usage = sum(1 for eq in equations if any(var in eq for var in self.variables))
        confidence += min(var_usage * 0.1, 0.2)
        
        # Mathematical complexity factor
        math_ops = sum(1 for eq in equations if any(op in eq for op in ['+', '-', '*', '/']))
        confidence += min(math_ops * 0.05, 0.15)
        
        # Pattern match factor
        pattern_matches = sum(1 for patterns in self.patterns.values() 
                            for pattern, _ in patterns if re.search(pattern, text_lower))
        confidence += min(pattern_matches * 0.1, 0.2)
        
        return min(confidence, 1.0)
    
    # AGE PATTERNS
    def _parse_age_difference(self, text_lower: str, original_text: str) -> List[str]:
        """Parse age difference patterns like 'A is X years older/younger than B'"""
        equations = []
        # Older than pattern
        older_matches = re.finditer(r'(\w)\s+is\s+(\d+)\s+years?\s+older\s+than\s+(\w)', text_lower)
        for match in older_matches:
            a, diff, b = match.groups()
            self.variables.update([a, b])
            equations.append(f"{a} = {b} + {diff}")
        
        # Younger than pattern
        younger_matches = re.finditer(r'(\w)\s+is\s+(\d+)\s+years?\s+younger\s+than\s+(\w)', text_lower)
        for match in younger_matches:
            a, diff, b = match.groups()
            self.variables.update([a, b])
            equations.append(f"{a} = {b} - {diff}")
        
        return equations

    def _parse_age_sum(self, text_lower: str, original_text: str) -> List[str]:
        """Parse age sum patterns"""
        sum_match = re.search(r'sum.*?ages?.*?(\d+)', text_lower)
        if sum_match and len(self.variables) >= 2:
            total = sum_match.group(1)
            vars_list = list(self.variables)
            return [f"{vars_list[0]} + {vars_list[1]} = {total}"]
        return []

    def _parse_temporal_past(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'X years ago' patterns"""
        past_match = re.search(r'(\d+)\s+years?\s+ago', text_lower)
        if past_match and self.variables:
            years = past_match.group(1)
            vars_list = list(self.variables)
            equations = []
            for var in vars_list:
                equations.append(f"{var}_past = {var} - {years}")
            return equations
        return []

    def _parse_temporal_future(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'in X years' patterns"""
        future_match = re.search(r'in\s+(\d+)\s+years?', text_lower)
        if future_match and self.variables:
            years = future_match.group(1)
            vars_list = list(self.variables)
            equations = []
            for var in vars_list:
                equations.append(f"{var}_future = {var} + {years}")
            return equations
        return []

    def _parse_possessive_age(self, text_lower: str, original_text: str) -> List[str]:
        """Parse possessive age patterns like 'A's age'"""
        possessive_matches = re.finditer(r'(\w)\s*\'s\s*age', text_lower)
        equations = []
        for match in possessive_matches:
            person = match.group(1)
            self.variables.add(f"{person}_age")
            # Look for relationships involving this person's age
            if "sum" in text_lower:
                sum_match = re.search(r'sum.*?(\d+)', text_lower)
                if sum_match:
                    equations.append(f"{person}_age + other_age = {sum_match.group(1)}")
        return equations

    # RATIO & PROPORTION
    def _parse_ratio_direct(self, text_lower: str, original_text: str) -> List[str]:
        """Parse direct ratio notation A:B = C:D"""
        matches = re.finditer(r'(\w)\s*:\s*(\w)\s*=\s*(\d+)\s*:\s*(\d+)', text_lower)
        equations = []
        for match in matches:
            a, b, num1, num2 = match.groups()
            self.variables.update([a, b])
            equations.extend([
                f"{a}/{b} = {num1}/{num2}",
                f"{a} = ({num1}/{num2}) * {b}",
                f"{b} = ({num2}/{num1}) * {a}"
            ])
        return equations

    def _parse_proportion(self, text_lower: str, original_text: str) -> List[str]:
        """Parse proportion problems"""
        matches = re.finditer(r'proportion.*?(\w).*?(\w).*?(\d+).*?(\d+)', text_lower)
        equations = []
        for match in matches:
            a, b, num1, num2 = match.groups()
            self.variables.update([a, b])
            equations.append(f"{a}/{b} = {num1}/{num2}")
        return equations

    def _parse_direct_proportion(self, text_lower: str, original_text: str) -> List[str]:
        """Parse direct proportion problems"""
        numbers = self._extract_all_numbers(original_text)
        if len(numbers) >= 2:
            return [f"y = k * x", f"{numbers[1]} = k * {numbers[0]}"]
        return ["y = k * x"]

    def _parse_inverse_proportion(self, text_lower: str, original_text: str) -> List[str]:
        """Parse inverse proportion problems"""
        numbers = self._extract_all_numbers(original_text)
        if len(numbers) >= 2:
            return [f"y = k / x", f"{numbers[1]} = k / {numbers[0]}"]
        return ["y = k / x"]

    # PERCENTAGE & FRACTIONS
    def _parse_discount(self, text_lower: str, original_text: str) -> List[str]:
        """Parse discount problems"""
        discount_match = re.search(r'(\d+)%\s*(?:discount|off|reduction)', text_lower)
        if discount_match:
            discount_pct = discount_match.group(1)
            # Find price
            price_match = re.search(r'\$?(\d+(?:\.\d+)?)', original_text)
            if price_match:
                price = price_match.group(1)
                return [
                    f"discount_amount = {price} * {discount_pct}/100",
                    f"final_price = {price} - discount_amount"
                ]
            return [
                f"discount_amount = original_price * {discount_pct}/100",
                f"final_price = original_price - discount_amount"
            ]
        return []

    def _parse_percentage_increase(self, text_lower: str, original_text: str) -> List[str]:
        """Parse percentage increase problems"""
        increase_match = re.search(r'(\d+)%\s*(?:increase|growth|profit)', text_lower)
        if increase_match:
            increase_pct = increase_match.group(1)
            # Find original value
            value_match = re.search(r'\b(\d+(?:\.\d+)?)\b', original_text)
            if value_match:
                value = value_match.group(1)
                return [f"new_value = {value} * (1 + {increase_pct}/100)"]
            return [f"new_value = original_value * (1 + {increase_pct}/100)"]
        return []

    def _parse_percentage_decrease(self, text_lower: str, original_text: str) -> List[str]:
        """Parse percentage decrease problems"""
        decrease_match = re.search(r'(\d+)%\s*(?:decrease|loss|depreciation)', text_lower)
        if decrease_match:
            decrease_pct = decrease_match.group(1)
            # Find original value
            value_match = re.search(r'\b(\d+(?:\.\d+)?)\b', original_text)
            if value_match:
                value = value_match.group(1)
                return [f"new_value = {value} * (1 - {decrease_pct}/100)"]
            return [f"new_value = original_value * (1 - {decrease_pct}/100)"]
        return []

    def _parse_increase_by(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'increased by X%' patterns"""
        increase_match = re.search(r'(\w+)\s*increased\s*by\s*(\d+)%', text_lower)
        if increase_match:
            variable, pct = increase_match.groups()
            self.variables.add(variable)
            return [f"new_{variable} = {variable} * (1 + {pct}/100)"]
        return []

    def _parse_decrease_by(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'decreased by X%' patterns"""
        decrease_match = re.search(r'(\w+)\s*decreased\s*by\s*(\d+)%', text_lower)
        if decrease_match:
            variable, pct = decrease_match.groups()
            self.variables.add(variable)
            return [f"new_{variable} = {variable} * (1 - {pct}/100)"]
        return []

    def _parse_fractions(self, text_lower: str, original_text: str) -> List[str]:
        """Parse fraction patterns like 'one-half', 'one-third'"""
        fraction_map = {
            'half': '1/2', 'third': '1/3', 'fourth': '1/4', 
            'quarter': '1/4', 'fifth': '1/5'
        }
        
        fraction_match = re.search(r'one-?(half|third|fourth|quarter|fifth)', text_lower)
        if fraction_match:
            fraction = fraction_match.group(1)
            fraction_value = fraction_map[fraction]
            # Look for what the fraction is of
            of_match = re.search(r'one-?(?:half|third|fourth|quarter|fifth)\s+of\s+(\w+)', text_lower)
            if of_match:
                variable = of_match.group(1)
                self.variables.add(variable)
                return [f"result = {fraction_value} * {variable}"]
        return []

    def _parse_fraction_of(self, text_lower: str, original_text: str) -> List[str]:
        """Parse fraction patterns like '2/3 of'"""
        fraction_match = re.search(r'(\d+)\s*/\s*(\d+)\s*of', text_lower)
        if fraction_match:
            num, den = fraction_match.groups()
            # Look for the variable
            var_match = re.search(r'of\s+(\w+)', text_lower)
            if var_match:
                variable = var_match.group(1)
                self.variables.add(variable)
                return [f"result = ({num}/{den}) * {variable}"]
        return []

    # GEOMETRY
    def _parse_triangle_area(self, text_lower: str, original_text: str) -> List[str]:
        """Parse triangle area problems"""
        numbers = self._extract_all_numbers(original_text)
        if len(numbers) >= 2:
            return [f"area = (1/2) * {numbers[0]} * {numbers[1]}"]  # base * height
        return ["area = (1/2) * base * height"]

    def _parse_cylinder_volume(self, text_lower: str, original_text: str) -> List[str]:
        """Parse cylinder volume problems"""
        numbers = self._extract_all_numbers(original_text)
        if len(numbers) >= 2:
            return [f"volume = 3.14 * {numbers[0]}^2 * {numbers[1]}"]  # πr²h
        return ["volume = pi * radius^2 * height"]

    def _parse_circumference(self, text_lower: str, original_text: str) -> List[str]:
        """Parse circumference problems"""
        numbers = self._extract_all_numbers(original_text)
        if numbers:
            return [f"circumference = 2 * 3.14 * {numbers[0]}"]  # 2πr
        return ["circumference = 2 * pi * radius"]

    def _parse_diameter_radius(self, text_lower: str, original_text: str) -> List[str]:
        """Parse diameter-radius relationships"""
        numbers = self._extract_all_numbers(original_text)
        if numbers:
            # If diameter is mentioned, derive radius and vice versa
            if 'diameter' in text_lower:
                return [f"radius = {numbers[0]} / 2", f"diameter = {numbers[0]}"]
            elif 'radius' in text_lower:
                return [f"diameter = 2 * {numbers[0]}", f"radius = {numbers[0]}"]
        return ["diameter = 2 * radius"]

    # ALGEBRAIC
    def _parse_find_variable(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'find X' patterns"""
        find_matches = re.finditer(r'find.*?([xyz])', text_lower)
        equations = []
        for match in find_matches:
            variable = match.group(1)
            self.variables.add(variable)
            # Look for equation context
            eq_context = re.search(r'([^\.!?]*=\s*[^\.!?]+)', original_text)
            if eq_context:
                equations.append(eq_context.group(1))
        return equations

    def _parse_find_value(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'value of X' patterns"""
        value_matches = re.finditer(r'value.*?([xyz])', text_lower)
        equations = []
        for match in value_matches:
            variable = match.group(1)
            self.variables.add(variable)
            # Look for related equations
            if 'sum' in text_lower and 'difference' in text_lower:
                sum_match = re.search(r'sum.*?(\d+)', text_lower)
                diff_match = re.search(r'difference.*?(\d+)', text_lower)
                if sum_match and diff_match:
                    equations.extend([
                        f"x + y = {sum_match.group(1)}",
                        f"x - y = {diff_match.group(1)}"
                    ])
        return equations

    def _parse_equation_mention(self, text_lower: str, original_text: str) -> List[str]:
        """Parse when equation is explicitly mentioned"""
        if 'equation' in text_lower:
            # Look for mathematical expressions around the word 'equation'
            context = re.search(r'([^\.!?]*equation[^\.!?]*)', original_text, re.IGNORECASE)
            if context:
                # Extract potential equations from the context
                potential_eqs = re.findall(r'([^\.!?]*=\s*[^\.!?]+)', context.group(1))
                return [eq for eq in potential_eqs if self._advanced_validation(eq)]
        return []

    # MOTION & RATE
    def _parse_speed_distance(self, text_lower: str, original_text: str) -> List[str]:
        """Parse speed-distance problems"""
        numbers = self._extract_all_numbers(original_text)
        if len(numbers) >= 2:
            # Assume format: speed distance
            return [
                f"time = {numbers[1]} / {numbers[0]}",
                f"distance = {numbers[1]}",
                f"speed = {numbers[0]}"
            ]
        return ["time = distance / speed"]

    def _parse_work_rate(self, text_lower: str, original_text: str) -> List[str]:
        """Parse work rate problems"""
        numbers = self._extract_all_numbers(original_text)
        if numbers:
            return [f"work_rate = 1/{numbers[0]}"]  # work per time unit
        return ["work_rate = 1/time"]

    def _parse_current_problems(self, text_lower: str, original_text: str) -> List[str]:
        """Parse current/stream problems"""
        numbers = self._extract_all_numbers(original_text)
        if len(numbers) >= 2:
            # Assume: boat speed and current speed
            return [
                f"downstream_speed = {numbers[0]} + current_speed",
                f"upstream_speed = {numbers[0]} - current_speed"
            ]
        return [
            "downstream_speed = boat_speed + current_speed",
            "upstream_speed = boat_speed - current_speed"
        ]

    # MIXTURE & COMBINATION
    def _parse_blend_percent(self, text_lower: str, original_text: str) -> List[str]:
        """Parse blend percentage problems"""
        blend_match = re.search(r'blend.*?(\d+)%', text_lower)
        if blend_match:
            concentration = blend_match.group(1)
            numbers = self._extract_all_numbers(original_text)
            if numbers:
                return [f"final_concentration = {concentration}"]
            return ["c1*v1 + c2*v2 = cf*(v1+v2)"]
        return []

    def _parse_combination(self, text_lower: str, original_text: str) -> List[str]:
        """Parse combination problems"""
        if 'combine' in text_lower or 'mix' in text_lower:
            numbers = self._extract_all_numbers(original_text)
            concentrations = re.findall(r'(\d+)%', original_text)
            
            if len(concentrations) >= 2 and len(numbers) >= 1:
                return [
                    f"({concentrations[0]}/100)*x + ({concentrations[1]}/100)*y = ({concentrations[0]}/100)*{numbers[0]}"
                ]
            elif len(concentrations) >= 2:
                return [
                    f"({concentrations[0]}/100)*v1 + ({concentrations[1]}/100)*v2 = cf*(v1+v2)"
                ]
        return []

    # NUMBER RELATIONSHIPS
    def _parse_product_sum(self, text_lower: str, original_text: str) -> List[str]:
        """Parse product and sum relationships"""
        product_match = re.search(r'product.*?(\d+)', text_lower)
        sum_match = re.search(r'sum.*?(\d+)', text_lower)
        
        if product_match and sum_match:
            product = product_match.group(1)
            total = sum_match.group(1)
            return [
                f"x * y = {product}",
                f"x + y = {total}"
            ]
        return []

    def _parse_consecutive_numbers(self, text_lower: str, original_text: str) -> List[str]:
        """Parse consecutive number problems"""
        numbers = self._extract_all_numbers(original_text)
        equations = []
        
        if 'consecutive' in text_lower:
            if 'integer' in text_lower or 'number' in text_lower:
                if numbers:
                    # If a specific number is mentioned
                    equations.extend([
                        "y = x + 1",
                        "z = x + 2"
                    ])
                else:
                    equations.extend([
                        "y = x + 1",
                        "z = x + 2"
                    ])
                
                # Add sum equation if mentioned
                sum_match = re.search(r'sum.*?(\d+)', text_lower)
                if sum_match:
                    equations.append(f"x + y + z = {sum_match.group(1)}")
        
        return equations

    def _parse_even_odd(self, text_lower: str, original_text: str) -> List[str]:
        """Parse even/odd number problems"""
        equations = []
        numbers = self._extract_all_numbers(original_text)
        
        if 'even' in text_lower and 'odd' in text_lower:
            equations.extend([
                "x = 2*n",      # even number
                "y = 2*m + 1"   # odd number
            ])
        
        # Add sum/difference if mentioned
        sum_match = re.search(r'sum.*?(\d+)', text_lower)
        if sum_match:
            equations.append(f"x + y = {sum_match.group(1)}")
        
        diff_match = re.search(r'difference.*?(\d+)', text_lower)
        if diff_match:
            equations.append(f"x - y = {diff_match.group(1)}")
        
        return equations

    def _parse_digit_problems(self, text_lower: str, original_text: str) -> List[str]:
        """Parse digit-based number problems"""
        if 'digit' in text_lower:
            equations = []
            
            # Two-digit number: 10a + b
            if 'two-digit' in text_lower or '2-digit' in text_lower:
                equations.extend([
                    "number = 10*a + b",
                    "a >= 1", "a <= 9", "b >= 0", "b <= 9"
                ])
            
            # Sum of digits
            sum_match = re.search(r'sum.*?digits.*?(\d+)', text_lower)
            if sum_match:
                equations.append(f"a + b = {sum_match.group(1)}")
            
            # Reverse number
            if 'reverse' in text_lower:
                equations.append("reverse_number = 10*b + a")
            
            return equations
        
        return []

    # COMPARATIVE & MULTIPLICATIVE
    def _parse_twice(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'twice' or 'double' patterns"""
        if 'twice' in text_lower or 'double' in text_lower:
            # Look for what is being doubled
            context = re.search(r'(twice|double).*?(\w+)', text_lower)
            if context:
                variable = context.group(2)
                self.variables.add(variable)
                return [f"result = 2 * {variable}"]
        return []

    def _parse_triple(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'three times' or 'triple' patterns"""
        if 'three times' in text_lower or 'triple' in text_lower:
            context = re.search(r'(three times|triple).*?(\w+)', text_lower)
            if context:
                variable = context.group(2)
                self.variables.add(variable)
                return [f"result = 3 * {variable}"]
        return []

    def _parse_times(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'X times' patterns"""
        times_match = re.search(r'(\d+)\s*times', text_lower)
        if times_match:
            multiplier = times_match.group(1)
            # Look for what is being multiplied
            context = re.search(r'(\d+)\s*times.*?(\w+)', text_lower)
            if context:
                variable = context.group(2)
                self.variables.add(variable)
                return [f"result = {multiplier} * {variable}"]
        return []

    def _parse_more_than(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'more than' patterns"""
        more_match = re.search(r'(\d+)\s*more\s*than', text_lower)
        if more_match:
            amount = more_match.group(1)
            # Look for the comparison
            context = re.search(r'(\w+)\s*is\s*(\d+)\s*more\s*than\s*(\w+)', text_lower)
            if context:
                a, amount, b = context.groups()
                self.variables.update([a, b])
                return [f"{a} = {b} + {amount}"]
        return []

    def _parse_less_than(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'less than' patterns"""
        less_match = re.search(r'(\d+)\s*less\s*than', text_lower)
        if less_match:
            amount = less_match.group(1)
            context = re.search(r'(\w+)\s*is\s*(\d+)\s*less\s*than\s*(\w+)', text_lower)
            if context:
                a, amount, b = context.groups()
                self.variables.update([a, b])
                return [f"{a} = {b} - {amount}"]
        return []

    def _parse_greater_than(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'greater than' patterns"""
        if 'greater than' in text_lower:
            # Look for inequality context
            num_match = re.search(r'greater than\s*(\d+)', text_lower)
            if num_match and self.variables:
                var = list(self.variables)[0]
                return [f"{var} > {num_match.group(1)}"]
        return []

    # FINANCIAL
    def _parse_interest(self, text_lower: str, original_text: str) -> List[str]:
        """Parse interest problems"""
        numbers = self._extract_all_numbers(original_text)
        rate_match = re.search(r'(\d+)%\s*interest', text_lower)
        
        if rate_match and numbers:
            rate = rate_match.group(1)
            if len(numbers) >= 2:
                # Assume: principal and time
                return [
                    f"interest = {numbers[0]} * {rate}/100 * {numbers[1]}",
                    f"amount = {numbers[0]} + interest"
                ]
            elif numbers:
                return [
                    f"interest = principal * {rate}/100 * time",
                    f"amount = principal + interest"
                ]
        
        return ["amount = principal * (1 + rate/100)^time"]

    def _parse_profit_loss(self, text_lower: str, original_text: str) -> List[str]:
        """Parse profit/loss problems"""
        numbers = self._extract_all_numbers(original_text)
        
        if 'profit' in text_lower:
            profit_match = re.search(r'profit.*?(\d+)', text_lower)
            if profit_match and numbers:
                profit = profit_match.group(1)
                return [
                    f"profit = selling_price - cost_price",
                    f"profit = {profit}"
                ]
        
        if 'loss' in text_lower:
            loss_match = re.search(r'loss.*?(\d+)', text_lower)
            if loss_match and numbers:
                loss = loss_match.group(1)
                return [
                    f"loss = cost_price - selling_price",
                    f"loss = {loss}"
                ]
        
        return []

    def _parse_cost_price(self, text_lower: str, original_text: str) -> List[str]:
        """Parse cost price problems"""
        numbers = self._extract_all_numbers(original_text)
        if numbers:
            return [f"cost_price = {numbers[0]}"]
        return ["cost_price = ..."]

    def _parse_selling_price(self, text_lower: str, original_text: str) -> List[str]:
        """Parse selling price problems"""
        numbers = self._extract_all_numbers(original_text)
        if numbers:
            return [f"selling_price = {numbers[0]}"]
        return ["selling_price = ..."]

    def _parse_discount_price(self, text_lower: str, original_text: str) -> List[str]:
        """Parse discount price problems"""
        numbers = self._extract_all_numbers(original_text)
        discount_match = re.search(r'(\d+)%\s*discount', text_lower)
        
        if discount_match and numbers:
            discount = discount_match.group(1)
            return [
                f"discount_amount = {numbers[0]} * {discount}/100",
                f"final_price = {numbers[0]} - discount_amount"
            ]
        
        return []
    def _parse_ratio(self, text_lower: str, original_text: str) -> List[str]:
        equations = []
        
        # Pattern 1: Ratio A to B is X:Y
        ratio_matches = re.finditer(r'ratio.*?(\w)\s*to\s*(\w).*?(\d+):(\d+)', text_lower)
        for match in ratio_matches:
            a, b, num1, num2 = match.groups()
            self.variables.update([a, b])
            
            # Generate multiple ratio equations
            equations.extend([
                f"{a}/{b} = {num1}/{num2}",
                f"{a} = ({num1}/{num2}) * {b}",
                f"{b} = ({num2}/{num1}) * {a}",
                f"{num2} * {a} = {num1} * {b}"  # Cross multiplication
            ])
            
            # Look for total sum in context
            total_match = re.search(r'total.*?(\d+)', text_lower)
            if total_match:
                total = total_match.group(1)
                equations.extend([
                    f"{a} + {b} = {total}",
                    f"{a} = ({num1}/({num1}+{num2})) * {total}",
                    f"{b} = ({num2}/({num1}+{num2})) * {total}"
                ])
        
        return equations

    def _parse_percentage_of(self, text_lower: str, original_text: str) -> List[str]:
        """Advanced percentage parsing with context-aware equation generation"""
        equations = []
        
        # Find all percentage patterns
        pct_matches = re.finditer(r'(\d+)%\s*of\s*(\w+)', text_lower)
        for match in pct_matches:
            pct, var = match.groups()
            self.variables.add(var)
            
            # Look for result value or equality
            result_match = re.search(rf'{re.escape(var)}.*?is.*?(\d+)', text_lower)
            if result_match:
                # Pattern: "X% of Y is Z"
                result = result_match.group(1)
                equations.append(f"({pct}/100) * {var} = {result}")
            else:
                # Pattern: "Find X% of Y"
                equations.append(f"result = ({pct}/100) * {var}")
            
            # Look for complementary percentages
            if 'remaining' in text_lower or 'left' in text_lower:
                remaining_pct = 100 - int(pct)
                equations.append(f"remaining_{var} = ({remaining_pct}/100) * {var}")
        
        # Handle percentage increase/decrease context
        if 'increase' in text_lower and equations:
            for eq in equations[:]:  # Copy to avoid modification during iteration
                if 'result' in eq:
                    equations.append(eq.replace('result', 'increased_value'))
        
        return equations

    def _parse_rectangle_area(self, text_lower: str, original_text: str) -> List[str]:
        """Advanced rectangle area parsing with dimension extraction"""
        equations = []
        numbers = self._extract_all_numbers(original_text)
        
        # Extract dimensions using multiple strategies
        length_match = re.search(r'length\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        width_match = re.search(r'width\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        area_match = re.search(r'area\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        
        if length_match and width_match:
            l, w = length_match.group(1), width_match.group(1)
            equations.append(f"area = {l} * {w}")
            
            # Also add perimeter if mentioned
            if 'perimeter' in text_lower:
                equations.append(f"perimeter = 2 * ({l} + {w})")
        
        elif area_match and (length_match or width_match):
            area = area_match.group(1)
            if length_match:
                l = length_match.group(1)
                equations.extend([
                    f"area = {area}",
                    f"width = {area} / {l}"
                ])
            elif width_match:
                w = width_match.group(1)
                equations.extend([
                    f"area = {area}",
                    f"length = {area} / {w}"
                ])
        
        elif len(numbers) >= 2:
            # Assume first two numbers are dimensions
            equations.append(f"area = {numbers[0]} * {numbers[1]}")
        
        else:
            equations.extend([
                "area = length * width",
                "perimeter = 2 * (length + width)"
            ])
        
        return equations

    def _parse_circle_area(self, text_lower: str, original_text: str) -> List[str]:
        """Advanced circle geometry parsing"""
        equations = []
        numbers = self._extract_all_numbers(original_text)
        
        # Extract radius/diameter
        radius_match = re.search(r'radius\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        diameter_match = re.search(r'diameter\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        area_match = re.search(r'area\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        circum_match = re.search(r'circumference\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        
        if radius_match:
            r = radius_match.group(1)
            equations.extend([
                f"area = 3.14159 * {r}^2",
                f"circumference = 2 * 3.14159 * {r}",
                f"diameter = 2 * {r}"
            ])
        elif diameter_match:
            d = diameter_match.group(1)
            r = f"({d}/2)"
            equations.extend([
                f"radius = {r}",
                f"area = 3.14159 * {r}^2",
                f"circumference = 3.14159 * {d}"
            ])
        elif area_match:
            area = area_match.group(1)
            equations.extend([
                f"area = {area}",
                f"radius = sqrt({area} / 3.14159)",
                f"diameter = 2 * sqrt({area} / 3.14159)"
            ])
        elif circum_match:
            circum = circum_match.group(1)
            equations.extend([
                f"circumference = {circum}",
                f"radius = {circum} / (2 * 3.14159)",
                f"diameter = {circum} / 3.14159"
            ])
        elif numbers:
            # Assume first number is radius
            equations.append(f"area = 3.14159 * {numbers[0]}^2")
        
        return equations

    def _parse_square_perimeter(self, text_lower: str, original_text: str) -> List[str]:
        """Advanced square geometry parsing"""
        equations = []
        numbers = self._extract_all_numbers(original_text)
        
        side_match = re.search(r'side\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        perimeter_match = re.search(r'perimeter\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        area_match = re.search(r'area\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        
        if side_match:
            s = side_match.group(1)
            equations.extend([
                f"perimeter = 4 * {s}",
                f"area = {s}^2",
                f"diagonal = {s} * sqrt(2)"
            ])
        elif perimeter_match:
            p = perimeter_match.group(1)
            s = f"({p}/4)"
            equations.extend([
                f"perimeter = {p}",
                f"side = {s}",
                f"area = {s}^2"
            ])
        elif area_match:
            area = area_match.group(1)
            s = f"sqrt({area})"
            equations.extend([
                f"area = {area}",
                f"side = {s}",
                f"perimeter = 4 * {s}"
            ])
        elif numbers:
            equations.extend([
                f"perimeter = 4 * {numbers[0]}",
                f"area = {numbers[0]}^2"
            ])
        
        return equations

    def _parse_rectangle_perimeter(self, text_lower: str, original_text: str) -> List[str]:
        """Advanced rectangle perimeter parsing"""
        equations = []
        numbers = self._extract_all_numbers(original_text)
        
        length_match = re.search(r'length\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        width_match = re.search(r'width\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        perimeter_match = re.search(r'perimeter\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        
        if length_match and width_match:
            l, w = length_match.group(1), width_match.group(1)
            equations.extend([
                f"perimeter = 2 * ({l} + {w})",
                f"area = {l} * {w}"
            ])
        elif perimeter_match and (length_match or width_match):
            p = perimeter_match.group(1)
            if length_match:
                l = length_match.group(1)
                equations.extend([
                    f"perimeter = {p}",
                    f"width = ({p}/2) - {l}"
                ])
            elif width_match:
                w = width_match.group(1)
                equations.extend([
                    f"perimeter = {p}",
                    f"length = ({p}/2) - {w}"
                ])
        elif len(numbers) >= 2:
            equations.append(f"perimeter = 2 * ({numbers[0]} + {numbers[1]})")
        
        return equations

    def _parse_cube_volume(self, text_lower: str, original_text: str) -> List[str]:
        """Advanced cube volume and surface area parsing"""
        equations = []
        numbers = self._extract_all_numbers(original_text)
        
        side_match = re.search(r'side\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        volume_match = re.search(r'volume\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        surface_match = re.search(r'surface\s*area\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        
        if side_match:
            s = side_match.group(1)
            equations.extend([
                f"volume = {s}^3",
                f"surface_area = 6 * {s}^2",
                f"space_diagonal = {s} * sqrt(3)"
            ])
        elif volume_match:
            vol = volume_match.group(1)
            s = f"cbrt({vol})"
            equations.extend([
                f"volume = {vol}",
                f"side = {s}",
                f"surface_area = 6 * {s}^2"
            ])
        elif surface_match:
            surface = surface_match.group(1)
            s = f"sqrt({surface}/6)"
            equations.extend([
                f"surface_area = {surface}",
                f"side = {s}",
                f"volume = {s}^3"
            ])
        elif numbers:
            equations.extend([
                f"volume = {numbers[0]}^3",
                f"surface_area = 6 * {numbers[0]}^2"
            ])
        
        return equations

    def _parse_simple_equation(self, text_lower: str, original_text: str) -> List[str]:
        """Advanced simple equation parsing with multiple formats"""
        equations = []
        
        # Pattern: x + 5 = 20, x - 3 = 15, etc.
        simple_matches = re.finditer(r'([xyz])\s*([\+\-])\s*(\d+)\s*=\s*(\d+)', text_lower)
        for match in simple_matches:
            var, op, num1, num2 = match.groups()
            self.variables.add(var)
            
            if op == '+':
                equations.append(f"{var} + {num1} = {num2}")
            else:  # op == '-'
                equations.append(f"{var} - {num1} = {num2}")
        
        # Also handle: 5 + x = 20, 20 - x = 15
        reversed_matches = re.finditer(r'(\d+)\s*([\+\-])\s*([xyz])\s*=\s*(\d+)', text_lower)
        for match in reversed_matches:
            num1, op, var, num2 = match.groups()
            self.variables.add(var)
            
            if op == '+':
                equations.append(f"{num1} + {var} = {num2}")
            else:
                equations.append(f"{num1} - {var} = {num2}")
        
        return equations

    def _parse_linear_equation(self, text_lower: str, original_text: str) -> List[str]:
        """Advanced linear equation parsing with coefficient handling"""
        equations = []
        
        # Pattern: 2x + 3y = 15, 5x - 2y = 20
        linear_matches = re.finditer(r'(\d*)([xyz])\s*([\+\-])\s*(\d*)([xyz])\s*=\s*(\d+)', text_lower)
        for match in linear_matches:
            coeff1, var1, op, coeff2, var2, result = match.groups()
            
            # Handle missing coefficients
            coeff1 = coeff1 if coeff1 else '1'
            coeff2 = coeff2 if coeff2 else '1'
            
            self.variables.update([var1, var2])
            
            equation = f"{coeff1}{var1} {op} {coeff2}{var2} = {result}"
            equations.append(equation)
            
            # Also generate normalized form
            if op == '+':
                equations.append(f"{coeff1}{var1} + {coeff2}{var2} - {result} = 0")
            else:
                equations.append(f"{coeff1}{var1} - {coeff2}{var2} - {result} = 0")
        
        return equations

    def _parse_solve_for(self, text_lower: str, original_text: str) -> List[str]:
        """Advanced 'solve for X' parsing with context extraction"""
        equations = []
        
        solve_matches = re.finditer(r'solve.*?([xyz])', text_lower)
        for match in solve_matches:
            target_var = match.group(1)
            self.variables.add(target_var)
            
            # Look for equation in the same sentence
            sentence_match = re.search(r'([^.!?]*solve[^.!?]*)', text_lower)
            if sentence_match:
                sentence = sentence_match.group(1)
                # Extract equations from the sentence
                eq_matches = re.finditer(r'([^=]*=\s*[^=]+)', sentence)
                for eq_match in eq_matches:
                    equation = eq_match.group(1).strip()
                    if self._advanced_validation(equation):
                        equations.append(equation)
            
            # If no equation found, create a generic one
            if not equations:
                equations.append(f"find {target_var}")
                
                # Add common equation patterns based on context
                if 'sum' in text_lower and 'difference' in text_lower:
                    sum_match = re.search(r'sum.*?(\d+)', text_lower)
                    diff_match = re.search(r'difference.*?(\d+)', text_lower)
                    if sum_match and diff_match:
                        equations.extend([
                            f"x + y = {sum_match.group(1)}",
                            f"x - y = {diff_match.group(1)}"
                        ])
        
        return equations

    def _parse_motion_basic(self, text_lower: str, original_text: str) -> List[str]:
        """Advanced motion problem parsing with multiple scenarios"""
        equations = []
        numbers = self._extract_all_numbers(original_text)
        
        # Basic distance formula
        equations.append("distance = speed * time")
        
        # Extract specific values
        speed_match = re.search(r'speed\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        time_match = re.search(r'time\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        distance_match = re.search(r'distance\s*[=:]?\s*(\d+(?:\.\d+)?)', text_lower)
        
        if speed_match and time_match:
            s, t = speed_match.group(1), time_match.group(1)
            equations.append(f"distance = {s} * {t}")
        elif distance_match and (speed_match or time_match):
            d = distance_match.group(1)
            if speed_match:
                s = speed_match.group(1)
                equations.append(f"time = {d} / {s}")
            elif time_match:
                t = time_match.group(1)
                equations.append(f"speed = {d} / {t}")
        
        # Handle relative motion
        if 'faster' in text_lower or 'slower' in text_lower:
            relative_match = re.search(r'(\d+)\s*km/h\s*(?:faster|slower)', text_lower)
            if relative_match:
                diff = relative_match.group(1)
                if 'faster' in text_lower:
                    equations.append(f"speed2 = speed1 + {diff}")
                else:
                    equations.append(f"speed2 = speed1 - {diff}")
        
        return equations

    def _parse_speed_time(self, text_lower: str, original_text: str) -> List[str]:
        """Advanced speed-time parsing with unit handling"""
        equations = []
        
        # Extract speed and time values
        speed_matches = re.finditer(r'(\d+(?:\.\d+)?)\s*km/h', text_lower)
        time_matches = re.finditer(r'(\d+(?:\.\d+)?)\s*hours?', text_lower)
        
        speeds = [m.group(1) for m in speed_matches]
        times = [m.group(1) for m in time_matches]
        
        # Pair speeds with times
        for i, (speed, time) in enumerate(zip(speeds, times)):
            equations.append(f"distance_{i+1} = {speed} * {time}")
            
            # If multiple speeds/times, handle average speed
            if len(speeds) > 1 or len(times) > 1:
                total_distance = " + ".join([f"distance_{j+1}" for j in range(len(speeds))])
                total_time = " + ".join(times)
                equations.append(f"average_speed = ({total_distance}) / ({total_time})")
        
        # Handle minutes conversion
        minute_matches = re.finditer(r'(\d+)\s*minutes?', text_lower)
        for match in minute_matches:
            minutes = match.group(1)
            equations.append(f"time_in_hours = {minutes} / 60")
        
        return equations

    def _parse_work_together(self, text_lower: str, original_text: str) -> List[str]:
        """Advanced work together problems with rate calculations"""
        equations = []
        numbers = self._extract_all_numbers(original_text)
        
        # Extract individual work times
        time_matches = re.finditer(r'(\d+)\s*(?:hours?|days?)', text_lower)
        times = [m.group(1) for m in time_matches]
        
        if len(times) >= 2:
            # Calculate combined work rate
            rate_sum = " + ".join([f"1/{t}" for t in times])
            equations.append(f"combined_rate = {rate_sum}")
            equations.append("time_together = 1 / combined_rate")
            
            # Add individual rate equations
            for i, time in enumerate(times):
                equations.append(f"rate_{i+1} = 1/{time}")
        
        # Handle work completion scenarios
        if 'complete' in text_lower and 'together' in text_lower:
            if numbers and len(numbers) >= 2:
                # Assume: A takes X hours, B takes Y hours
                equations.extend([
                    f"rate_A = 1/{numbers[0]}",
                    f"rate_B = 1/{numbers[1]}",
                    f"combined_rate = 1/{numbers[0]} + 1/{numbers[1]}",
                    f"time_together = 1 / (1/{numbers[0]} + 1/{numbers[1]})"
                ])
        
        return equations

    def _parse_mixture_percent(self, text_lower: str, original_text: str) -> List[str]:
        """Advanced mixture problems with percentage concentrations"""
        equations = []
        
        # Extract all percentages
        pct_matches = re.finditer(r'(\d+)%', original_text)
        concentrations = [m.group(1) for m in pct_matches]
        
        # Extract quantities
        quantity_matches = re.finditer(r'(\d+(?:\.\d+)?)\s*(?:liters?|ml|grams?)', text_lower)
        quantities = [m.group(1) for m in quantity_matches]
        
        if len(concentrations) >= 2:
            c1, c2 = concentrations[0], concentrations[1]
            
            if len(quantities) >= 1:
                # Known total quantity
                total = quantities[0]
                equations.extend([
                    f"({c1}/100)*x + ({c2}/100)*y = ({c1}/100)*{total}",
                    f"x + y = {total}"
                ])
            else:
                # Generic mixture equation
                equations.extend([
                    f"({c1}/100)*v1 + ({c2}/100)*v2 = cf*(v1+v2)",
                    f"total_volume = v1 + v2"
                ])
            
            # Handle final concentration if mentioned
            final_match = re.search(r'(\d+)%.*?mixture', text_lower)
            if final_match:
                cf = final_match.group(1)
                equations.append(f"final_concentration = {cf}")
        
        return equations

    def _parse_solution_concentration(self, text_lower: str, original_text: str) -> List[str]:
        """Advanced solution concentration problems"""
        equations = []
        
        # Extract concentration
        conc_match = re.search(r'(\d+)%', original_text)
        if conc_match:
            concentration = conc_match.group(1)
            
            # Extract quantity if available
            quantity_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:liters?|ml)', text_lower)
            if quantity_match:
                quantity = quantity_match.group(1)
                equations.extend([
                    f"concentration = {concentration}",
                    f"total_solution = {quantity}",
                    f"amount_solute = ({concentration}/100) * {quantity}"
                ])
            else:
                equations.extend([
                    f"concentration = {concentration}",
                    f"amount_solute = ({concentration}/100) * total_solution"
                ])
        
        # Handle dilution scenarios
        if 'dilute' in text_lower or 'add water' in text_lower:
            equations.append("final_concentration = (initial_solute / (initial_volume + added_water)) * 100")
        
        return equations
    
    # PROBABILITY & STATISTICS
    def _parse_probability_fraction(self, text_lower: str, original_text: str) -> List[str]:
        """Parse probability expressed as fractions: P(event) = 3/5"""
        equations = []
        
        matches = re.finditer(r'probability.*?(\w+).*?(\d+)/(\d+)', text_lower)
        for match in matches:
            event, numerator, denominator = match.groups()
            equations.append(f"P({event}) = {numerator}/{denominator}")
            equations.append(f"P(not_{event}) = 1 - {numerator}/{denominator}")
        
        return equations

    def _parse_probability_direct(self, text_lower: str, original_text: str) -> List[str]:
        """Parse direct probability notation: P(A) = 0.25"""
        equations = []
        
        matches = re.finditer(r'P\(([^)]+)\)\s*=\s*(\d+\.?\d*)', original_text)
        for match in matches:
            event, prob = match.groups()
            equations.append(f"P({event}) = {prob}")
            
            # Add complementary probability
            if float(prob) <= 1.0:
                equations.append(f"P(not_{event}) = 1 - {prob}")
        
        return equations

    def _parse_probability_percent(self, text_lower: str, original_text: str) -> List[str]:
        """Parse probability as percentage: 25% chance"""
        equations = []
        
        matches = re.finditer(r'(\d+)%\s*chance', text_lower)
        for match in matches:
            percent = match.group(1)
            equations.append(f"probability = {percent}/100")
        
        return equations

    def _parse_mean(self, text_lower: str, original_text: str) -> List[str]:
        """Parse mean/average problems"""
        equations = []
        numbers = self._extract_all_numbers(original_text)
        
        mean_match = re.search(r'mean.*?(\d+(?:\.\d+)?)', text_lower)
        if mean_match:
            mean_val = mean_match.group(1)
            equations.append(f"mean = {mean_val}")
            
            if len(numbers) >= 2:
                # Assume numbers are data points
                sum_expr = " + ".join(numbers)
                count = len(numbers)
                equations.append(f"({sum_expr}) / {count} = {mean_val}")
        
        return equations

    def _parse_median(self, text_lower: str, original_text: str) -> List[str]:
        """Parse median problems"""
        equations = []
        numbers = self._extract_all_numbers(original_text)
        
        median_match = re.search(r'median.*?(\d+(?:\.\d+)?)', text_lower)
        if median_match:
            median_val = median_match.group(1)
            equations.append(f"median = {median_val}")
            
            if numbers:
                sorted_nums = sorted([int(n) for n in numbers])
                equations.append(f"sorted_data = {sorted_nums}")
        
        return equations

    def _parse_std_dev(self, text_lower: str, original_text: str) -> List[str]:
        """Parse standard deviation problems"""
        return [
            "variance = average of squared differences from mean",
            "standard_deviation = sqrt(variance)"
        ]

    # SET OPERATIONS
    def _parse_set_operations(self, text_lower: str, original_text: str) -> List[str]:
        """Parse set union and intersection operations"""
        equations = []
        
        # Look for set names
        set_matches = re.finditer(r'[A-Z]', original_text)
        sets = list(set([m.group() for m in set_matches if m.group().isupper()]))
        
        if len(sets) >= 2:
            if 'union' in text_lower and 'intersection' in text_lower:
                equations.extend([
                    f"n(A ∪ B) = n(A) + n(B) - n(A ∩ B)",
                    f"n({sets[0]} ∪ {sets[1]}) = n({sets[0]}) + n({sets[1]}) - n({sets[0]} ∩ {sets[1]})"
                ])
        
        return equations

    def _parse_set_intersection(self, text_lower: str, original_text: str) -> List[str]:
        """Parse set intersection notation: A ∩ B"""
        equations = []
        
        matches = re.finditer(r'(\w+)\s*∩\s*(\w+)', original_text)
        for match in matches:
            set1, set2 = match.groups()
            equations.append(f"{set1} ∩ {set2}")
            equations.append(f"n({set1} ∩ {set2})")
        
        return equations

    def _parse_set_union(self, text_lower: str, original_text: str) -> List[str]:
        """Parse set union notation: A ∪ B"""
        equations = []
        
        matches = re.finditer(r'(\w+)\s*∪\s*(\w+)', original_text)
        for match in matches:
            set1, set2 = match.groups()
            equations.append(f"{set1} ∪ {set2}")
            equations.append(f"n({set1} ∪ {set2}) = n({set1}) + n({set2}) - n({set1} ∩ {set2})")
        
        return equations

    def _parse_set_complement(self, text_lower: str, original_text: str) -> List[str]:
        """Parse set complement operations"""
        equations = []
        
        if 'complement' in text_lower:
            set_match = re.search(r'complement.*?([A-Z])', original_text)
            if set_match:
                set_name = set_match.group(1)
                equations.extend([
                    f"{set_name}' = universal_set - {set_name}",
                    f"n({set_name}') = n(universal_set) - n({set_name})"
                ])
        
        return equations

    # SEQUENCES & SERIES
    def _parse_arithmetic_sequence(self, text_lower: str, original_text: str) -> List[str]:
        """Parse arithmetic sequence problems"""
        equations = []
        numbers = self._extract_all_numbers(original_text)
        
        equations.extend([
            "a_n = a_1 + (n-1)*d",
            "S_n = n/2 * (2*a_1 + (n-1)*d)"
        ])
        
        if len(numbers) >= 2:
            # Assume first two numbers are terms
            equations.append(f"d = {numbers[1]} - {numbers[0]}")
        
        return equations

    def _parse_geometric_sequence(self, text_lower: str, original_text: str) -> List[str]:
        """Parse geometric sequence problems"""
        equations = []
        numbers = self._extract_all_numbers(original_text)
        
        equations.extend([
            "a_n = a_1 * r^(n-1)",
            "S_n = a_1 * (1 - r^n) / (1 - r)"
        ])
        
        if len(numbers) >= 2 and int(numbers[0]) != 0:
            # Assume first two numbers are terms
            equations.append(f"r = {numbers[1]} / {numbers[0]}")
        
        return equations

    def _parse_sequence_terms(self, text_lower: str, original_text: str) -> List[str]:
        """Parse sequence terms: 2, 5, 8, ..."""
        equations = []
        
        matches = re.finditer(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', original_text)
        for match in matches:
            a, b, c = match.groups()
            a_num, b_num, c_num = int(a), int(b), int(c)
            
            # Check if arithmetic
            if b_num - a_num == c_num - b_num:
                d = b_num - a_num
                equations.extend([
                    f"a_1 = {a}",
                    f"d = {d}",
                    f"a_n = {a} + (n-1)*{d}"
                ])
            # Check if geometric
            elif a_num != 0 and b_num / a_num == c_num / b_num:
                r = b_num / a_num
                equations.extend([
                    f"a_1 = {a}",
                    f"r = {r}",
                    f"a_n = {a} * {r}^(n-1)"
                ])
        
        return equations

    def _parse_common_difference(self, text_lower: str, original_text: str) -> List[str]:
        """Parse common difference in arithmetic sequences"""
        diff_match = re.search(r'common difference.*?(\d+(?:\.\d+)?)', text_lower)
        if diff_match:
            d = diff_match.group(1)
            return [f"d = {d}", f"a_n = a_1 + (n-1)*{d}"]
        return []

    def _parse_common_ratio(self, text_lower: str, original_text: str) -> List[str]:
        """Parse common ratio in geometric sequences"""
        ratio_match = re.search(r'common ratio.*?(\d+(?:\.\d+)?)', text_lower)
        if ratio_match:
            r = ratio_match.group(1)
            return [f"r = {r}", f"a_n = a_1 * {r}^(n-1)"]
        return []

    # COORDINATE GEOMETRY
    def _parse_coordinates(self, text_lower: str, original_text: str) -> List[str]:
        """Parse coordinate points: (x, y)"""
        equations = []
        
        matches = re.finditer(r'\((\d+),(\d+)\)', original_text)
        points = []
        
        for match in matches:
            x, y = match.groups()
            points.append((x, y))
            equations.append(f"point = ({x}, {y})")
        
        # If we have two points, calculate distance and slope
        if len(points) >= 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            equations.extend([
                f"distance = sqrt(({x2} - {x1})^2 + ({y2} - {y1})^2)",
                f"slope = ({y2} - {y1}) / ({x2} - {x1})"
            ])
        
        return equations

    def _parse_slope(self, text_lower: str, original_text: str) -> List[str]:
        """Parse slope problems"""
        equations = []
        
        slope_match = re.search(r'slope.*?(\d+(?:\.\d+)?)', text_lower)
        if slope_match:
            m = slope_match.group(1)
            equations.append(f"m = {m}")
            equations.append(f"y = {m}*x + b")
        
        return equations

    def _parse_distance_points(self, text_lower: str, original_text: str) -> List[str]:
        """Parse distance between points"""
        equations = []
        coord_matches = list(re.finditer(r'\((\d+),(\d+)\)', original_text))
        
        if len(coord_matches) >= 2:
            x1, y1 = coord_matches[0].groups()
            x2, y2 = coord_matches[1].groups()
            equations.append(f"distance = sqrt(({x2} - {x1})^2 + ({y2} - {y1})^2)")
        
        return equations

    def _parse_midpoint(self, text_lower: str, original_text: str) -> List[str]:
        """Parse midpoint problems"""
        equations = []
        coord_matches = list(re.finditer(r'\((\d+),(\d+)\)', original_text))
        
        if len(coord_matches) >= 2:
            x1, y1 = coord_matches[0].groups()
            x2, y2 = coord_matches[1].groups()
            equations.append(f"midpoint_x = ({x1} + {x2}) / 2")
            equations.append(f"midpoint_y = ({y1} + {y2}) / 2")
            equations.append(f"midpoint = (({x1}+{x2})/2, ({y1}+{y2})/2)")
        
        return equations

    # TEMPORAL LOGIC
    def _parse_future_time(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'in X years' temporal relationships"""
        equations = []
        
        future_match = re.search(r'in\s+(\d+)\s+years', text_lower)
        if future_match and self.variables:
            years = future_match.group(1)
            for var in list(self.variables):
                if 'age' in var.lower() or var in ['a', 'b', 'x', 'y']:
                    equations.append(f"{var}_future = {var} + {years}")
        
        return equations

    def _parse_past_time(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'X years ago' temporal relationships"""
        equations = []
        
        past_match = re.search(r'(\d+)\s+years\s+ago', text_lower)
        if past_match and self.variables:
            years = past_match.group(1)
            for var in list(self.variables):
                if 'age' in var.lower() or var in ['a', 'b', 'x', 'y']:
                    equations.append(f"{var}_past = {var} - {years}")
        
        return equations

    def _parse_after_time(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'after X hours' temporal relationships"""
        equations = []
        
        after_match = re.search(r'after\s+(\d+)\s+hours', text_lower)
        if after_match:
            hours = after_match.group(1)
            equations.append(f"time_after = current_time + {hours}")
            
            # For motion problems
            if any(word in text_lower for word in ['speed', 'distance']):
                equations.append(f"distance_after = speed * {hours}")
        
        return equations

    def _parse_before_time(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'before X hours' temporal relationships"""
        equations = []
        
        before_match = re.search(r'before\s+(\d+)\s+hours', text_lower)
        if before_match:
            hours = before_match.group(1)
            equations.append(f"time_before = current_time - {hours}")
        
        return equations

    # COMPARATIVE RELATIONSHIPS
    def _parse_twice_relationship(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'twice as much/many/old as' relationships"""
        equations = []
        
        # Extract the two quantities being compared
        match = re.search(r'twice as (?:much|many|old) as (\w+)', text_lower)
        if match:
            compared_to = match.group(1)
            # Find the subject (what is twice as much)
            subject_match = re.search(r'(\w+)\s+is twice', text_lower)
            if subject_match:
                subject = subject_match.group(1)
                equations.append(f"{subject} = 2 * {compared_to}")
            else:
                equations.append(f"quantity = 2 * {compared_to}")
        
        return equations

    def _parse_half_relationship(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'half of' relationships"""
        equations = []
        
        match = re.search(r'half of (\w+)', text_lower)
        if match:
            whole = match.group(1)
            equations.append(f"part = (1/2) * {whole}")
        
        return equations

    def _parse_three_times(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'three times' relationships"""
        equations = []
        
        match = re.search(r'three times (\w+)', text_lower)
        if match:
            base = match.group(1)
            equations.append(f"result = 3 * {base}")
        
        return equations

    def _parse_n_times_more(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'X times more than' relationships"""
        equations = []
        
        match = re.search(r'(\d+) times more than (\w+)', text_lower)
        if match:
            multiplier, base = match.groups()
            equations.append(f"result = {multiplier} * {base}")
        
        return equations

    def _parse_n_times_less(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'X times less than' relationships"""
        equations = []
        
        match = re.search(r'(\d+) times less than (\w+)', text_lower)
        if match:
            multiplier, base = match.groups()
            equations.append(f"result = {base} / {multiplier}")
        
        return equations

    # UNIT CONVERSIONS
    def _parse_km_to_m(self, text_lower: str, original_text: str) -> List[str]:
        """Parse kilometer to meter conversions"""
        equations = []
        
        km_match = re.search(r'(\d+(?:\.\d+)?)\s*km', original_text)
        if km_match:
            km = km_match.group(1)
            equations.append(f"meters = {km} * 1000")
        
        return equations

    def _parse_hours_to_minutes(self, text_lower: str, original_text: str) -> List[str]:
        """Parse hours to minutes conversions"""
        equations = []
        
        hour_match = re.search(r'(\d+(?:\.\d+)?)\s*hours?', original_text)
        if hour_match:
            hours = hour_match.group(1)
            equations.append(f"minutes = {hours} * 60")
        
        return equations

    def _parse_currency_conversion(self, text_lower: str, original_text: str) -> List[str]:
        """Parse currency conversions"""
        equations = []
        
        # Look for amounts in different currencies
        usd_match = re.search(r'\$(\d+(?:\.\d+)?)', original_text)
        inr_match = re.search(r'[₹₹](\d+(?:\.\d+)?)', original_text)
        
        if usd_match and inr_match:
            usd = usd_match.group(1)
            inr = inr_match.group(1)
            equations.append(f"exchange_rate = {inr} / {usd}")
        elif usd_match:
            usd = usd_match.group(1)
            equations.append(f"INR = {usd} * exchange_rate")
        elif inr_match:
            inr = inr_match.group(1)
            equations.append(f"USD = {inr} / exchange_rate")
        
        return equations

    def _parse_feet_to_meters(self, text_lower: str, original_text: str) -> List[str]:
        """Parse feet to meters conversions"""
        equations = []
        
        feet_match = re.search(r'(\d+(?:\.\d+)?)\s*feet', original_text)
        if feet_match:
            feet = feet_match.group(1)
            equations.append(f"meters = {feet} * 0.3048")
        
        return equations

    # CONSTRAINT PROBLEMS
    def _parse_at_least(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'at least' constraints"""
        equations = []
        
        match = re.search(r'at least (\d+)', text_lower)
        if match and self.variables:
            min_val = match.group(1)
            var = list(self.variables)[0]  # Use first detected variable
            equations.append(f"{var} >= {min_val}")
        
        return equations

    def _parse_at_most(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'at most' constraints"""
        equations = []
        
        match = re.search(r'at most (\d+)', text_lower)
        if match and self.variables:
            max_val = match.group(1)
            var = list(self.variables)[0]
            equations.append(f"{var} <= {max_val}")
        
        return equations

    def _parse_maximum(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'maximum' constraints"""
        equations = []
        
        match = re.search(r'maximum.*?(\d+)', text_lower)
        if match and self.variables:
            max_val = match.group(1)
            var = list(self.variables)[0]
            equations.append(f"{var} <= {max_val}")
        
        return equations

    def _parse_minimum(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'minimum' constraints"""
        equations = []
        
        match = re.search(r'minimum.*?(\d+)', text_lower)
        if match and self.variables:
            min_val = match.group(1)
            var = list(self.variables)[0]
            equations.append(f"{var} >= {min_val}")
        
        return equations

    def _parse_between(self, text_lower: str, original_text: str) -> List[str]:
        """Parse 'between X and Y' constraints"""
        equations = []
        
        match = re.search(r'between (\d+) and (\d+)', text_lower)
        if match and self.variables:
            low, high = match.groups()
            var = list(self.variables)[0]
            equations.append(f"{low} <= {var} <= {high}")
        
        return equations

    def _analyze_dependencies(self, sent) -> List[str]:

        equations = []
        
        if not self.SPACY_AVAILABLE:
            return equations
        
        try:
            # Analyze subject-verb-object relationships
            for token in sent:
                # Look for mathematical verbs: is, equals, totals, etc.
                if token.lemma_ in ['be', 'equal', 'total', 'sum']:
                    subject = self._get_full_subject(token)
                    complement = self._get_full_complement(token)
                    
                    if subject and complement:
                        # Check if this represents a mathematical relationship
                        if self._is_mathematical_relation(subject, complement):
                            equation = f"{subject} = {complement}"
                            if self._advanced_validation(equation):
                                equations.append(equation)
                                self._extract_variables_from_text(subject)
                                self._extract_variables_from_text(complement)
                
                # Look for prepositional relationships indicating operations
                if token.dep_ == "prep" and token.head.pos_ == "NOUN":
                    preposition = token.text.lower()
                    object_phrase = " ".join([child.text for child in token.children])
                    
                    if preposition == "of" and token.head.text in ['sum', 'product', 'difference']:
                        # Pattern: "sum of A and B"
                        operands = re.findall(r'(\w+)', object_phrase)
                        if len(operands) >= 2:
                            operation = token.head.text
                            if operation == 'sum':
                                equations.append(f"{operands[0]} + {operands[1]}")
                            elif operation == 'product':
                                equations.append(f"{operands[0]} * {operands[1]}")
                            elif operation == 'difference':
                                equations.append(f"{operands[0]} - {operands[1]}")
        
        except Exception as e:
            logger.debug(f"Dependency analysis failed: {e}")
        
        return equations

    def _extract_comparatives(self, sent) -> List[str]:
        """Extract comparative relationships from sentence"""
        equations = []
        
        if not self.SPACY_AVAILABLE:
            return equations
        
        try:
            for token in sent:
                # Handle comparative adjectives
                if token.pos_ == "ADJ" and token.dep_ in ["acomp", "attr"]:
                    comparative_form = token.text.lower()
                    
                    # Common comparative patterns
                    if comparative_form in ['more', 'greater', 'larger', 'bigger']:
                        self._handle_greater_comparison(token, equations)
                    elif comparative_form in ['less', 'smaller', 'fewer']:
                        self._handle_less_comparison(token, equations)
                    elif comparative_form in ['equal', 'same']:
                        self._handle_equal_comparison(token, equations)
                
                # Handle comparative adverbs
                elif token.pos_ == "ADV" and token.dep_ == "advmod":
                    if token.text.lower() in ['twice', 'thrice', 'double', 'triple']:
                        self._handle_multiplicative_comparison(token, equations)
        
        except Exception as e:
            logger.debug(f"Comparative extraction failed: {e}")
        
        return equations

    def _extract_mathematical_relations(self, sent) -> List[str]:
        """Extract explicit mathematical relationships"""
        equations = []
        
        if not self.SPACY_AVAILABLE:
            return equations
        
        try:
            # Look for numeric modifiers and their relationships
            for token in sent:
                if token.like_num:
                    number = token.text
                    # Find what this number modifies
                    self._extract_numeric_relationships(token, number, equations)
                
                # Look for mathematical operations in the text
                if token.text in ['plus', 'minus', 'times', 'divided']:
                    self._extract_verbal_operations(token, equations)
            
            # Extract ratio and proportion relationships
            self._extract_ratio_relationships(sent, equations)
            
            # Extract percentage relationships
            self._extract_percentage_relationships(sent, equations)
        
        except Exception as e:
            logger.debug(f"Mathematical relations extraction failed: {e}")
        
        return equations

    def _infer_from_number_relationships(self, text_lower: str) -> List[str]:
        """Infer equations from numerical patterns and relationships"""
        equations = []
        numbers = self.numbers
        
        if len(numbers) < 2:
            return equations
        
        try:
            # Sum inference
            if any(word in text_lower for word in ['sum', 'total', 'together', 'combined']):
                sum_match = re.search(r'sum.*?(\d+)', text_lower)
                if sum_match:
                    total = sum_match.group(1)
                    # Try different number combinations
                    for i in range(len(numbers)):
                        for j in range(i+1, len(numbers)):
                            if int(numbers[i]) + int(numbers[j]) == int(total):
                                equations.append(f"{numbers[i]} + {numbers[j]} = {total}")
            
            # Difference inference
            if 'difference' in text_lower:
                diff_match = re.search(r'difference.*?(\d+)', text_lower)
                if diff_match:
                    difference = diff_match.group(1)
                    for i in range(len(numbers)):
                        for j in range(len(numbers)):
                            if i != j and abs(int(numbers[i]) - int(numbers[j])) == int(difference):
                                equations.append(f"{numbers[i]} - {numbers[j]} = {difference}")
            
            # Product inference
            if any(word in text_lower for word in ['product', 'times', 'multiplied']):
                product_match = re.search(r'product.*?(\d+)', text_lower)
                if product_match:
                    product = product_match.group(1)
                    for i in range(len(numbers)):
                        for j in range(i+1, len(numbers)):
                            if int(numbers[i]) * int(numbers[j]) == int(product):
                                equations.append(f"{numbers[i]} * {numbers[j]} = {product}")
            
            # Ratio inference
            if 'ratio' in text_lower and len(numbers) >= 2:
                # Check if numbers form a simple ratio
                num1, num2 = int(numbers[0]), int(numbers[1])
                gcd_val = math.gcd(num1, num2)
                if gcd_val > 1:
                    simplified_ratio = f"{num1//gcd_val}:{num2//gcd_val}"
                    equations.append(f"ratio = {simplified_ratio}")
            
            # Sequential number inference (for consecutive numbers)
            if 'consecutive' in text_lower and len(numbers) >= 1:
                start_num = int(numbers[0])
                equations.extend([
                    f"first_number = {start_num}",
                    f"second_number = {start_num + 1}",
                    f"third_number = {start_num + 2}"
                ])
            
            # Even/odd inference
            if any(word in text_lower for word in ['even', 'odd']):
                even_numbers = [n for n in numbers if int(n) % 2 == 0]
                odd_numbers = [n for n in numbers if int(n) % 2 == 1]
                
                if even_numbers and odd_numbers:
                    equations.append(f"even_number = {even_numbers[0]}")
                    equations.append(f"odd_number = {odd_numbers[0]}")
        
        except Exception as e:
            logger.debug(f"Number relationship inference failed: {e}")
        
        return equations

    def _infer_from_mathematical_context(self, text_lower: str) -> List[str]:
        """Infer equations from mathematical context and keywords"""
        equations = []
        
        try:
            # Age problem inferences
            if any(word in text_lower for word in ['year', 'age', 'old', 'young']):
                equations.extend(self._infer_age_equations(text_lower))
            
            # Geometry inferences
            if any(word in text_lower for word in ['area', 'perimeter', 'volume', 'circle', 'rectangle']):
                equations.extend(self._infer_geometry_equations(text_lower))
            
            # Motion inferences
            if any(word in text_lower for word in ['speed', 'distance', 'time', 'km/h', 'mph']):
                equations.extend(self._infer_motion_equations(text_lower))
            
            # Work rate inferences
            if any(word in text_lower for word in ['work', 'complete', 'together', 'rate']):
                equations.extend(self._infer_work_equations(text_lower))
            
            # Mixture inferences
            if any(word in text_lower for word in ['mixture', 'solution', 'concentration', 'blend']):
                equations.extend(self._infer_mixture_equations(text_lower))
            
            # Financial inferences
            if any(word in text_lower for word in ['interest', 'profit', 'loss', 'discount', 'price']):
                equations.extend(self._infer_financial_equations(text_lower))
        
        except Exception as e:
            logger.debug(f"Mathematical context inference failed: {e}")
        
        return equations

    # Helper methods for the above functions
    def _get_full_subject(self, token):
        """Get complete subject phrase"""
        if not self.SPACY_AVAILABLE:
            return None
        
        subject_words = []
        for child in token.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                subject_words.extend(self._get_phrase_words(child))
        
        return " ".join(subject_words) if subject_words else None

    def _get_full_complement(self, token):
        """Get complete complement phrase"""
        if not self.SPACY_AVAILABLE:
            return None
        
        complement_words = []
        for child in token.children:
            if child.dep_ in ["attr", "acomp", "dobj", "prep"]:
                complement_words.extend(self._get_phrase_words(child))
        
        return " ".join(complement_words) if complement_words else None

    def _get_phrase_words(self, token):
        """Get all words in a phrase starting from token"""
        words = [token.text]
        for child in token.children:
            if child.dep_ in ["amod", "compound", "det", "nummod"]:
                words.extend(self._get_phrase_words(child))
        return words

    def _is_mathematical_relation(self, subject: str, complement: str) -> bool:
        """Check if subject-complement represents a mathematical relationship"""
        # Check for numeric content
        subject_has_num = any(c.isdigit() for c in subject)
        complement_has_num = any(c.isdigit() for c in complement)
        
        # Check for variable-like patterns
        subject_has_var = bool(re.search(r'[a-zA-Z]', subject))
        complement_has_var = bool(re.search(r'[a-zA-Z]', complement))
        
        # Check for mathematical operations
        has_operations = any(op in subject + complement for op in ['+', '-', '*', '/'])
        
        return (subject_has_num or complement_has_num) and (subject_has_var or complement_has_var or has_operations)

    def _extract_variables_from_text(self, text: str):
        """Extract variables from natural language text"""
        # Look for single-letter variables
        var_matches = re.findall(r'\b([a-zA-Z])\b', text)
        self.variables.update(var_matches)
        
        # Look for descriptive variables
        descriptive_vars = re.findall(r'\b(age|speed|distance|time|length|width|area|volume)\b', text)
        for var in descriptive_vars:
            # Convert to single letter or keep as is
            var_map = {'age': 'a', 'speed': 's', 'distance': 'd', 'time': 't', 
                    'length': 'l', 'width': 'w', 'area': 'A', 'volume': 'V'}
            self.variables.add(var_map.get(var, var))

    def _handle_greater_comparison(self, token, equations: List[str]):
        """Handle 'greater than' comparisons"""
        subject = self._get_full_subject(token.head)
        amount_match = re.search(r'(\d+)', " ".join([child.text for child in token.children]))
        
        if subject and amount_match:
            amount = amount_match.group(1)
            # Look for what it's greater than
            for child in token.children:
                if child.dep_ == "prep" and child.text == "than":
                    than_obj = " ".join([c.text for c in child.children])
                    if than_obj:
                        equations.append(f"{subject} = {than_obj} + {amount}")

    def _handle_less_comparison(self, token, equations: List[str]):
        """Handle 'less than' comparisons"""
        subject = self._get_full_subject(token.head)
        amount_match = re.search(r'(\d+)', " ".join([child.text for child in token.children]))
        
        if subject and amount_match:
            amount = amount_match.group(1)
            for child in token.children:
                if child.dep_ == "prep" and child.text == "than":
                    than_obj = " ".join([c.text for c in child.children])
                    if than_obj:
                        equations.append(f"{subject} = {than_obj} - {amount}")

    def _handle_equal_comparison(self, token, equations: List[str]):
        """Handle 'equal to' comparisons"""
        subject = self._get_full_subject(token.head)
        for child in token.children:
            if child.dep_ == "prep" and child.text == "to":
                equal_obj = " ".join([c.text for c in child.children])
                if subject and equal_obj:
                    equations.append(f"{subject} = {equal_obj}")

    def _handle_multiplicative_comparison(self, token, equations: List[str]):
        """Handle 'twice', 'double', etc. comparisons"""
        multiplier_map = {'twice': '2', 'double': '2', 'thrice': '3', 'triple': '3'}
        multiplier = multiplier_map.get(token.text.lower())
        
        if multiplier:
            subject = self._get_full_subject(token.head)
            for child in token.children:
                if child.dep_ == "prep" and child.text == "as":
                    # Pattern: "twice as much as X"
                    as_obj = " ".join([c.text for c in child.children])
                    if subject and as_obj:
                        equations.append(f"{subject} = {multiplier} * {as_obj}")

    def _extract_numeric_relationships(self, token, number: str, equations: List[str]):
        """Extract relationships involving numeric tokens"""
        # Check if number modifies a noun (e.g., "5 apples")
        if token.head.pos_ == "NOUN":
            noun = token.head.text
            self.variables.add(noun)
            equations.append(f"count_{noun} = {number}")
        
        # Check for numeric predicates
        if token.dep_ in ["acomp", "attr"]:
            subject = self._get_full_subject(token.head)
            if subject:
                equations.append(f"{subject} = {number}")

    def _extract_verbal_operations(self, token, equations: List[str]):
        """Extract mathematical operations expressed verbally"""
        operation_map = {
            'plus': '+', 'minus': '-', 'times': '*', 'divided': '/'
        }
        
        operation = operation_map.get(token.text.lower())
        if operation:
            # Find the operands
            left_operand = self._find_operand(token, "left")
            right_operand = self._find_operand(token, "right")
            
            if left_operand and right_operand:
                equations.append(f"{left_operand} {operation} {right_operand}")

    def _find_operand(self, token, direction: str) -> str:
        """Find operand in given direction from operation token"""
        if direction == "left":
            # Look for conjuncts or previous tokens
            for child in token.head.children:
                if child.dep_ == "conj" and child.i < token.i:
                    return child.text
        else:  # right
            # Look for objects or subsequent tokens
            for child in token.children:
                if child.dep_ in ["dobj", "prep"]:
                    return " ".join([c.text for c in child.children])
        
        return None

    def _extract_ratio_relationships(self, sent, equations: List[str]):
        """Extract ratio relationships from sentence"""
        for token in sent:
            if token.text.lower() == "ratio" and token.head.pos_ == "NOUN":
                # Pattern: "ratio of A to B"
                ratio_phrase = " ".join([t.text for t in token.subtree])
                ratio_match = re.search(r'ratio of (\w+) to (\w+)', ratio_phrase.lower())
                if ratio_match:
                    a, b = ratio_match.groups()
                    self.variables.update([a, b])
                    equations.append(f"ratio_{a}_{b} = {a}/{b}")

    def _extract_percentage_relationships(self, sent, equations: List[str]):
        """Extract percentage relationships from sentence"""
        for token in sent:
            if token.text == "%" and token.head.pos_ == "NUM":
                percentage = token.head.text
                # Find what the percentage applies to
                for child in token.head.children:
                    if child.dep_ == "prep" and child.text == "of":
                        of_obj = " ".join([c.text for c in child.children])
                        if of_obj:
                            equations.append(f"percentage_part = ({percentage}/100) * {of_obj}")

    # Inference helper methods
    def _infer_age_equations(self, text_lower: str) -> List[str]:
        """Infer age-related equations"""
        equations = []
        numbers = self.numbers
        
        if len(numbers) >= 2:
            # Common age problem pattern: sum and difference
            if 'sum' in text_lower and 'difference' in text_lower:
                equations.extend([
                    f"age1 + age2 = {numbers[0]}",
                    f"age1 - age2 = {numbers[1]}"
                ])
            # Age difference pattern
            elif 'older' in text_lower or 'younger' in text_lower:
                equations.append(f"age1 = age2 + {numbers[0]}")
        
        return equations

    def _infer_geometry_equations(self, text_lower: str) -> List[str]:
        """Infer geometry equations"""
        equations = []
        
        if 'rectangle' in text_lower:
            equations.extend(["area = length * width", "perimeter = 2*(length + width)"])
        elif 'circle' in text_lower:
            equations.extend(["area = pi * radius^2", "circumference = 2 * pi * radius"])
        elif 'triangle' in text_lower:
            equations.append("area = (1/2) * base * height")
        elif 'square' in text_lower:
            equations.extend(["area = side^2", "perimeter = 4 * side"])
        
        return equations

    def _infer_motion_equations(self, text_lower: str) -> List[str]:
        """Infer motion equations"""
        equations = ["distance = speed * time"]
        
        if 'relative' in text_lower or 'faster' in text_lower or 'slower' in text_lower:
            equations.extend([
                "relative_speed = speed1 - speed2",
                "meeting_time = distance / relative_speed"
            ])
        
        return equations

    def _infer_work_equations(self, text_lower: str) -> List[str]:
        """Infer work rate equations"""
        equations = []
        
        if 'together' in text_lower:
            equations.extend([
                "combined_rate = rate1 + rate2",
                "time_together = 1 / combined_rate"
            ])
        else:
            equations.append("work_done = rate * time")
        
        return equations

    def _infer_mixture_equations(self, text_lower: str) -> List[str]:
        """Infer mixture equations"""
        return [
            "total_amount = amount1 + amount2",
            "final_concentration = (concentration1*amount1 + concentration2*amount2) / total_amount"
        ]

    def _infer_financial_equations(self, text_lower: str) -> List[str]:
        """Infer financial equations"""
        equations = []
        
        if 'interest' in text_lower:
            equations.append("amount = principal * (1 + rate*time)")
        elif 'profit' in text_lower:
            equations.extend([
                "profit = selling_price - cost_price",
                "profit_percentage = (profit / cost_price) * 100"
            ])
        elif 'discount' in text_lower:
            equations.extend([
                "discount_amount = original_price * (discount_rate/100)",
                "sale_price = original_price - discount_amount"
            ])
        
        return equations

def extract_equations_advanced(text: str) -> List[str]:
    """
    Main advanced extraction function
    """
    parser = AdvancedMathParser()
    result = parser.parse_problem(text)
    
    logger.info(f"Extracted {len(result['equations'])} equations with confidence {result['confidence']:.2f}")
    
    return result['equations']



class EnhancedMathParser(AdvancedMathParser):
    def __init__(self):
        super().__init__()
        self.variable_tracker = {}  # Track variable values across steps
        self.equation_chain = []    # Store chained equations
        self.problem_steps = []     # Track multi-step reasoning
        
    def parse_multi_step_problem(self, text: str) -> Dict:
        """
        Handle problems with sequential dependencies
        Example: "John has 5 apples. He gives 2 to Mary. How many does he have left?"
        """
        sentences = re.split(r'[.!?]', text)
        equations = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Parse current sentence
            step_result = self.parse_problem(sentence.strip())
            
            # Apply variable propagation
            if step_result['equations']:
                equations.extend(step_result['equations'])
                self._propagate_variables(step_result)
                
        return {
            'equations': equations,
            'variables': list(self.variable_tracker.keys()),
            'problem_type': 'multi_step',
            'equation_chain': self.equation_chain
        }
    
    def _propagate_variables(self, step_result: Dict):
        """Update variable values for next step"""
        for eq in step_result['equations']:
            # Extract solved variables and their values
            if '=' in eq and any(op not in eq for op in ['+', '-', '*', '/']):
                # Simple assignment like "x = 5"
                try:
                    left, right = eq.split('=')
                    left = left.strip()
                    right = right.strip()
                    
                    if left in self.variables and right.isdigit():
                        self.variable_tracker[left] = float(right)
                except:
                    pass



class MathTemplateMatcher:
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self):
        return {
            # AGE PROBLEM TEMPLATES
            'age_difference_sum': {
                'pattern': r'(\w+) is (\d+) years older than (\w+)\. The sum of their ages is (\d+)',
                'equations': ['{0} = {1} + {2}', '{0} + {1} = {3}'],
                'variables': ['age_{0}', 'age_{1}']
            },
            
            # MIXTURE PROBLEM TEMPLATES  
            'mixture_percentage': {
                'pattern': r'mix.*?(\d+)%.*?(\d+)%.*?(\d+) liters',
                'equations': ['({0}/100)*x + ({1}/100)*y = final_concentration*(x+y)', 'x + y = {2}'],
                'variables': ['x', 'y']
            },
            
            # WORK RATE TEMPLATES
            'work_together': {
                'pattern': r'(\w+) can complete.*?(\d+) hours.*?(\w+) can complete.*?(\d+) hours.*?together',
                'equations': ['1/{1} + 1/{3} = 1/total_time'],
                'variables': ['total_time']
            },
            
            # MOTION TEMPLATES
            'relative_speed': {
                'pattern': r'train.*?(\d+) km/h.*?train.*?(\d+) km/h.*?meet.*?(\d+) hours',
                'equations': ['({0} + {1}) * {2} = distance'],
                'variables': ['distance']
            },
            
            # Add 50+ more templates as suggested
            # ...
        }
    
    def match_template(self, text: str) -> Optional[List[str]]:
        for template_name, template in self.templates.items():
            match = re.search(template['pattern'], text.lower())
            if match:
                groups = match.groups()
                equations = []
                
                for eq_template in template['equations']:
                    try:
                        equation = eq_template.format(*groups)
                        equations.append(equation)
                    except:
                        continue
                
                return equations
        return None



class MathGrammarParser:
    """
    Parse using mathematical grammar rules instead of just regex
    """
    
    def __init__(self):
        self.grammar_rules = [
            # Subject-Verb-Quantity patterns
            (r'(\w+) (has|have|contains) (\d+)', self._parse_possession),
            
            # Comparative patterns  
            (r'(\w+) is (\d+) more than (\w+)', self._parse_comparative_more),
            (r'(\w+) is (\d+) less than (\w+)', self._parse_comparative_less),
            
            # Ratio patterns with different phrasing
            (r'the ratio of (\w+) to (\w+) is (\d+) to (\d+)', self._parse_ratio_alt),
            
            # Percentage patterns
            (r'(\w+) is (\d+)% of (\w+)', self._parse_percentage_of_alt),
            
            # Distribution patterns
            (r'divide (\d+) into parts? in ratio (\d+):(\d+)', self._parse_distribution),
        ]
    
    def parse_with_grammar(self, text: str) -> List[str]:
        equations = []
        
        for pattern, handler in self.grammar_rules:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                eqs = handler(match)
                if eqs:
                    equations.extend(eqs)
        
        return equations
    
    def _parse_possession(self, match):
        subject, verb, quantity = match.groups()
        return [f"{subject} = {quantity}"]
    
    def _parse_comparative_more(self, match):
        a, diff, b = match.groups()
        return [f"{a} = {b} + {diff}"]
    
    def _parse_comparative_less(self, match):
        a, diff, b = match.groups()
        return [f"{a} = {b} - {diff}"]
    




class EnsembleCanonicalizer:
    """
    Combine multiple extraction methods for higher accuracy
    """
    
    def __init__(self):
        self.parsers = [
            AdvancedMathParser(),      # Original regex-based
            MathTemplateMatcher(),     # Template-based  
            MathGrammarParser(),       # Grammar-based
            EnhancedMathParser(),      # Multi-step capable
        ]
        
        self.weights = [0.3, 0.4, 0.2, 0.1]  # Confidence weights
    
    def ensemble_parse(self, text: str) -> Dict:
        all_equations = []
        confidences = []
        
        for parser in self.parsers:
            try:
                if hasattr(parser, 'parse_problem'):
                    result = parser.parse_problem(text)
                    equations = result.get('equations', [])
                elif hasattr(parser, 'match_template'):
                    equations = parser.match_template(text) or []
                else:
                    equations = parser.parse_with_grammar(text)
                
                if equations:
                    all_equations.extend(equations)
                    # Higher confidence for more equations or specific patterns
                    confidence = min(len(equations) * 0.3, 1.0)
                    confidences.append(confidence)
                    
            except Exception as e:
                logger.debug(f"Parser failed: {e}")
                continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_equations = []
        for eq in all_equations:
            if eq not in seen:
                seen.add(eq)
                unique_equations.append(eq)
        
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'equations': unique_equations,
            'confidence': overall_confidence,
            'parser_count': len([c for c in confidences if c > 0])
        }
    




    


# Test function
def test_advanced_parser():
    """Test the advanced parser"""
    test_cases = [
        "A is 5 years older than B. The sum of their ages is 25.",
        "The ratio of boys to girls is 3:2. There are 15 boys.",
        "25% of 200 is what number?",
        "A rectangle has length 10 and width 5. What is its area?",
        "x + 5 = 20",
        "If a car travels at 60 km/h for 2 hours, how far does it go?",
        "The sum of two numbers is 40 and their difference is 10.",
        "A store offers 20% discount on a $50 item.",
        "The product of two numbers is 48 and their sum is 14."
    ]
    
    parser = AdvancedMathParser()
    for problem in test_cases:
        print(f"\n{'='*50}")
        print(f"Problem: {problem}")
        result = parser.parse_problem(problem)
        print(f"Equations: {result['equations']}")
        print(f"Variables: {result['variables']}")
        print(f"Type: {result['problem_type']}")
        print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    test_advanced_parser()



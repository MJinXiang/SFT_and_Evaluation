# WIKITQ_EVAL = """You are tasked with comparing a candidate answer to a correct answer to determine if they convey the same meaning based on the following criteria.
#  If they match, output True; if they do not, output False.

# ### Candidate Answer:
# {candidate_answer}

# ### Correct Answer:
# {correct_answer}

# # Criteria:
# - Numerical values are considered consistent if they are the same despite different formats. For example, 0.88 vs 88% → True.
# - Numerical values are also considered consistent if rounding leads to the same result. For example, 8 vs 7.96 → True.
# - Numbers with or without commas (e.g., "8,000" vs "8000") are considered the same.
# - Units such as "pages", "years", "km/h", "days" are considered semantically optional if the core number matches. For example, "2" vs "2 years", "183" vs "183 pages", and "1 week" vs "7 days" all match.
# - Lists of names or categories separated by commas, vertical bars, or spaces are considered equivalent. For example, "Craig Phillips, Tom McDermott" vs "Craig Phillips|Tom McDermott".
# - Parentheses containing additional explanations are ignored. For example, "75 km/h" vs "75 km/h (47 mph)" → True.
# - Diacritics (e.g., "Mónaco" vs "Monaco") are ignored.



# # Output:
# Answer: xxx (True/False)
# """

# WIKITQ_EVAL = """You are tasked with comparing a candidate answer to a correct answer to determine if they convey the same meaning or are equivalent based on the following criteria.

# ### Candidate Answer:
# {candidate_answer}

# ### Correct Answer:
# {correct_answer}

# # Criteria for Judging Equivalence:
# 1. EXACT MATCH: If the answers are identical (ignoring case, punctuation, and spaces), they are equivalent.
# 2. NUMERICAL EQUIVALENCE:
#    - Numerical values are considered consistent if they are the same despite different formats (e.g., 0.88 vs 88%).
#    - Numerical values are also considered consistent if rounding leads to the same result (e.g., 8 vs 7.96).
#    - Numbers with or without commas (e.g., "8,000" vs "8000") are considered the same.
#    - Units such as "pages", "years", "km/h", "days" are considered semantically optional if the core number matches (e.g., "2" vs "2 years").
# 3. LIST EQUIVALENCE:
#    - Lists of names or categories separated by commas, vertical bars, or spaces are considered equivalent (e.g., "Craig Phillips, Tom McDermott" vs "Craig Phillips|Tom McDermott").
# 4. MINOR VARIATIONS:
#    - Parentheses containing additional explanations are ignored (e.g., "75 km/h" vs "75 km/h (47 mph)").
#    - Diacritics are ignored (e.g., "Mónaco" vs "Monaco").

# IMPORTANT: Different entities, places, names, or concepts ARE NOT equivalent. For example, "Bordeaux" and "Anderlecht" are different cities and would be judged as NOT equivalent.

# # Examples:
# -
# Candidate Answer: The capital of France is Paris
# Correct Answer: Paris is the capital of France
# Answer: True
# -
# Candidate Answer: 5 km
# Correct Answer: 5000 m
# Answer: True
# -
# Candidate Answer: 1 million
# Correct Answer: 1000000
# Answer: True
# -
# Candidate Answer: 2023-10-01
# Correct Answer: October 1, 2023
# Answer: True
# -
# Candidate Answer: 300 pages
# Correct Answer: 300
# Answer: True
# -
# Candidate Answer: 25°C
# Correct Answer: 77°F
# Answer: False
# -
# Candidate Answer: 5 km
# Correct Answer: 10 km
# Answer: False
# -
# Candidate Answer: 2023-10-01
# Correct Answer: 2023-11-01
# Answer: False

# # Final Decision:
# Based on your analysis, provide a final verdict of either True (answers are equivalent) or False (answers are not equivalent).

# # Output Format:
# Answer: xxx (only use True or False, not any other words)
# """


WIKITQ_EVAL = """
Here is the original question, the correct answer, and the candidate answer. Please evaluate whether the correct answer and the candidate answer are consistent. 

# Examples:
-
Question: What is the capital of France?
Candidate Answer: The capital of France is Paris
Correct Answer: Paris is the capital of France
Consistent: Yes
-
Question: What is the distance from Paris to London?
Candidate Answer: 5 km
Correct Answer: 5000 m
Consistent: Yes
-
Question: How many people live in the city?
Candidate Answer: 1 million
Correct Answer: 1000000
Consistent: Yes
-
Question: What is the date today?
Candidate Answer: 2023-10-01
Correct Answer: October 1, 2023
Consistent: Yes
-
Question: How many pages are in the book?
Candidate Answer: 300 pages
Correct Answer: 300
Consistent: Yes
-
Question: What is the temperature in Paris?
Candidate Answer: 25°C
Correct Answer: 77°F
Consistent: No
-
Question: What is the distance from Paris to London?
Candidate Answer: 5 km
Correct Answer: 10 km
Consistent: No
-
Question: What is the date today?
Candidate Answer: 2023-10-01
Correct Answer: 2023-11-01
Consistent: No
-
Question: in which three consecutive years was the record the same?
Candidate Answer: 1971,1972,1976
Correct Answer: 2004|2005|2006
Consistent: No

# YOUR TASK
Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

Question: {question}
Candidate Answer: {candidate_answer}
Correct Answer: {correct_answer}
Consistent:
"""
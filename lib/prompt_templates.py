
ptrue_prompt = """
    Q: {q}
    A: {a}
    Is the answer correct? 
    Output True or False in format: 
    Correctness: $\\boxed{{True}}$ or $\\boxed{{False}}$.
"""

verbC_prompt = """
    Q: {q}
    A: {a}
    How likely is the above answer to be correct?
    Output a confidence score ranges from integer 0 ~ 10 that indicates how likely you think the answer is correct.
    Output in the following format:  
    Confidence score: $\\boxed{{score}}$.
"""

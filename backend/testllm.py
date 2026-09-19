from app.services.llm import generate_answer


question = "What is the company's renewable energy target?"

context = """
The company aims to source 60% of its electricity
from renewable sources by 2028.

Currently, renewable electricity represents 42%
of total electricity consumption.
"""


answer = generate_answer(
    question,
    context
)


print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(answer)
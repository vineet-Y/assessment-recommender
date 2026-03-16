import json
import re
from openai import OpenAI

client = OpenAI()


def reason_query(query: str):

    prompt = f"""
You are an expert HR assessment consultant.

Analyze the hiring query or job description.

Extract structured hiring requirements.

Return JSON with fields:

role
seniority
experience_years
technical_skills
competencies
assessment_types
max_duration

Query:
{query}

Return JSON only.
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    text = res.choices[0].message.content

    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except:
        pass

    return {}
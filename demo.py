
import boto3
import json
import streamlit as st
import uuid
import os

os.environ["AWS_PROFILE"] = "default"

REGION = "ap-south-1"
KNOWLEDGE_BASE_ID = "I4MLJIXD3F"
AGENT_ID = "BQCSWTROG1"
AGENT_ALIAS_ID = "RALMMINJ7E"


# =========================================
# AWS CLIENTS
# =========================================
bedrock_agent_client = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION
)


# =========================================
# STUDENT DATA
# =========================================
STUDENTS = {
    "emma": {
        "id": "STU001",
        "name": "Emma",
        "academic_details": {
            "class": "12th Grade",
            "department": "Science",
            "year": 2025
        },
        "subjects": [
            {"name": "Mathematics", "grade": 30},
            {"name": "Physics", "grade": 32},
            {"name": "Chemistry", "grade": 99},
            {"name": "computer science", "grade": 30}
        ],
        "strengths": [
            "Excellent problem-solving skills",
            "Quick understanding of scientific concepts"
        ],
        "weaknesses": [
            "Lower confidence in literature-based subjects",
            "Difficulty summarizing long reading content"
        ],
        "areas_for_improvement": [
            "Improve reading comprehension and essay writing",
            "Practice literature concepts regularly",
            "Enhance note-taking for theory-heavy subjects"
        ],
        "learning_style": {
            "type": "Step-by-step learning",
            "description": "Learns best through structured steps and logical flow.",
            "examples_based_on_strengths": [
                "Math-based explanations with clear formulas",
                "Physics concepts taught using diagrams and real-world analogies",
                "Breaking complex problems into smaller steps"
            ]
        }
    },

    "michael": {
        "id": "STU002",
        "name": "Michael",
        "academic_details": {
            "class": "12th Grade",
            "department": "Arts & Literature",
            "year": 2025
        },
        "subjects": [
            {"name": "Art", "grade": 95},
            {"name": "computer science", "grade": 90},
            {"name": "Mathematics", "grade": 65},
            {"name": "Physics", "grade": 70}
        ],
        "strengths": [
            "Creative expression and imagination",
            "Strong memory and storytelling ability",
            "Good conceptual understanding in humanities subjects"
        ],
        "weaknesses": [
            "Struggles with numerical problem-solving",
            "Difficulty understanding abstract formulas"
        ],
        "areas_for_improvement": [
            "Practice basic mathematics daily",
            "Use visual and real-life examples to learn physics",
            "Strengthen logical reasoning with simple step-based exercises"
        ],
        "learning_style": {
            "type": "Example-based learning",
            "description": "Learns best through visual examples, stories, and comparisons.",
            "examples_based_on_strengths": [
                "Use art-based analogies to explain physics concepts",
                "Explain math problems using visual metaphors",
                "Use storytelling to teach difficult subjects"
            ]
        }
    }
}


# =========================================
# AGENT INVOKE
# =========================================
def invoke_agent_system(system_prompt, user_message):
    session_id = str(uuid.uuid4())

    payload = {
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [{"text": user_message}]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.9 
    }

    response = bedrock_agent_client.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId=session_id,
        inputText=json.dumps(payload)
    )

    text_output = ""
    for event in response.get("completion", []):
        if "chunk" in event:
            text_output += event["chunk"]["bytes"].decode("utf-8")

    return text_output.strip()



def build_full_student_profile(student):

    # ✅ Extract modules & grades dynamically
    modules_and_grades = "\n".join(
        [f"- {s['name']}: {s['grade']}/100" for s in student["subjects"]]
    )

    # ✅ DESCRIPTIVE SYSTEM PROMPT
    system_prompt = """
You are a student profiling assistant.

Your task is to generate a DESCRIPTIVE and ELABORATIVE student profile based ONLY on the given modules and grades.

You must describe the following FOUR sections in detailed paragraph form:

1. Domain of Speciality (explain why the student belongs to this domain)
2. Main Strengths (explain what the student is good at and why)
3. Areas of Improvement (explain where the student struggles and what that means)
4. Suggested Teaching Style (explain HOW the student should be taught based on strengths and weaknesses)

Rules (MANDATORY):
- Domain of Speciality must be inferred ONLY from the modules.
- Main Strengths must be based ONLY on the higher grades.
- Areas of Improvement must be based ONLY on the lower grades.
- Suggested Teaching Style must be inferred from the strongest subjects.
- Do NOT invent any subjects.
- Do NOT use external knowledge.
- Keep the explanation clear, student-focused, and educational.
- Output must be detailed, but not in bullet points — use full sentences and short paragraphs.
"""

    user_message = f"""
Here are the modules the student is enrolled in and the grades they have obtained:

{modules_and_grades}

Generate a DESCRIPTIVE student profile with the four sections above.
"""

    # ✅ CALL YOUR BEDROCK AGENT
    ai_profile = invoke_agent_system(system_prompt, user_message)

    # ✅ FINAL COMBINED FULL PROFILE
    full_profile = f"""
Student ID: {student['id']}
Student Name: {student['name']}
Class: {student['academic_details']['class']}
Department: {student['academic_details']['department']}
Year: {student['academic_details']['year']}

Modules & Grades:
{modules_and_grades}

Generated Descriptive Academic Profile:
{ai_profile}
"""

    print(ai_profile)
    return full_profile.strip()


# =========================================
# KNOWLEDGE BASE RETRIEVAL
# =========================================
def retrieve_learning_material(question, debug=False):

    try:
        response = bedrock_agent_client.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": question},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": 5}
            }
        )

        results = response.get("retrievalResults", [])

        if debug:
            print(json.dumps(results, indent=2))

        if not results:
            return None

        combined = "\n\n".join(
            r["content"]["text"] for r in results if r["content"]["text"]
        )

        return combined.strip()

    except Exception as e:
        print("KB ERROR:", e)
        return None



def generate_personalized_answer(full_profile, learning_material, question):

    if not learning_material:
        return (
            "❌ I could not find any relevant information in the Knowledge Base "
            "for your question."
        )
    system_prompt = """
You are a STRICT Knowledge-Base-Only Student Companion Assistant.
Your job is to generate descriptive, personalized explanations for the student using ONLY the provided learning material and the student's profile.

========================
MANDATORY RULES
========================

ON FIRST MESSAGE:
- When the student enters the chat for the first time, you MUST:
  - Greet the student warmly (e.g., “Welcome back!” / “Hello, glad to see you!”).
  - Clearly point out the student's MAIN STRENGTH.
  - Clearly point out the student's WEAKNESS.
  - Clearly point out the AREA they need to improve.
- This introduction must be based entirely on the student's profile.

KNOWLEDGE USE:
- Use ONLY the provided learning material for factual explanations.
- Do NOT add any external information.
- If the answer is not found in the learning material, reply EXACTLY:
  "I could not find this information in the provided study material."

PERSONALIZATION:
- Personalize the explanation using the student's MAIN STRENGTH from the profile.
- Always mention the student's improvement area when relevant.
- If the student asks to build a roadmap → generate a personalized roadmap for that specific improvement area.
- Adapt explanations to the student's DOMAIN OF SPECIALTY.
- Use the student's TEACHING STYLE to decide HOW to explain.
- Match the explanation with the student's LEARNING STYLE.
- Adjust difficulty based on strengths and weaknesses:
  - If the student is strong in related subjects → go deeper.
  - If the student is weak → simplify, break into small steps, and include examples.

STYLE & TONE:
- Explanations must be descriptive, elaborative, and student-friendly.
- Maintain a polite, friendly, encouraging, and supportive tone.
- Provide gentle improvement tips when needed.
- Use light sentiment analysis:
  - If the student seems confused or stressed → respond with empathy and reassurance.

RESTRICTIONS:
- Do NOT mention Knowledge Bases, embeddings, retrieval, RAG, vector search, or internal system behavior.
- Do NOT refer to yourself as an AI model.
- Do NOT hallucinate or create information not found in the provided study material.
"""



    user_message = f"""
Learning Material:
{learning_material}

Full Student Profile:
{full_profile}

Student Question:
{question}
"""

    return invoke_agent_system(system_prompt, user_message)




st.set_page_config(page_title="Student Companion", page_icon="🎓", layout="wide")
st.title("🎓 Personalized Student Companion (KB-Only & Student-Aware Mode)")



if st.button("🧪 Test KB Connection"):
    st.json(
        bedrock_agent_client.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": "Module"}
        )
    )

col1, col2 = st.columns([1, 2])

# LEFT PANEL — Student Info
with col1:

    student_id = st.selectbox(
        "Select Student",
        options=list(STUDENTS.keys()),
        format_func=lambda s: STUDENTS[s]["name"]
    )

    student = STUDENTS[student_id]

    st.subheader("📘 Student Profile")
    st.write(f"**ID:** {student['id']}")
    st.write(f"**Name:** {student['name']}")
    st.write(f"**Class:** {student['academic_details']['class']}")
    st.write(f"**Department:** {student['academic_details']['department']}")

    st.subheader("📊 Subject Grades")
    for subject in student["subjects"]:
        st.metric(subject["name"], f"{subject['grade']}/100")


# RIGHT PANEL — KB QA
with col2:

    st.subheader("💬 Ask AWS Module Questions")
    question = st.text_area("Enter your question:", height=120)

    if st.button("🚀 Get Personalized Answer", use_container_width=True):

        with st.spinner("Retrieving AWS module content..."):
            material = retrieve_learning_material(question, debug= False)

        if not material:
            st.error("❌ No matching KB content found.")
        else:
            with st.spinner("Generating personalized answer..."):
                full_profile = build_full_student_profile(student)
                answer = generate_personalized_answer(
                    full_profile, material, question
                )

            st.success("✅ Personalized Answer (KB Only):")
            st.markdown(answer)

            with st.expander("📚 Source Material"):
                st.text(material)




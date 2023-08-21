import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI 
from langchain.prompts.chat import PromptTemplate
from langchain.chains import LLMChain,SequentialChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import os

##os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

llm = ChatOpenAI(openai_api_key=st.secrets['OPENAI_API_KEY'],temperature=0.0)

def generate_questions(job_description,llm):
    hr_prompt = """
You are an HR professional tasked with understanding a job description thoroughly. 
Your objective is to accurately extract the primary role and compile a comprehensive list of key concepts 
that the interviewee should be familiar with based on the job description provided in the triple hashtags below:

###{job_description}###

If the provided input is not a job description then output 'none' for role and 'job description not given' for topics.

{format_instructions}
"""

    response_schemas = [
    ResponseSchema(name="role", description="role of the given job description"),
    ResponseSchema(name="topics", description="python list of concepts")
]
    output_parser_1 = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser_1.get_format_instructions()

    concept_template = PromptTemplate(input_variables=["format_instructions","job_description"],template=hr_prompt)

    concept_chain = LLMChain(llm=llm, prompt=concept_template, output_key="concepts")
    interviewer_prompt = """
As an expert interviewer, your task is to formulate a set of 10 probing questions for candidates based on the concepts provided below between triple hastags.
Craft questions that delve into each concept, assessing the candidate's understanding and ability to articulate their knowledge.

###{concepts}###

if The topics contains 'job descriptions not given' then output ['Please give a valid job description'] for questions key.

{instruction_format}
"""
    response_schemas = [
    ResponseSchema(name="questions", description="python list of 10 questions"),
]
    output_parser_2 = StructuredOutputParser.from_response_schemas(response_schemas)
    instructions = output_parser_2.get_format_instructions()

    interviewer_template = PromptTemplate(input_variables=["concepts","instruction_format"],template=interviewer_prompt)

    question_chain = LLMChain(llm=llm, prompt=interviewer_template, output_key="questions")
    
    overall_chain = SequentialChain(
    chains=[concept_chain, question_chain],
    input_variables=["job_description", "format_instructions","instruction_format"],
    output_variables=["concepts", "questions"],
    verbose=False)
    
    output = overall_chain({"job_description":job_description,"format_instructions":format_instructions,"instruction_format":instructions})
    role = output_parser_1.parse(output['concepts'])['role']
    questions = output_parser_2.parse(output['questions'])['questions']
    return {'role':role,'questions':questions}
    

def evaluate_answer(role,question,answer,llm):
    evaluator_prompt = """"
You are an expert {role}, and your task involves evaluating the interviewee's response to the question below:

Question: {question}

Follow the following steps to evaulate the interviewee's response:

STEP 1: Check if the answer is relevent to the question. If yes continue , If no then set all rubrics to 0.
STEP 2: Use the rubric below to assess the given response between the delimeter ###:

Rubric:

Accuracy: Assess the correctness of historical information.
Depth: Consider the level of detail and expansion beyond basic facts.
Coherence: Evaluate the logical flow and organization of ideas.
Grammar and Clarity: Check for proper language use and clarity of expression.
Technical Skills: Rate the candidate's grasp of technical concepts and their ability to apply them effectively.Provide reasoning for the score assigned based on the alignment of technical knowledge with the question's requirements.
Problem-Solving Abilities: Gauge the candidate's approach to problem-solving, including their methodology and creative thinking.Offer reasoning behind the score provided, considering how well the solution addressed the question's context.
Creativity: Assess the originality and innovative thinking demonstrated in the response.Provide rationale for the score assigned, focusing on the uniqueness of the candidate's approach.

Assign scores for each criterion on a scale from 1 to 10.Be strict with the scores.

Response: ###{answer}###

{output_format}
"""
    response_schemas = [
    ResponseSchema(name="question", description="question that was asked"),
    ResponseSchema(name="Accuracy", description="the accuracy score assigned"),
    ResponseSchema(name="Accuracy_reason", description="The reason for the accuracy score assigned"),    
    ResponseSchema(name="Depth", description="the depth score assigned"),
    ResponseSchema(name="Depth_reason", description="The reason for the Depth score assigned"),
    ResponseSchema(name="Coherence", description="the Coherence score assigned"),
    ResponseSchema(name="Coherence_reason", description="The reason for the Coherence score assigned"),
    ResponseSchema(name="Grammar and Clarity", description="the grammar and clarity score assigned"),
    ResponseSchema(name="Grammar_reason", description="The reason for the Grammar and Clarity score assigned"),
    ResponseSchema(name="Technical Skills", description="the Technical skills score assigned"),
    ResponseSchema(name="Technical_reason", description="The reason for the Technical Skill score assigned"),
    ResponseSchema(name="Problem-Solving", description="the Problem-Solving score assigned"),
    ResponseSchema(name="Problem_Solving_reason", description="The reason for the  score assigned"),
    ResponseSchema(name="Creativity", description="the Creativity score assigned"),
    ResponseSchema(name="Creativity_reason", description="The reason for the Creativity score assigned"),
]
    output_parser_3 = StructuredOutputParser.from_response_schemas(response_schemas)
    output_format = output_parser_3.get_format_instructions()
    
    if answer == '':
        answer = 'The applicant didnt answer anything. Set all rubric score to 0.'

    evaluator_template = PromptTemplate(input_variables=["role","question","answer","output_format"],template=evaluator_prompt)
    message = evaluator_template.format(role=role,question=question,answer=answer,output_format=output_format)
    output = llm.predict(message)
    return output_parser_3.parse(output)
    

def evaluate_total_score(role,questions,answers,llm):
    evaluations = []
    for i in range(len(questions)):
        try:
            evaluation = evaluate_answer(role,questions[i],answers[i],llm)
            evaluations.append(evaluation)
        except:
            pass
            
    df = pd.DataFrame(evaluations)
    cols = ['Accuracy','Depth','Coherence','Grammar and Clarity','Technical Skills','Problem-Solving','Creativity']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            
    return df

def result(df):
    cols = ['Accuracy','Depth','Coherence','Grammar and Clarity','Technical Skills','Problem-Solving','Creativity']
    result = df[cols].sum()
    total = len(df)*10
    result /= total
    result *= 100
    final_score = result.mean()
    result['total_score'] = final_score
    return result 

pd.set_option('max_colwidth',None)
    
def display_result(role,questions,llm):
    for i in range(len(questions)):
        answers.append(st.session_state[i])
    df = evaluate_total_score(role,questions,answers,llm)
    final_score = result(df)
    st.write("final score",final_score)
    st.write(df)


# Title
st.header('Hire AI')

with st.sidebar:
        st.title("Hire AI")
        name = st.text_input("name", key="name", value="John Doe")
        job_description = st.text_area("job description(copy paste from linkedin or any site)", key="job_description", height=400)


if not job_description:
    st.info("Please enter a job description")
    st.stop()

role_questions = generate_questions(job_description,llm)
role = role_questions['role']
questions = role_questions['questions']
st.write("Welcome to Hire AI",name,", I am your AI interviwer. I will be asking you a few questions to evaluate your skills. for the job role of a",role,". ")
answers = []
with st.form('interview',clear_on_submit=True):
    for i in range(len(questions)):
        st.write(questions[i])
        st.text_area("answer", key=i, height=400)
    form_submit = st.form_submit_button("submit")
if form_submit:
    st.info("Thank you for your time. I will now evaluate your answers. Please wait for a minute or two....")
    display_result(role,questions,llm)













import streamlit as st
import pandas as pd
from langchain.llms import OpenAI 
from langchain.prompts.chat import PromptTemplate
from langchain.chains import LLMChain,SequentialChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import os

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

llm = OpenAI(openai_api_key=os.environ['OPEN_API_KEY'],temperature=0.0)

def generate_questions(job_description,llm):
    hr_prompt = """
You are an HR professional tasked with understanding a job description thoroughly. 
Your objective is to accurately extract the primary role and compile a comprehensive list of theoretical and techniqal key concepts 
that the interviewee should be familiar with based on the job description provided in the triple backticks below:
```{job_description}```
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
As an expert interviewer, your task is to formulate a set of 10 theoretical and technical questions for candidates based on the concepts provided below.
Craft questions that delve into each concept, assessing the candidate's understanding and ability to articulate their knowledge.
```{concepts}```
{instruction_format}
"""
    response_schemas = [
    ResponseSchema(name="questions", description="python list of 10 questions"),
]
    output_parser_2 = StructuredOutputParser.from_response_schemas(response_schemas)
    instructions = output_parser_2.get_format_instructions()

    interviewer_template = PromptTemplate(input_variables=["concepts","instruction_format"],template=interviewer_prompt)

    concepts = """"'{'role': 'Data Scientist', 'topics': ['Interacting with customers', 'Explaining technical concepts in layman terms', 'Troubleshooting model performance', 'Reviewing model metrics', 'Exploring and recommending best practices for structuring dataset', 'Performing data cleaning and pre-processing', 'Removing rows/cols', 'Stripping values', 'Changing units', 'Normalization', 'Imputing missing values', 'Expanding cols', 'Conducting feature engineering', 'Creating new cols out of existing cols', 'Running statistical functions on rows/cols', 'Transforming columns', 'Analyzing large amounts of information', 'Discovering trends and patterns', 'Building predictive models', 'Building machine-learning algorithms', "Using Obviously Al's No-Code tool", 'Proposing solutions and strategies to business challenges', 'Collaborating with engineering and product development teams', 'Updating and managing technical support tickets and requests']}'"""

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
You possess the expertise of a {role}, and your task involves evaluating the interviewee's response to the following inquiry:

Question: {question}

Employ the provided rubric below to meticulously assess the given response between the delimeter ###:
Kindly assign scores for each criterion on a scale from 1 to 10.
check If the response is empty or irrelevant, please assign a score of 0 if it is empty to all rubric.

Rubric:

Accuracy: Assess the correctness of historical information.Ensure the assessment is relevant to the context of the question; otherwise, mark it as 0.Justify the score.
Depth: Consider the level of detail and expansion beyond basic facts.Ensure the assessment is relevant to the context of the question; otherwise, mark it as 0.Justify the score.
Coherence: Evaluate the logical flow and organization of ideas.Justify the score.
Grammar and Clarity: Check for proper language use and clarity of expression.
Technical Skills: Rate the candidate's grasp of technical concepts and their ability to apply them effectively. Ensure the assessment is relevant to the context of the question; otherwise, mark it as 0. Provide reasoning for the score assigned based on the alignment of technical knowledge with the question's requirements.
Problem-Solving Abilities: Gauge the candidate's approach to problem-solving, including their methodology and creative thinking. Evaluate the relevance of the solution to the given question; otherwise, mark it as 0. Offer reasoning behind the score provided, considering how well the solution addressed the question's context.
Creativity: Assess the originality and innovative thinking demonstrated in the response. Determine whether the creative element aligns with the question's requirements; otherwise, mark it as 0. Provide rationale for the score assigned, focusing on the uniqueness of the candidate's approach.
Attention to Detail: Evaluate the precision and thoroughness of the candidate's response. Ensure the attention to detail is applied in context; otherwise, mark it as 0. Justify the score by illustrating how thoroughly the candidate addressed the question's nuances.
Problem Identification: Assess the candidate's capacity to identify key issues or challenges. Verify whether the identified problems are directly related to the question; otherwise, mark it as 0. Provide reasoning for the score assigned, emphasizing the relevance of the identified problems to the question.

Response: ###{answer}###

{output_format}
"""
    response_schemas = [
    ResponseSchema(name="question", description="question that was asked"),
    ResponseSchema(name="Accuracy", description="the accuracy score assigned"),
    ResponseSchema(name="Depth", description="the depth score assigned"),
    ResponseSchema(name="Coherence", description="the Coherence score assigned"),
    ResponseSchema(name="Grammar and Clarity", description="the grammar and clarity score assigned"),
    ResponseSchema(name="Technical Skills", description="the Technical skills score assigned"),
    ResponseSchema(name="Problem-Solving", description="the Problem-Solving score assigned"),
    ResponseSchema(name="Creativity", description="the Creativity score assigned"),
    ResponseSchema(name="Attention to Detail", description="the Attention to Detail score assigned"),
    ResponseSchema(name="Problem Identification", description="the Problem Identification score assigned"), 
]
    output_parser_3 = StructuredOutputParser.from_response_schemas(response_schemas)
    output_format = output_parser_3.get_format_instructions()

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
    cols = df.columns.drop('question')
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            
    return df
    
    
def result(df):
    result = df.iloc[:,1:10].sum()
    final_score = result.mean()
    result['total_score'] = final_score
    return result
    
placeholder = """2-4 yrs experience with Bachelor's/Master's degree with a focus on CS, Machine Learning, Signal Processing.
Strong knowledge of various ML concepts/algorithms and hands on experience in relevant projects.
Experience in machine learning platform such as TensorFlow, PyTorch and solid programming development skills on Python.
Ability to learn new tools, languages and frameworks quickly.
Familiarity with databases, data transformations techniques, ability to work with unstructured data like OCR/ speech/text data.
Previous experience with working in Conversational AI is a plus.
Git portfolios will be helpful.
"""
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
        job_description = st.text_area("job description(copy paste from linkedin or any site)", key="job_description", height=400,value=placeholder)


if not job_description:
    st.info("Please enter a job description")
    
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
    display_result(role,questions,llm)













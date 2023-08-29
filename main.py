import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import PromptTemplate
from langchain.chains import LLMChain,SequentialChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from supabase import create_client, Client

llm = ChatOpenAI(openai_api_key=st.secrets['OPENAI_API_KEY'],temperature=0.0)       

def generate_questions(job_description,llm):
    hr_prompt = """
You are an HR professional tasked with understanding a job description thoroughly. 
Your objective is to accurately extract the primary role and compile a comprehensive list of key concepts 
that the interviewee should be familiar with based on the job description provided in the triple hashtags below:

###{job_description}###

If the provided input is not a valid job description then output 'none' for role,'none' for category and 'job description not given' for topics.

{format_instructions}
"""

    response_schemas = [
    ResponseSchema(name="role", description="role of the given job description"),
    ResponseSchema(name="Category", description='Put the role in one of the following categories ["Leadership and Management","Sales and Marketing","Finance and Operations","Technology and Innovation","Operation and Supply Chain","Human Resource"]'),
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
    category = output_parser_1.parse(output['concepts'])['Category']
    questions = output_parser_2.parse(output['questions'])['questions']
    return {'role':role,'category':category,'questions':questions}
    

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

Assign scores for each criterion on a scale from 1 to 10.

Response: ###{answer}###

{output_format}
"""
    response_schemas = [
    ResponseSchema(name="question", description="question that was asked"),
    ResponseSchema(name="Accuracy_reason", description="The reason for the accuracy score to be assigned"),
    ResponseSchema(name="Accuracy", description="the accuracy score assigned"),
    ResponseSchema(name="Accuracy_tips", description="Tips to improve the accuracy score assigned"),
    ResponseSchema(name="Depth_reason", description="The reason for the Depth score to be assigned"),
    ResponseSchema(name="Depth", description="the depth score assigned"),
    ResponseSchema(name="Depth_tips", description="Tips to improve the depth score"),
    ResponseSchema(name="Coherence_reason", description="The reason for the Coherence score to be assigned"),
    ResponseSchema(name="Coherence", description="the Coherence score assigned"),
    ResponseSchema(name="Coherence_tips", description="Tips to improve the coherence score"),
    ResponseSchema(name="Grammar_reason", description="The reason for the Grammar and Clarity score to be assigned"),
    ResponseSchema(name="Grammar and Clarity", description="the grammar and clarity score assigned"),
    ResponseSchema(name="Grammar_tips", description="Tips to improve the Grammar and Clarity score"),
    ResponseSchema(name="Technical_reason", description="The reason for the Technical Skill score to be assigned"),
    ResponseSchema(name="Technical Skills", description="the Technical skills score assigned"),
    ResponseSchema(name="Technical_tips", description="Tips to improve the Technical score"),
    ResponseSchema(name="Problem_Solving_reason", description="The reason for the  score to be assigned"),
    ResponseSchema(name="Problem-Solving", description="the Problem-Solving score assigned"),
    ResponseSchema(name="Problem_Solving_tips", description="Tips to improve the problem solving score"),
    ResponseSchema(name="Creativity_reason", description="The reason for the Creativity score to be assigned"),
    ResponseSchema(name="Creativity", description="the Creativity score assigned"),
    ResponseSchema(name="Creativity_tips", description="Tips to improve the Creativity score")
]
    output_parser_3 = StructuredOutputParser.from_response_schemas(response_schemas)
    output_format = output_parser_3.get_format_instructions()
    
    if answer == '':
        answer = 'The applicant didnt answer anything. Set all rubric score to 0.'

    evaluator_template = PromptTemplate(input_variables=["role","question","answer","output_format"],template=evaluator_prompt)
    message = evaluator_template.format(role=role,question=question,answer=answer,output_format=output_format)
    output = llm.predict(message)
    return output_parser_3.parse(output)

def result(evaluations):
    df = pd.DataFrame(evaluations)
    cols = ['Accuracy','Depth','Coherence','Grammar and Clarity','Technical Skills','Problem-Solving','Creativity']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    result = df[cols].sum()
    total = len(df)*10
    result /= total
    result *= 100
    final_score = result.mean()
    result['total_score'] = final_score
    return result

def score(evaluation):
    score = int(evaluation['Accuracy']) + int(evaluation['Depth']) + int(evaluation['Coherence'])+int(evaluation['Grammar and Clarity'])+ int(evaluation['Technical Skills'])+int(evaluation['Problem-Solving'])+int(evaluation['Creativity'])
    score /= 70
    score *=100
    return score

pd.set_option('max_colwidth',None)
    
def display_result(role,questions,llm):
    evaluations = []
    for i in range(len(questions)):
        try:
            evaluation = evaluate_answer(role,questions[i],st.session_state[i],llm)
            with st.expander("Question: "+questions[i]):
                st.write("Evaluation: ",score(evaluation))
                st.write("Breakdown, Tips and Reasoning: ")
                st.table(pd.Series(evaluation))
                st.write("Answer:")
                st.write(st.session_state[i])
            evaluations.append(evaluation)
        except:
            pass
    final = result(evaluations)
    st.write("Result: Your score is out of 100.")
    st.table(final)
    data , count = supabase.table('Leaderboard').insert({'Name':name,'Role':st.session_state.role,'Category':st.session_state.category,'Final Score':final['total_score']}).execute()
    st.write("Thank you for your time. We will get back to you soon.")
    st.balloons()
    st.write('Please refresh the page to start a new interview.')

def init_connection():
    url = st.secrets["supabase_url"]
    key = st.secrets["supabase_key"]
    return create_client(url, key)

supabase = init_connection()

st.set_page_config(page_title="Hire AI",page_icon=" :briefcase: ",layout="wide")

css = r'''
    <style>
        [data-testid="stForm"] {border: 0px}
    </style>
'''

st.markdown(css, unsafe_allow_html=True)

st.header('AI Interviewer')

with st.sidebar:
        st.title("Hire AI :briefcase:")
        with st.form(key="form"):
            name = st.text_input("Name", key="name")
            job_description = st.text_area("Job description(copy paste from linkedin or any site)", key="job_description", height=200)
            submit_button = st.form_submit_button(label="Submit")

if not (job_description and name):
    st.info("Please enter a name and job description in the Sidebar!")
    st.stop()

if "disabled" not in st.session_state:
    st.session_state["disabled"] = False

if "messages" not in st.session_state:
    role_questions = generate_questions(job_description,llm)
    st.session_state.role = role_questions['role']
    st.session_state.category = role_questions['category']
    st.session_state.questions = role_questions['questions']
    st.session_state.i = 0
    st.session_state.messages = [{'role':'assistant','content':"Welcome to Hire AI "+name+", I am your AI interviwer. I will be asking you a few questions to evaluate your skills. for the job role of a "+st.session_state.role+" in category "+st.session_state.category+"."},
                                 {'role':'assistant','content':"Please answer the following "+str(len(st.session_state.questions))+" questions to the best of your ability. You are being evaluated based on accuracy, depth, coherence, grammar, technical skills, problem-solving, and creativity. To achieve high scores in all these criteria, please provide detailed answers with examples, use cases, and innovations if applicable. If you don't know the answer to a question, please type 'I don't know' or 'I don't know the answer to this question'."},
                                 {'role':'assistant','content':st.session_state.questions[0]}]
    
if st.session_state.i == (len(st.session_state.questions)-1):
    st.session_state["disabled"] = True
    st.info("Thank you for your time. I will now evaluate your answers. Please wait for a minute or two....")
    display_result(st.session_state.role,st.session_state.questions,llm)
    st.stop()

with st.container():
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if answer := st.chat_input('answer',disabled=st.session_state["disabled"]):
        st.chat_message('user').markdown(answer)
        st.session_state.messages.append({'role':'user','content':answer})
        i = st.session_state.i
        st.session_state[i] = answer
        st.session_state.i += 1
        i += 1
        
        question = st.session_state.questions[i]
        with st.chat_message('assistant'):
            st.markdown(question)
        st.session_state.messages.append({'role':'assistant','content':question})

   

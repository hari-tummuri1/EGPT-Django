from django.shortcuts import render
from rest_framework import status, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view
from RAG.serializers import EnterpriseDocumentSerializer,UserSerializer, ConversationSerializer, MessageSerializer, UserDocumentSerializer, RephrasedQuestionsSerializer, EnterpriseDictionarySerializer, SystemPromptSerializer, LLMTemperatureSerializer, LLMSerializer, ContentFilterSerializer
from .models import EnterpriseDocuments,Users, Conversation, Message, UserDocuments, EnterpriseDictionary, SystemPrompt, LLMTemperature, LLM, ContentFilters
from django.db.models import Count  
from django.db.models.functions import TruncDate  
from collections import defaultdict 
# from .anonymization import anonymize, deAnonymize
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings  
  

# from .models import Conversation, Message
# from .serializers import ConversationSerializer, MessageSerializer

import os
import openai
# from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from django.core.files.storage import FileSystemStorage
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

## for pinecone
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeEmbeddings,PineconeVectorStore
import asyncio

response_from = ""
# Create your views here.
@api_view(['POST'])
def RagResponse(request):
    
    query = request.data.get('query', None)
    conv_id = request.data.get('conv_id', None)

    if query is None:  
        return Response({'error': 'No query provided'}, status=status.HTTP_400_BAD_REQUEST)
    elif conv_id is None:
         return Response({'error': 'No conversation id provided'}, status=status.HTTP_400_BAD_REQUEST)
    # print(query)
    # print(conv_id)

    # replacement_dict = {  
    #         "Zscaler": "VPN",  
    #         "myWipro": "Intranet"    
    #         # Add other words here    
    #     } 
    
    result = replace_words(query)
    print ("rePhrased Prompt :"+ result[0])
    replaced = ""
    ewords = ""
    if result[0] != query:
        words = result[1]
        replaced_words = result[2]
        for i in range(len(words)):
            if words[i] == replaced_words[i]:
                continue
            else:
                # ewords.append(words[i])
                # replaced.append(replaced_words[i])
                if i == len(words)-1:
                    replaced += "**"+replaced_words[i]+"** "
                    ewords += "**"+words[i]+"** "
                else:
                    replaced += "**"+replaced_words[i]+"** , "
                    ewords += "**"+words[i]+"** , "
    if replaced.endswith(", "):  
        replaced = replaced[:-2]  # remove the last 2 characters
    if ewords.endswith(", "):  
        ewords = ewords[:-2]  # remove the last 2 characters

    result = result[0]
    q_dict = {
        "original_query" : query,
        "rephrased_query" : result
    }


    serializer2 = RephrasedQuestionsSerializer(data=q_dict)
    if serializer2.is_valid():
        serializer2.save()
        # return Response(serializer2.data, status=status.HTTP_201_CREATED)

    userMsg = {
         'msg' : result,
         'conv_id' : conv_id,
         'msg_type' : "user"
    }
    # print(userMsg)
    userserializer = MessageSerializer(data=userMsg)
    print(userserializer)
    

    # anonymized_prompt = anonymize(result)
    # print(anonymized_prompt)

    # answer = StreamingHttpResponse(process_query(anonymized_prompt, conv_id))
    # answer['Content-Type'] = 'text/plain'
    global response_from
    if find_substring(result, "Wipro policy"):
        # pass
        print("Fetching from enterprise knowledge")
        answer = process_query(result, conv_id, "e-gpt") 
        response_from =  "This response is generated using Wipro internal documents"
    else:
        print("Fetching from LLM")
        answer = llmCall(result, conv_id)
        # response_from = "This response is generated by Enterprise dictionary assistance"
    if response_from == "This response is generated by Enterprise dictionary assistance":
        note = "Your query consists of enterprise word(s) "+ewords+". Replacing with "+replaced+ " for better response"
        answer = note+"\n\n"+answer
    print("response : " +answer)
    # answer = deAnonymize(answer)
    print("response from : "+ response_from)

    count = Message.objects.filter(conv_id=conv_id).count()
    if count == 0:
        try:
            conversation = Conversation.objects.get(id=conv_id)
        except Conversation.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)
        
        cdata = {
            'conv_name' : result
        }
        convserializer = ConversationSerializer(conversation, data=cdata, partial=True)
        if convserializer.is_valid():
            convserializer.save()
            print("conv name changed to :" + result)

    if userserializer.is_valid():
        userserializer.save()
    else:
         print('Not valid question')

    botMsg = {
        'msg' : answer,
        'conv_id' : conv_id,
        'msg_type' : "assistant",
        'response_from' : response_from
    }

    botSerializer = MessageSerializer(data = botMsg)
    if botSerializer.is_valid():
        botSerializer.save()
        return Response(botSerializer.data, status=status.HTTP_201_CREATED)
        
    return Response(botSerializer.error_messages, status=status.HTTP_400_BAD_REQUEST)
    # return answer
    # Return the answer  
    # return Response({'answer': answer}) 

# for azure search

# def process_query(query, conv_id, index):
    OPENAI_API_BASE = "https://dwspoc.openai.azure.com/"
    OPENAI_API_KEY = settings.OPENAI_API_KEY
    print("api key" + str(OPENAI_API_KEY))
    OPENAI_API_VERSION = "2024-02-15-preview"
    AZURE_COGNITIVE_SEARCH_SERVICE_NAME = 'enterprisegptaisearch'
    # AZURE_COGNITIVE_SEARCH_INDEX_NAME = "e-gpt"
    AZURE_COGNITIVE_SEARCH_INDEX_NAME = index
    vector_store_address= "https://enterprisegptaisearch.search.windows.net"
    vector_store_password= settings.AZURE_COGNITIVE_SEARCH_API_KEY

    openai.api_type = "azure"
    openai.base_url = OPENAI_API_BASE
    openai.api_key = OPENAI_API_KEY
    openai.api_version = OPENAI_API_VERSION

    conversations = Message.objects.filter(conv_id = conv_id)
    serializer = MessageSerializer(conversations, many=True)
    print(serializer.data)
    context=""
    # for obj in serializer.data[-6:]:  
    #     # get the msg_type and msg  
    #     msg_type = obj.get('msg_type')  
    #     msg = obj.get('msg')  
        
    #     # add the msg_type and msg to the string  
    #     context += f"{msg_type}: {msg}" 



    #initializing LLMs 

    llm = AzureChatOpenAI(deployment_name="GPT4ContentFilter", openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE, openai_api_version=OPENAI_API_VERSION,streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
    # embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002", chunk_size=500, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE, openai_api_version=OPENAI_API_VERSION)
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2023-05-15",
        azure_endpoint=OPENAI_API_BASE,
        api_key= OPENAI_API_KEY)
    
    #connect to azure cognitive search
    acs = AzureSearch(azure_search_endpoint=vector_store_address, azure_search_key=vector_store_password, index_name=AZURE_COGNITIVE_SEARCH_INDEX_NAME, embedding_function=embeddings.embed_query)

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be standalone question. if the provided context does not have information, then say relevant information not found

    Chat history:
    {chat_history}
    Follow up input: {question}
    Standalone question:""")

    qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                                retriever=acs.as_retriever(),
                                                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                                                return_source_documents=True,
                                                verbose=False
                                                )
    
    chat_history = []
    question = 'Answer the following question using the provided context and chat history only, give least priority to chat history. If the provided context and chat history does not have any information about it then say "provided context does not have information about it". Chat history is'+context+' Query : '+query+'generate complete response in 10 seconds'

    print("searching in index "+index)
    result = qa({"question": question, "chat_history": chat_history})
    # for i in qa({"question": question, "chat_history": chat_history}):
    #     yield f"Data chunk {i}\n"
    # print("Question:", query)
    # print("answer:", result["answer"])

    return result['answer']


#for chroma db 

# def process_query(query, conv_id, index):
#     vectorstore_dir = f"user_stores/vectorstore_{index}"
#     print(f'checking in {vectorstore_dir}')
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)

#     # Use the retriever from this vector store
#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
#     retrieved_docs = retriever.invoke(query)
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.3, max_tokens=500)

#     system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise."
#     "\n\n"
#     "{context}"
#     )

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system_prompt),
#             ("human", "{input}"),
#         ]
#     )

#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     rag_chain = create_retrieval_chain(retriever, question_answer_chain)
#     response = rag_chain.invoke({"input": "what is the global psh policy?"})
#     print(response["answer"])
#     return response["answer"]

#for pinecone db
def process_query(query, conv_id, index):
    from langchain.chains import RetrievalQA 
    print('inside process query')
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.3, max_tokens=500)
    index_name = "user-vector-store"
    namespace = index
    PINECONE_API_KEY = settings.PINECONE_API_KEY
    model_name = "multilingual-e5-large"  
    async def initialize_embeddings():
            return PineconeEmbeddings(
                model=model_name,
                pinecone_api_key=PINECONE_API_KEY
            )
    embeddings = asyncio.run(initialize_embeddings())
    print('embeddibgs initiated')

    knowledge = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                namespace=namespace,
                embedding=embeddings
            )

    # Initialize a LangChain object for chatting with the LLM
    # with knowledge from Pinecone. 
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=knowledge.as_retriever()
    )

    answer = qa.invoke(query).get("result")
    return answer


def find_substring(main_string, substring):  
    # Convert both strings to lower case  
    main_string_lower = main_string.lower()  
    substring_lower = substring.lower()  
  
    # Check if the lower-cased substring is in the lower-cased main string  
    if substring_lower in main_string_lower:  
        return True  
    else:  
        return False

def llmCall(query,conv_id):
    import json
    import requests 
    import google.generativeai as genai  

    try:  
        person = SystemPrompt.objects.get(pk=1) 
    except SystemPrompt.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    serializer = SystemPromptSerializer(person)
    system_prompt = serializer.data["prompt"] + "I will give the knowledge about modules present in mywipro with definition, if the employee asks anything related to this modules you please understand the things and assist with the user query. Below are the modules Frequently used modules: myBenefits App for Wipro Benefits Plan (WBP), Superannuation, School Fee, Australia Flex, UK Living Allowance, Furniture & Equipment (F&E) and country specific benefits. myCareer My Career - Performance NXT is the performance management process at Wipro and enables you to set inspiring goals as well as receive meaningful feedback. myData My Data is used to update your profile with contact info and your bank account details. Check your supervisor and HR partner and access your Personal Staffing Page (PSP). myFinancials Access finance, compensation, Salary, Pay slip, Loan & Advances, Survivor benefits, Leave encashment, PF & Pension, Income Tax declarations, Form 16, FormW4-US employees, Onsite statutory Reports & Variable pay. My Learning MyLearning application enables employees to view the competency assessment scores and details. Employees covered in MySkill-Z can map and acquire high demand skills as per their practice, to fuel their career growth. myMedical Claim Medical Assistance Program is aimed towards providing employees and their immediate family (spouse & children) reimbursement claims towards various medical expenses. myRequest This is a one stop shop for raising requests like Guest Pass, Gate Pass, Bus Pass, Id Card etc. You can also track and modify your requests from here. mySpace Book your seat in Wipro locations across India. Seamlessly navigate floor layouts and reserve workstations/half cabins in your ODC or hot-desking space. myTime Mark your attendance, apply for earned time-off/leaves, and submit your overtime and compensatory off details. View your shift schedule. myTransport My Transport is an application for Wipro employees to book roster (Regular, Ad-hoc, Shuttle and special cab). myTravel Book your business travel and accommodation. myDelegationThis application is used to by supervisor to delegate certain tasks and approvals to an employee for e.g., approval of cash claims, travel requests, E-CAR requests, etc. to actionize on behalf of your supervisor. Ariba This application provides a collaborative solution for Wipro Procurement, Strategic Sourcing, Contract Management & Supplier Management. Code of Business Conduct (COBC) Code of Business Conduct(COBC) is the guiding spirit for doing business ethically within the Company. It is not only about compliance but also about meeting our commercial commitments in a legal and ethical manner. Expense Claims Expenses incurred by employees under wipro policy can raise a claim to reimburse. Personal Staffing Page (PSP) This platform gives view of the employee experience details, billability/billing trend, leave, Key contacts (HR,WMG, Manager), Project allocations, tenure information, staff Category and Skill/Role Catalog view. WiServe WiServe is our service management platform to Raise service requests and report incidents. WiLearn Our in-house learning and training platform. Access 9,000+ courses, 22,000+ e-learning modules, and comprehensive learning pathways. Rest of the modules: 360° Survey - Intranet The 360° Survey is a leadership development tool designed to provide holistic feedback to a leader. The survey includes a leader’s manager, co-workers, team members and internal customers who will provide feedback. A3 Reporting Centre A3 is the one source of information for all of Wipro, allowing you to Analyse, to Anticipate and to Act with the confidence of clear and relevant data and insights in real time. Action Tracker Action Tracker is the repository capturing the function-wise activities. It enables tracking of key actions with a view to report progress on a regular basis and also trigger reminders prior to closure date. BGV Master Console Back ground verification configuration for new hire Org -DOP. Candid Voice Candid Voice application provides a platform to raise proactive alerts on potential issues and provide fair feedbacks on projects thus helping smooth execution of projects, positive experience and successful outcomes. Certificate of Coverage-Admin The admin module for Certificate of Coverage (COC) required by an Indian employee travelling overseas to claim exemption from the host(Treaty Signed Country) country's social security. Certificate of Coverage-Employee The EPFO has launched an online facility to apply for a Certificate of Coverage (COC) required by an Indian employee travelling overseas to claim exemption from the host(Treaty Signed Country) country's social security. Claims and Benefit (DOP India) Used to upload and process Incentive for extra productivity and also raise a team celebration claim incurred by DOP INDIA employees. Compliance admin this application is used for generating COBC reports Conflict of Interest This application is designed for employees to identify and disclose any such situation which may be perceived to be an actual or potential conflict with the interests of the company. Cool OffThis application is used for defining competitor account with a cool off clause period of 6 months. The key resources tagged to an account can be allocated to a competitor account based on exception approval. Corporate Internal Audit Portal Audit portal captures the Audit-wise activities in line with ISO 9001. It documents the scope, approach, risk and findings of the Audit steps undertaken. It also track evidence, implementation of agreed Action plan. Customer Supplied Materials (CSM) CSM is one single application that takes care of tracking & accounting of Client Supplied Material [CSM]. User can raise Indent for CSM shipments. Digi-Q digi-q is a strategic and transformational initiative of the Quality function. It is an Enterprise Application with integrated environment for end to end project lifecycle management and focus on Project Governance. DMTS DMTS(Distinguished Member of Technical Staff) is a cadre of expert technologists who have the proficiency and thought leadership to shape Wipro’s technology roadmap and who aspire to work on cutting edge technology. DTS IND Disciplanary tracking system for raising DTS, creating abort/closure of DTS for absconding cases. EBC Reservations Application for EBC rooms booking management for client visits - FMG operations. ECOE Estimation portal for calculating Resource loading sheet(RLS) for Opportunities. Embark Employee Login Embark is an organization-wide application integrated with iVerify to enhance employee experience. Fill this form with accurate and complete information, to help us conduct your verification. Embark – Manager/Onboarding Team Allows the onboarding SPOCs to perform actions like Verify joining mandates, Activate the Employee id for new joinees. Employee movement This application can be used for raising relocation, transfer, supervisor change, review of movement requests by manager and raising support tagging/ de-tagging request. Employee Referral The Employee Referral Portal also called as Wiplinks is a platform that allows you to refer potential candidates who aspire to kickstart their professional journey at Wipro. You can also track and earn rewards. Employee Separation An integrated platform that enables employees to render their resignation and track settlement progress; HR to accept/reverse resignations; admin to view settlement status; clearing agents to update due clearance. Enterprise SearchA search engine to look for Case Studies, Best Practices, etc. available in the Enterprise Knowledge Portal Forecast Solution Standardized forecasting process for revenue and resource. Gift Tracker This Portal is used to disclose the gifts that are received by employees which exceeds the acceptable limit of the value of the gift . Global Delivery Catalogue Global Delivery Catalogue (GDC) is an application which brings information together with access to updated Geo inputs thus helping Wiproite in taking quick sales / delivery related decisions. GMG This application is used by employees to create and process Visa request for their international travel. Information security mandatory awareness trainings Protect Wipro’s sensitive data and digital assets with Infosec Awareness trainings for compliance, risk awareness, and incident reporting. Internal Job Portal (IJP) This application allows employees to keep their profile updated and apply for matching internal open positions. IP Gateway IP Gateway is Wipro's online service for all IPR clearances covering Wipro Ownership verification, Infringement Risk Analysis, Open Source Clearance, Security Assessment and Quality Checks. KYC Vault KYC Vault is the repository to store the documents of banking authorized signatories. It enables tracking of requests for KYC verification of signatory with an approval workflow to avoid any misuse of KYC documents. Lead Smart For enabling new requisition for internal talent movement to Lead Smart Level 1 - first line manager, Lead Smart Level 2 - middle line manager roles. - DOP. Lean Project Management Portal Lean project management portal is used for managing Lean projects through its complete lifecycle. Letters This application helps employees to generate letters for Business Visa, Reference, Employee verification and Address Proofs. Liquidity Damages (LD) Fresh Identification Portal LD Fresh Identification Portal enables Operations team to update fresh identification of LD against the invoices through an approval process. Mentoring NetworksMentoring Networks serves as a technology-enabled marketplace of opportunities for mentors and mentees at Wipro to connect, learn, create bonds and grow. Merit Salary Increase (MSI) The application is used to Generate and release merit salary letter to employee. myEmployment My Employment is an application for raising employee confirmation, L2 confirmation, STAR - identify talent at risk in DOP. myKnowledge A one stop solution for all information about Wipro as well as selected premium knowledge artifacts. myPolicies My Policy is your handbook for country specific and global policies of Wipro bifurcated under various categories such as My Career, My Financials, My Day at Work, My Travel and My Information etc. myStatus This application help users to view the status of various request raised by an Employee. myVoice DP my Voice DP is a forum where IRMC,DP related calls will be raised by an employee and these concerns/calls are investigated, addressed by concerned admin / SPOCS of DP Team. myWorklist It is a system used by employees to manage worklist, Approve/Reject requests which is pending for action. This application also caters to Employee Task and gives a view and status of Employee's Requests. Ombuds Process Portal for raising grievance relating to workplace, breach of COBCE guidelines and policies. PIAM PIAM application is used to effectively monitor employee swipe details. Global Security Group use PIAM tool to manage controllers, map doors and schedule reports. Pragati Employee having any improvement idea can be submitted in this portal, Pragati certificate will be generated when the request is approved. An Integrated application for WT and DOP. Prevention of Sexual Harassment (PSH) Wipro’s Global Policy on Prevention of Sexual Harassment at workplace provides a robust framework of confidentiality, assurance, and protection to all employees, irrespective of gender. Promotion Tool Nomination for employee progression and Manger/HR workflow is done using this application. QAS QAS is a tool to perform auditing for HRSS and IMG. Qualtrics Qualtrics is a employee engagement survey platform.Receivable Management System (RMS) Receivable Management system is portal to manage Collection, CNR, commit amount and not commit amounts against invoices. Sequentra Real Estate Management app that allow users to control,manage & unlock the data within their organization. Sequentra provides highest quality analysis tools to inform decision-making & build competitive advantages. Shift and Roster Bulk roster upload for wipro employees to book a Cab. Spazio-Visual Used to manage work location and seat management. Stay Connected Stay Connected is an application for Wipro employees who are enrolled in Extended Bench Leave Program(EBLP) apprising them about the release letter. Step Portal for step/ progression for IJP, IPP postings. Talent Marketplace Talent Marketplace is a platform for talent in Wipro to explore internal opportunities to further their career aspirations. Talent Skilling Talent Skilling portal enables employees to access the learning offerings from Talent Skilling team. These cover Training programs for Technology skills, Project / Delivery / Program Management areas. Talent Supply Chain (TSC) Workbench This application has modules - Integrated Fulfillment Portal, MTE – Workflow and Bench Management used by IFP, WMG heads and PM/DM for internal fulfillment, act on JC requests and unlock the employee opportunity level. Total Rewards Total Rewards is gateway to your compensation, benefits and beneficiary declaration including medical insurance, retirals, wellbeing, ESI, leaves, WBP, Furniture & Equipment(F&E), MAS, Annual Health Check-up, Car Lease. Veloci-Q This is a quality Management system and document storage repository. Visa Declaration A platform for work visa holders currently at onsite, to declare visa related questionnaires adhering to compliance mandates. Winners' CircleWinners’ Circle is Wipro’s Rewards and Recognition platform. It can be used to reward (monetary) and send appreciations (non-monetary). Points earned from all awards can be accessed here and redeemed. Wipro Cares Wipro Cares is an initiative by the Wipro employees to contribute in the areas of social welfare. Employees can also make contributions as per their convenience to Wipro Cares. Wipro One Wipro One covers the wide gamut of sub processes within the Order to Cash process. It includes Project structure creation, Resource planning, Staffing, Invoicing, etc. Work Permit Planning A platform where Delivery Managers (DM), WMGs and Account Delivery Heads (ADH) perform account-wise planning, quota distribution in order to nominate eligible candidates for H-1B Cap program.If the user question contains PII (Personal Identifiable Information) or confidential information like salary and financial related, then dont process the question instead say 'Sorry! this contains Personal Identifiable Information, i cant process this information' and give what is that Personal Identifiable Information(Do not repeat the PII information, just tell what type of info it is). If the users asks for their previous question, if their previous questions contains PII (Personal Identifiable Information) or confidential information like salary and financial related then give the response saying Sorry! this contains Personal Identifiable Information, i cant process this information' and give what is that Personal Identifiable Information. Give response only in text, don't give response in special characters"
    try:
        temperature = LLMTemperature.objects.get(pk=1)
    except LLMTemperature.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND) 
    
    tempSerializer = LLMTemperatureSerializer(temperature)
    temperature = float(tempSerializer.data["temperature"])
    # print("temperature : "+ temperature)
    print(type(temperature))
    conversations = Message.objects.filter(conv_id = conv_id)
    messageSerializer = MessageSerializer(conversations, many=True)
    system_prompt = {"role":"system","content":system_prompt}
    # query = {"role":"user", "content": query}

    # Gemini Configuration
    genai.configure(api_key=settings.GOOGLE_API_KEY)

    generation_config = {
        "temperature" : 1,
        "top_p":0.95,
        "top_k": 64,
        "max_output_tokens":8192,
        "response_mime_type":"text/plain",
    }

    model = genai.GenerativeModel(
        model_name = "gemini-1.5-flash",
        generation_config=generation_config,
    )
    context = [{"role": "model" if item["msg_type"]=='assitant' else "user", "parts": [item["msg"]]} for item in messageSerializer.data[-6:]] 

    chat_session = model.start_chat( history=context )
    response = chat_session.send_message(query)
    return response.text

    # Openai Configuration

    # context = [{"role": item["msg_type"], "content": item["msg"]} for item in messageSerializer.data[-6:]] 
    # context.insert(0, system_prompt)
    # context.append(query) 

    # print("Context : "+str(context))
    # url = "https://dwspoc.openai.azure.com/openai/deployments/GPT4ContentFilter/chat/completions?api-version=2024-02-15-preview"  
    # headers = {  
    #     "Content-Type": "application/json",  
    #     "api-key": settings.OPENAI_API_KEY
    # }

    # data = {  
    #         "messages": context,  
    #         "max_tokens": 800,  
    #         "temperature": temperature,  
    #         "frequency_penalty": 0,  
    #         "presence_penalty": 0,  
    #         "top_p": 0.95,  
    #         "stop": None  
    #     }

    # response = requests.post(url, headers=headers, data=json.dumps(data))
    # print("this is the llm response : "+str(response.json))
    # return response.json()['choices'][0]['message']['content']
def replace_words(sentence):  
    global response_from
    # words = sentence.split()  
    words = sentence.lower().split()  
    e_words = EnterpriseDictionary.objects.all()
    e_words = EnterpriseDictionarySerializer(e_words, many=True)
    # print(e_words.data)
    e_words = {item['original_word']: item['enterprise_word'] for item in e_words.data}
    # replacement_dict_lower = {key.lower(): value for key, value in e_words.items()} 
    e_words = {k.lower(): v for k, v in e_words.items()} 
    print(e_words)
    replaced_words = [e_words.get(word, word) for word in words]  
    # replaced_words = [replacement_dict[replacement_dict_lower.get(word.lower(), word)] for word in words] 
    replaced_sentence = ' '.join(replaced_words).capitalize()
    if replaced_sentence.lower() == str(sentence).lower():
        response_from = "This response is generated by AI model"
    else:
        response_from = "This response is generated by Enterprise dictionary assistance"
    return [replaced_sentence, words, replaced_words] 

@api_view(['POST'])
def addEnterpriseWord(request):
    serializer = EnterpriseDictionarySerializer(data=request.data)
    if serializer.is_valid():
        # original_word = serializer.validated_data.get('original_word').lower() 
        original_word = serializer.validated_data.get('original_word')  
        original_word = original_word.lower() if original_word else original_word  
        enterprise_word = serializer.validated_data.get('enterprise_word')  
        obj, created = EnterpriseDictionary.objects.get_or_create(  
            original_word__iexact=original_word,  
            defaults={'enterprise_word': enterprise_word, 'original_word': original_word},  
        )

        if not created and obj.enterprise_word != enterprise_word: 
            # obj.original_word = original_word
            obj.enterprise_word = enterprise_word  
            obj.save()
        
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def getAllEnterpriseWords(request):
    words = EnterpriseDictionary.objects.all()
    serializer = EnterpriseDictionarySerializer(words, many=True)
    return Response(serializer.data)

@api_view(['POST'])
def updateEnterpriseWords(request, id):
    try:
        word = EnterpriseDictionary.objects.get(id=id)

    except EnterpriseDictionary.DoesNotExist:
         return Response(status=status.HTTP_404_NOT_FOUND)
    
    serializer = EnterpriseDictionarySerializer(word, data=request.data, partial=True)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_200_OK)
    return Response(status=status.HTTP_400_BAD_REQUEST)

@api_view(['DELETE'])
def deleteEnterpriseWord(request, id):
    try:
         word = EnterpriseDictionary.objects.get(pk=id) 
    except EnterpriseDictionary.DoesNotExist:
         return Response(status=status.HTTP_404_NOT_FOUND)
    
    #deleting the word
    word.delete()
    return Response({'message': 'Enterprise word deleted successfully'},status=status.HTTP_204_NO_CONTENT)

@api_view(['POST'])
def getChatmsgs(request, conv_id):
    msgs = Message.objects.filter(conv_id=conv_id)
    serializer = MessageSerializer(msgs, many=True)
    return Response(serializer.data, status=status.HTTP_202_ACCEPTED)

@api_view(['POST'])
def updateMsg(request, id):
    try:
        message = Message.objects.get(id=id)

    except Message.DoesNotExist:
         return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'POST':
        serializer = MessageSerializer(message, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(status=status.HTTP_400_BAD_REQUEST)


    
@api_view(['POST'])
def DocumentUploadView(request):
    if 'document' not in request.FILES:  
            return Response({'error': 'No document in request.'}, status=status.HTTP_400_BAD_REQUEST) 
        
    document = request.FILES['document']  
    fs = FileSystemStorage(location= r'RAG\EnterpriseDocs\\')  
    filename = fs.save(document.name, document) 
    path = r'RAG\EnterpriseDocs\\' + filename
    docdata = {
         "edoc_path" : path
    }
   
    print(path)


    file_path = fs.path(filename) 
    print(file_path) 
    index_name = 'e-gpt'
    if processDocument(file_path,index_name) == 'Document added':
        serializer = EnterpriseDocumentSerializer(data = docdata)
        if serializer.is_valid():
             serializer.save()
        else:
             print('not a valid data')
        return Response(serializer.data, status=status.HTTP_200_OK)
    
def processDocument(filepath, index_name):
        # By using Azure search

        # OPENAI_API_BASE = "https://dwspoc.openai.azure.com/"
        # OPENAI_API_KEY = settings.OPENAI_API_KEY
        # OPENAI_API_VERSION = "2024-02-15-preview"
        # AZURE_COGNITIVE_SEARCH_SERVICE_NAME = 'enterprisegptaisearch'
        # AZURE_COGNITIVE_SEARCH_INDEX_NAME = index_name
        # vector_store_address= "https://enterprisegptaisearch.search.windows.net"
        # vector_store_password= settings.AZURE_COGNITIVE_SEARCH_API_KEY

        # openai.api_type = "azure"
        # openai.base_url = OPENAI_API_BASE
        # openai.api_key = OPENAI_API_KEY
        # openai.api_version = OPENAI_API_VERSION

        # embeddings = AzureOpenAIEmbeddings(
        #                 azure_deployment="text-embedding-ada-002",
        #                 openai_api_version="2023-05-15",
        #                 azure_endpoint=OPENAI_API_BASE,
        #                 api_key= OPENAI_API_KEY
        #             )
        
        # #Connecting to azure cognitive search
        # acs = AzureSearch(azure_search_endpoint=vector_store_address, azure_search_key=vector_store_password, index_name=AZURE_COGNITIVE_SEARCH_INDEX_NAME, embedding_function=embeddings.embed_query)
        
        # loader = PyPDFLoader(filepath)
        # document = loader.load()
        # text_splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=20)
        # docs = text_splitter.split_documents(document)
        # acs.add_documents(documents=docs)
        # return 'Document added'

        # #By using chroma db

        # # Initialize embedding model
        # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # loader = PyPDFLoader(filepath)
        # data = loader.load()

        # # Split the data into chunks
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        # docs = text_splitter.split_documents(data)

        # # Create a separate directory for each PDF's vector store
        # vectorstore_dir = f"user_stores/vectorstore_{index_name}"
        
        # # Create a Chroma vector store for each PDF
        # vectorstore = Chroma.from_documents(
        #     documents=docs, 
        #     embedding=embeddings,
        #     persist_directory=vectorstore_dir
        # )

    

        # print(f"Vector store for {filepath} created at {vectorstore_dir}")
        # return 'Document added'

        # By using Pinecone

        # Initialize embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        loader = PyPDFLoader(filepath)
        data = loader.load()
        # Split the data into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)
        print('text has been split')
        # Initialize a LangChain embedding object.
        PINECONE_API_KEY = settings.PINECONE_API_KEY
        # print(f'api key {PINECONE_API_KEY}')
        model_name = "multilingual-e5-large"  
        # embeddings = PineconeEmbeddings(  
        #     model=model_name,  
        #     pinecone_api_key=PINECONE_API_KEY 
        # ) 
        async def initialize_embeddings():
            return PineconeEmbeddings(
                model=model_name,
                pinecone_api_key=PINECONE_API_KEY
            )
        embeddings = asyncio.run(initialize_embeddings())
        print('embeddibgs initiated')
        # Embed each chunk and upsert the embeddings into your Pinecone index.
        namespace = index_name
        index_name = "user-vector-store"
        docsearch = PineconeVectorStore.from_documents(
                        documents=docs,
                        index_name=index_name,
                        embedding=embeddings, 
                        namespace=namespace
                    )
        print(f"Vector store for {filepath} created at {index_name}/{namespace}")
        return 'Document added'




@api_view(['DELETE'])
def deleteDocument(request, id):
    try:
         document = EnterpriseDocuments.objects.get(pk=id) 
    except EnterpriseDocuments.DoesNotExist:
         return Response(status=status.HTTP_404_NOT_FOUND)
    serializer = EnterpriseDocumentSerializer(document)
    filepath = serializer.data['edoc_path']

    # Deleting file in filesystem

    print(filepath)
    if os.path.exists(filepath):  
        os.remove(filepath)  
        print("File deleted.")  
    else:  
        return Response(status=status.HTTP_404_NOT_FOUND)  
    
    #deleting the file path in database
    document.delete()

    #Updating the vector store
    endpoint = "https://enterprisegptaisearch.search.windows.net"
    admin_key = settings.AZURE_COGNITIVE_SEARCH_API_KEY 
    index_name = "e-gpt"
    credential = AzureKeyCredential(admin_key)  
    client = SearchIndexClient(endpoint, AzureKeyCredential(admin_key))
    client.delete_index(index_name)  

    pre_path = r'RAG\EnterpriseDocs\\'
    documents = []
    pdf_files = [f for f in os.listdir(pre_path) if f.endswith(".pdf")]  
    print(pdf_files)
    for i in pdf_files:
        processDocument(pre_path+i, index_name)
    
    return Response(status=status.HTTP_204_NO_CONTENT)

@api_view(['DELETE'])
def delete_all_enterprisedocs(request):
    from azure.core.credentials import AzureKeyCredential  
    # from azure.search.documents import SearchIndexClient  
    from azure.search.documents.indexes import SearchIndexClient

    if request.method == 'DELETE':  
        count = EnterpriseDocuments.objects.all().delete()

        folder_path = r'RAG\EnterpriseDocs\\'
  
        for filename in os.listdir(folder_path):  
            if filename.endswith('.pdf'):  
                os.remove(os.path.join(folder_path, filename)) 

        endpoint = "https://enterprisegptaisearch.search.windows.net"
        admin_key = settings.AZURE_COGNITIVE_SEARCH_API_KEY 
        index_name = "e-gpt"  
        # index_name = 'user-docs'
  
        credential = AzureKeyCredential(admin_key)  
        client = SearchIndexClient(endpoint, AzureKeyCredential(admin_key))
        
        client.delete_index(index_name)
        return Response(status=status.HTTP_204_NO_CONTENT)
    
             
@api_view(['GET', 'POST'])             
def UserDetails(request):
     if request.method == 'GET':
        users = Users.objects.all()
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)
     elif request.method == 'POST':
          serializer = UserSerializer(data=request.data)
          if serializer.is_valid():
               serializer.save()
               return Response(serializer.data, status=status.HTTP_201_CREATED)
          return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST) 
     
@api_view(['POST'])     
def ConvDetails(request):
    if request.method == 'POST':
        email_id = request.data.get('email_id', None)
        print(email_id)
        data = {
            'email_id' : email_id,
            'conv_name' : 'New Conversation'
        }
        serializer = ConversationSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            #conversation = Conversation.objects.get(email_id=email_id)
            return Response(serializer.data, status=status.HTTP_201_CREATED) 
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
@csrf_exempt
@api_view(['POST'])
def getAllUserConv(request,email):
    if request.method == 'POST':
        conversations = Conversation.objects.filter(email_id = email)
        serializer = ConversationSerializer(conversations, many=True)
        print("initial conversations "+str(len(serializer.data)))
        # Get all conv_id from Message model
        message_ids = Message.objects.values_list('conv_id', flat=True)

        # Filter the conversations queryset to only include ids present in RagMessage  
        filtered_conversations = conversations.filter(id__in=message_ids)

        # Serialize the queryset with the new serializer  
        serializer = ConversationSerializer(filtered_conversations, many=True)
        print("final conversations "+str(len(serializer.data)))  

        return Response(serializer.data)
         

@api_view(['POST','PUT','GET','DELETE'])
def ConvDetailsPK(request, id):
    try:
        conversation = Conversation.objects.get(id=id)

    except Conversation.DoesNotExist:
         return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'POST':
        serializer = ConversationSerializer(conversation, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(status=status.HTTP_400_BAD_REQUEST)
    elif request.method == 'DELETE':
        Message.objects.filter(conv_id=id).delete()  
        conversation.delete()
        return Response({'message': 'Conversation deleted successfully'},status=status.HTTP_204_NO_CONTENT)
         

@api_view(['POST'])
def userDocumetUpload(request, email_id):
    if 'document' not in request.FILES:  
            return Response({'error': 'No document in request.'}, status=status.HTTP_400_BAD_REQUEST)
    document = request.FILES['document']  
    fs = FileSystemStorage(location= r'RAG\UserDocs\\')  
    filename = fs.save(document.name, document) 
    path = r'RAG\UserDocs\\' + filename
    print(path)
    docdata = {
        'udoc_path' : path,
        'email_id' : email_id
    }

    print(docdata)

    file_path = fs.path(filename)
    # print(file_path)
    print(file_path)

    #parsing file name
    # filename = "GlobalPSHPolicy_5bjLLDt.pdf"  
    filename = filename[:-4]  # remove the last 4 characters (.pdf)  
    filename = filename.replace('_', '-')  # replace '_' with '-'  
    filename = filename.lower()  # convert to lowercase  

    #parsing email_id
    username = email_id.split('@')[0]  # split the string at '@' and get the first part   
    email_id = username.replace('.', '-')  # replace '.' with '_'  

    index_name = filename+"-"+email_id
    print("index name : "+index_name)
    # index_name = 'user-docs'
    if processDocument(file_path,index_name) == 'Document added':
        serializer = UserDocumentSerializer(data = docdata)
        if serializer.is_valid():
             serializer.save()
        else:
             print('not a valid data')
        return Response(serializer.data, status=status.HTTP_200_OK)
   
@api_view(['POST'])   
def deleteUserDoc(request, id):
    email_id = request.data.get('email_id', None)

    try:
         document = UserDocuments.objects.get(pk=id) 
    except UserDocuments.DoesNotExist:
         return Response(status=status.HTTP_404_NOT_FOUND)
    
    serializer = UserDocumentSerializer(document)
    filepath = serializer.data['udoc_path']
    filename = os.path.basename(filepath)

    #parsing file name
    # filename = "GlobalPSHPolicy_5bjLLDt.pdf"  
    filename = filename[:-4]  # remove the last 4 characters (.pdf)  
    filename = filename.replace('_', '-')  # replace '_' with '-'  
    filename = filename.lower()  # convert to lowercase  

    #parsing email_id
    username = email_id.split('@')[0]  # split the string at '@' and get the first part   
    email_id = username.replace('.', '-')  # replace '.' with '_'  

    index_name = filename+"-"+email_id
    print("index name : "+index_name)

    print(filepath)
    if os.path.exists(filepath):  
        os.remove(filepath)  
        print("File deleted from filesystem.")  
    else:  
        return Response(status=status.HTTP_404_NOT_FOUND) 
    
    #deleting the file path in database
    document.delete()
    print('deleted from database')

    #For Azure search

    # #Updating the vector store
    # endpoint = "https://enterprisegptaisearch.search.windows.net"
    # admin_key = settings.AZURE_COGNITIVE_SEARCH_API_KEY  
    # # index_name = 'user-docs'
    # credential = AzureKeyCredential(admin_key)  
    # client = SearchIndexClient(endpoint, AzureKeyCredential(admin_key))
    # print("deleting index " + index_name)
    # client.delete_index(index_name)

    ## for pinecone db
    PINECONE_API_KEY = settings.PINECONE_API_KEY
    namespace = index_name
    index_name = "user-vector-store"
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    index.delete(delete_all=True, namespace=namespace)
    print('deleted from vectorstore')

    # # for chromadb
    # import shutil

    # print(f"Deleting index vectorstore_{index_name}")
    # shutil.rmtree(f"user_stores/vectorstore_{index_name}")

    # import subprocess
    # import platform
    # import shutil
    # import os
    # system_platform = platform.system()
    # folder_path = f"user_stores/vectorstore_{index_name}"

    # try:
    #     if system_platform == "Windows":
    #         # Use 'rd /s /q' to forcefully delete a folder on Windows
    #         subprocess.run(['rd', '/s', '/q', folder_path], check=True)
    #     elif system_platform in ["Linux", "Darwin"]:  # Darwin is for macOS
    #         # Use 'rm -rf' to forcefully delete a folder on Linux and macOS
    #         subprocess.run(['rm', '-rf', folder_path], check=True)
    #     else:
    #         raise NotImplementedError(f"Unsupported OS: {system_platform}")

    #     print(f"Folder '{folder_path}' has been deleted successfully.")
    # except subprocess.CalledProcessError as e:
    #     print(f"Error occurred: {e}")
    # except FileNotFoundError:
    #     print(f"Folder '{folder_path}' does not exist.")
    # except PermissionError:
    #     print(f"Permission denied: Unable to delete '{folder_path}'.")
    # except NotImplementedError as e:
    #     print(e)
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")

    return Response(status=status.HTTP_204_NO_CONTENT)

@api_view(['DELETE'])
def deleteSpecificUserDocs(request, email_id):
    # Retrieve all MyTable objects for the given email_id  
    user_docs = UserDocuments.objects.filter(email_id=email_id)
    serializer = UserDocumentSerializer(user_docs, many=True)
    folder_path = r'RAG\UserDocs\\'
    # Extract filenames and store in a list  
    filenames = [os.path.basename(item["udoc_path"]) for item in serializer.data]

    for item in serializer.data:
        file_path = os.path.join(folder_path, os.path.basename(item["udoc_path"]))
        try:  
            os.remove(file_path)  
            print(f"File {file_path} has been deleted")  
        except FileNotFoundError:  
            print(f"File {file_path} not found")
    
    index_name=[]
    for item in serializer.data:
        filename = item['udoc_path']
        email_id = item['email_id']

        filename = os.path.basename(filename)
        #parsing filename
        filename = filename[:-4]  # remove the last 4 characters (.pdf)  
        filename = filename.replace('_', '-')  # replace '_' with '-'  
        filename = filename.lower()  # convert to lowercase  
        #parsing email_id
        username = email_id.split('@')[0]  # split the string at '@' and get the first part   
        email_id = username.replace('.', '-')  # replace '.' with '_'  
        index = filename+"-"+email_id
        index_name.append(index)
    print(index_name)
    endpoint = "https://enterprisegptaisearch.search.windows.net"
    admin_key = settings.AZURE_COGNITIVE_SEARCH_API_KEY
    credential = AzureKeyCredential(admin_key)  
    client = SearchIndexClient(endpoint, AzureKeyCredential(admin_key))

    for index in index_name:
        print("deleting vectorstore "+index)
        client.delete_index(index)

    # Delete the records  
    for doc in user_docs:  
        doc.delete() 
    
    print(deleteAllUserConvs(email_id=email_id))

    return Response({'message':'Deleted all data related to '+email_id})



    
@api_view(['DELETE'])
def deleteAllUserDocs(request):
    # from azure.core.credentials import AzureKeyCredential  
    # # from azure.search.documents import SearchIndexClient  
    # from azure.search.documents.indexes import SearchIndexClient

    if request.method == 'DELETE':
        documents = UserDocuments.objects.all() 
        document_serializer = UserDocumentSerializer(documents, many=True)

        folder_path = r'RAG\UserDocs\\'
  
        for filename in os.listdir(folder_path):  
            if filename.endswith('.pdf'):  
                os.remove(os.path.join(folder_path, filename)) 

        index_name=[]
        for item in document_serializer.data:
            filename = item['udoc_path']
            email_id = item['email_id']

            filename = os.path.basename(filename)
            #parsing filename
            filename = filename[:-4]  # remove the last 4 characters (.pdf)  
            filename = filename.replace('_', '-')  # replace '_' with '-'  
            filename = filename.lower()  # convert to lowercase  
            #parsing email_id
            username = email_id.split('@')[0]  # split the string at '@' and get the first part   
            email_id = username.replace('.', '-')  # replace '.' with '_'  
            index = filename+"-"+email_id
            index_name.append(index) 

        print(index_name)
        endpoint = "https://enterprisegptaisearch.search.windows.net"
        admin_key = settings.AZURE_COGNITIVE_SEARCH_API_KEY 
        # index_name = "e-gpt"  
        # index_name = 'user-docs'
  
        credential = AzureKeyCredential(admin_key)  
        client = SearchIndexClient(endpoint, AzureKeyCredential(admin_key))
        for index in index_name:
            print("deleting vectorstore "+index)
            client.delete_index(index)
        count = UserDocuments.objects.all().delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
@api_view(['POST'])
def getUserDocs(request):
    email_id = request.data.get('email_id', None)
    documents = UserDocuments.objects.filter(email_id = email_id)
    serializer = UserDocumentSerializer(documents, many = True)
    # Modify the data to only include the filename  
    for item in serializer.data:  
        item['udoc_path'] = os.path.basename(item['udoc_path'])
    return Response(serializer.data)

@api_view(['POST'])
def userDocResponse(request):
    query = request.data.get('query', None)
    filename = request.data.get('filename', None)
    conv_id = request.data.get('conv_id', None)

    if query is None:  
        return Response({'error': 'No query provided'}, status=status.HTTP_400_BAD_REQUEST)
    elif conv_id is None:
         return Response({'error': 'No conversation id provided'}, status=status.HTTP_400_BAD_REQUEST)
    elif filename is None:
         return Response({'error': 'No filename provided'}, status=status.HTTP_400_BAD_REQUEST)
    # email_id = ""
    try:  
        conversation = Conversation.objects.get(pk=conv_id)  
        # email_id = conversation.email_id
        # return Response({'email_id': conversation.email_id})  
    except Conversation.DoesNotExist:  
        return Response({'error': 'Conversation not found'}, status=404)
    conv_serializer = ConversationSerializer(conversation)
    email_id = conv_serializer.data['email_id']
    # email_id = "tummuri.hari@lwpcoe.com"
    
     #parsing file name
    # filename = "GlobalPSHPolicy_5bjLLDt.pdf" 
    filename2 = filename
    filename = filename[:-4]  # remove the last 4 characters (.pdf)  
    filename = filename.replace('_', '-')  # replace '_' with '-'  
    filename = filename.lower()  # convert to lowercase  

    #parsing email_id
    username = email_id.split('@')[0]  # split the string at '@' and get the first part   
    email_id = username.replace('.', '-')  # replace '.' with '_'  

    index_name = filename+"-"+email_id
    print("index name : "+index_name)
    userMsg = {
         'msg' : query,
         'conv_id' : conv_id,
         'msg_type' : "user"
    }
    # print(userMsg)
    userserializer = MessageSerializer(data=userMsg)
    print(userserializer)

    # anonymized_prompt = anonymize(query)
    # print(anonymized_prompt)

    # answer = StreamingHttpResponse(process_query(anonymized_prompt, conv_id))
    # answer['Content-Type'] = 'text/plain'
    answer = process_query(query, conv_id, index_name)
    # answer = deAnonymize(answer)
    print(answer)

    count = Message.objects.filter(conv_id=conv_id).count()
    if count == 0:
        try:
            conversation = Conversation.objects.get(id=conv_id)
        except Conversation.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)
        
        cdata = {
            'conv_name' : query
        }
        convserializer = ConversationSerializer(conversation, data=cdata, partial=True)
        if convserializer.is_valid():
            convserializer.save()
            print("conv name changed to :" + query)


    if userserializer.is_valid():
        userserializer.save()
    else:
         print('Not valid question')

    botMsg = {
        'msg' : answer,
        'conv_id' : conv_id,
        'msg_type' : "assistant",
        'response_from' : 'this response is generated from '+filename2
    }

    botSerializer = MessageSerializer(data = botMsg)
    if botSerializer.is_valid():
        botSerializer.save()
        return Response(botSerializer.data, status=status.HTTP_201_CREATED)
    else:
        print("bot serializer error")
        return Response(botSerializer.error_messages, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def addSystemPrompt(request):
    serializer = SystemPromptSerializer(data = request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.error_messages, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def updateSystemPrompt(request, id):
    try:
        prompt = SystemPrompt.objects.get(id=id)
    except SystemPrompt.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    serializer = SystemPromptSerializer(prompt, data=request.data, partial=True)
    if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
    return Response(status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def getSystemPrompt(request, id):
    try:  
        person = SystemPrompt.objects.get(pk=id) 
    except SystemPrompt.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    serializer = SystemPromptSerializer(person)
    return Response(serializer.data)

@api_view(['POST'])
def addTemperature(request):
    serializer = LLMTemperatureSerializer(data = request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.error_messages, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def updateTemp(request, id):
    try:
        temperature = LLMTemperature.objects.get(id=id)
    except LLMTemperature.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    serializer = LLMTemperatureSerializer(temperature, data=request.data, partial=True)
    if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
    return Response(status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def getTemp(request, id):
    try:  
        person = LLMTemperature.objects.get(pk=id) 
    except LLMTemperature.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    serializer = LLMTemperatureSerializer(person)
    return Response(serializer.data)

@api_view(['POST'])
def dashBoardData(request):
    # pass
    message_feedback = Message.objects.values('feedback').annotate(count=Count('feedback'))
    message_feedback = {item['feedback']: item['count'] for item in message_feedback} 
    grouped_data = Message.objects.annotate(date=TruncDate('time')).values('date', 'conv_id').distinct()

    result = defaultdict(list)
    for item in grouped_data:  
        result[item['date']].append(item['conv_id'])

    date_conv_id = [{'date': key, 'conv_id': list(set(value))} for key, value in result.items()]
    

    for item in date_conv_id:
        conv_id = item['conv_id']
        convs = Conversation.objects.filter(id__in=conv_id)  # get all convs with those ids
        emails = [conv.email_id for conv in convs]
        emails = list(set(emails))
        item['conv_id'] = len(emails)
        # print(str(conv_id)+"\n")
    # grouped_data = Message.objects.annotate(date=TruncDate('time')).values('date').annotate(count=Count('id'))

    # dashboard_data = {
    #     'liked_count' : liked_count,
    #     'disliked_count': disliked_count,
    #     'neutral_count': neutral_count
    # }
    users_per_day = [{"date": d["date"], "user_count": d["conv_id"]} for d in date_conv_id]

    dashboard_data = {
        "feedback" : message_feedback,
        "userUsage" : users_per_day
    }
    return Response(dashboard_data, status=status.HTTP_202_ACCEPTED)

def deleteAllUserConvs(email_id):
    print('email id received '+email_id)
    conversations = Conversation.objects.filter(email_id=email_id)
    serializer = ConversationSerializer(conversations, many=True)

    # Extract the id field  
    id_list = [item['id'] for item in serializer.data]  

    # Delete the records  
    Message.objects.filter(id__in=id_list).delete() 

    # Delete the records in Conversation  
    conversations.delete()  

    return "deleted conversation history of "+email_id

@api_view(['POST'])
def addLLM(request):
    serializer = LLMSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST) 

@api_view(['POST'])
def getAllLLMs(request):
    llms = LLM.objects.all()
    serializer = LLMSerializer(llms, many=True)
    return Response(serializer.data)

@api_view(['POST'])
def getEnabledLLMs(request):
    enabled_llms = LLM.objects.filter(enabled=True)  
    serializer = LLMSerializer(enabled_llms, many=True)  
    return Response(serializer.data)

@api_view(['POST'])
def updateLLM(request, id):
    try:
        llm = LLM.objects.get(id=id)

    except LLM.DoesNotExist:
         return Response(status=status.HTTP_404_NOT_FOUND)
    
    serializer = LLMSerializer(llm, data=request.data, partial=True)

    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_200_OK)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST) 

@api_view(['DELETE'])
def deleteLLM(request, id):
    try:
         llm = LLM.objects.get(pk=id) 
    except LLM.DoesNotExist:
         return Response(status=status.HTTP_404_NOT_FOUND)
    
    # If there are only two records and the other record is not enabled, enable it before deleting the current instance  
    if LLM.objects.count() == 2:  
        other_instance = LLM.objects.exclude(pk=id).first()  
        if not other_instance.enabled:  
            other_instance.enabled = True  
            other_instance.save()
            other_instance.refresh_from_db()  # Ensures that the save operation has completed

    # If there's only one record in the database, return an error and don't delete it  
    elif LLM.objects.count() == 1:  
        return Response({"error": "At least one record must be kept."}, status=400)
    
    # If there's more than one record, proceed with deletion 
    llm.delete()

    return Response({'message': 'LLM deleted successfully'},status=status.HTTP_204_NO_CONTENT)

@api_view(['POST'])
def addContentFilter(request):
    serializer = ContentFilterSerializer(data=request.data)

    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def updateContentFilter(request, id):
    try:
        filter = ContentFilters.objects.get(id=id)

    except ContentFilters.DoesNotExist:
         return Response(status=status.HTTP_404_NOT_FOUND)
    
    serializer = ContentFilterSerializer(filter, data=request.data, partial=True)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_200_OK)
    return Response(status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def getContentFilter(request, id):
    try:  
        filter = ContentFilters.objects.get(pk=id) 
    except ContentFilters.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    serializer = ContentFilterSerializer(filter)
    return Response(serializer.data)

@api_view(['POST'])
def dummyApiCall(request):
    from langchain.chat_models import AzureChatOpenAI
    from langchain.schema import HumanMessage
    from openai import AzureOpenAI
    import json
    import os

    openai_api_base = "https://dwspoc.openai.azure.com/"
    openai_api_version = "2024-02-15-preview"
    deployment_name ="GPT4"
    openai_api_key = settings.OPENAI_API_KEY
    openai_api_type="azure"
    client = AzureOpenAI(
                azure_endpoint = openai_api_base, 
                api_key=openai_api_key,  
                api_version=openai_api_version
            )
    query = request.data.get('query', None)
    annonyised_query = anonymize(query)
    response = client.chat.completions.create(
                model="GPT4", # model = "deployment_name".
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": annonyised_query},
                ]
            )
    
    
    print(annonyised_query)
    # answer = llm([HumanMessage(content=annonyised_query)])
    # answer = answer.to_dict()
    deAnonymised_answer = deAnonymize(response.choices[0].message.content)
    return Response({'answer' : deAnonymised_answer})

@api_view(['GET'])
def hello(request):
    return Response({'Hello world!!!'}, status=status.HTTP_200_OK)


     
         

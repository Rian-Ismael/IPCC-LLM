from langchain.schema import HumanMessage, SystemMessage
from src.nodes.answerer import make_llm

llm = make_llm()

MODERATOR_SYSTEM_PROMPT = """Você é um moderador de IA. Sua tarefa é classificar a pergunta do usuário em uma de três categorias:
1. 'unsafe': Se a pergunta for perigosa, antiética, ilegal ou promover danos.
2. 'off_topic': Se a pergunta não estiver relacionada a mudanças climáticas, IPCC, meio ambiente, energia ou sustentabilidade.
3. 'safe_and_on_topic': Se a pergunta for segura e estiver dentro do escopo do IPCC.

Responda APENAS com UMA destas strings exatas: unsafe | off_topic | safe_and_on_topic
"""

REJECTION_UNSAFE = "Desculpe, não posso responder a perguntas sobre tópicos perigosos ou antiéticos."
REJECTION_OFF_TOPIC = "Desculpe, sou um assistente focado em responder perguntas sobre o relatório do IPCC sobre mudanças climáticas."

def moderate(query: str) -> str:
    messages = [
        SystemMessage(content=MODERATOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Pergunta do usuário: '{query}'")
    ]
    response = llm.invoke(messages)
    category = (response.content or "").strip().lower()

    if category == "unsafe":
        return "reject_unsafe"
    if category == "off_topic":
        return "reject_off_topic"
    return "proceed"

# src/nodes/moderator.py
from langchain.schema import HumanMessage, SystemMessage
from src.nodes.answerer import make_llm # Reutilize seu factory de LLM

llm = make_llm()

MODERATOR_SYSTEM_PROMPT = """Você é um moderador de IA. Sua tarefa é classificar a pergunta do usuário em uma de três categorias:
1.  'unsafe': Se a pergunta for perigosa, antiética, ilegal ou promover danos.
2.  'off_topic': Se a pergunta não estiver relacionada a mudanças climáticas, IPCC, meio ambiente, energia ou sustentabilidade.
3.  'safe_and_on_topic': Se a pergunta for segura e estiver dentro do escopo do IPCC.

Responda APENAS com a categoria exata.
"""

REJECTION_UNSAFE = "Desculpe, não posso responder a perguntas sobre tópicos perigosos ou antiéticos."
REJECTION_OFF_TOPIC = "Desculpe, sou um assistente focado em responder perguntas sobre o relatório do IPCC sobre mudanças climáticas."

def moderate(query: str) -> str:
    """Classifica a pergunta e retorna a próxima ação."""
    messages = [
        SystemMessage(content=MODERATOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Pergunta do usuário: '{query}'")
    ]
    
    response = llm.invoke(messages)
    category = response.content.strip().lower()

    if "unsafe" in category:
        return "reject_unsafe"
    if "off_topic" in category:
        return "reject_off_topic"
    
    return "proceed"
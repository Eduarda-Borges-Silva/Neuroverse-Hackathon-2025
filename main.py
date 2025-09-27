import streamlit as st

MENSAGENS_EXEMPLO = [
    ("user", 'Olá', "🤓"),
    ("ai", 'Tudo Bem?', "🤖"),  
    ("user", 'Tudo Otimo', "🤓"),
    
]

def pagina_chat():
    st.title('Hackaton 2025')
    st.caption("Chatbot • Autorizações • Agendamentos")


def main():
    st.set_page_config(page_title = "Hackaton 2025", layout = "centered")
    pagina_chat()

    messages = st.session_state.get('mensagens', MENSAGENS_EXEMPLO)
    
    for message in messages:
        chat = st.chat_message(message[0], avatar=message[2])
        chat.markdown(message[1])
    

if _name_ == '_main_':
    main()
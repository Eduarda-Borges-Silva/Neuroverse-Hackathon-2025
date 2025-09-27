import os
import io
import re
import random
import string
import tempfile
import unicodedata
import difflib
from datetime import datetime

import streamlit as st
import pandas as pd

from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# opcionais p/ LLMs customizados
try:
    from langchain_openai import ChatOpenAI  # OpenAI-compatível
except Exception:
    ChatOpenAI = None

try:
    from langchain_community.llms import HuggingFaceEndpoint  # texto-LLM (não chat)
except Exception:
    HuggingFaceEndpoint = None

# OCR libs
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract  # requer Tesseract instalado no sistema (CLI); Tesseract.js é alternativa JS
except Exception:
    pytesseract = None

from loaders import *

# ----------------------------
# Configuração base
# ----------------------------
st.set_page_config(page_title="Oráculo", page_icon="🔮", layout="wide")

TIPOS_ARQUIVOS = ['Site', 'Link Youtube', '.PDF', '.CSV', '.TXT']
MEMORIA_PADRAO = ConversationBufferMemory()

# Chaves em st.secrets (sem .env)
GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")
GOOGLE_KEY = st.secrets.get("GOOGLE_API_KEY", "")

# Provedores PRONTOS
PROVEDORES_BASE = {
    'Groq': {
        'modelos': ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile', 'openai/gpt-oss-120b', 'openai/gpt-oss-20b'],
        'chat_cls': ChatGroq,
        'api_key': GROQ_KEY,
        'extra': {}
    },
    'Gemini': {
        'modelos': ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-lite'],
        'chat_cls': ChatGoogleGenerativeAI,
        'api_key': GOOGLE_KEY,
        'extra': {}
    }
}

# Estado global
if 'provedores' not in st.session_state:
    st.session_state['provedores'] = PROVEDORES_BASE.copy()
if 'memoria' not in st.session_state:
    st.session_state['memoria'] = MEMORIA_PADRAO
if 'upload' not in st.session_state:
    st.session_state['upload'] = {'tipo': None, 'arquivo': None}
if 'memoria_upload' not in st.session_state:
    st.session_state['memoria_upload'] = ConversationBufferMemory()
if 'auditorias_pendentes' not in st.session_state:
    # lista de dicts: {protocolo, paciente, documento, prazo_dias, itens, created_at}
    st.session_state['auditorias_pendentes'] = []

# ----------------------------
# Utilidades gerais (chat)
# ----------------------------
def carrega_contexto_raiz(path_padrao: str = "contexto.txt") -> str:
    """Lê o contexto fixo da raiz do projeto (contexto.txt)."""
    try:
        with open(path_padrao, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "(!) Arquivo 'contexto.txt' não encontrado na raiz do projeto."
    except Exception as e:
        return f"(!) Erro ao ler 'contexto.txt': {e}"

def carrega_arquivos(tiposArquivo, arquivo):
    """Carrega e retorna texto do(s) documento(s) conforme o tipo (ou vazio)."""
    if not arquivo:
        return ""
    try:
        if hasattr(arquivo, "seek"):
            try:
                arquivo.seek(0)
            except Exception:
                pass

        if tiposArquivo == 'Site':
            return carrega_sites(arquivo)
        if tiposArquivo == 'Link Youtube':
            return carrega_youtube(arquivo)  # ajuste o nome se no seu loaders for diferente
        if tiposArquivo == '.PDF':
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
                temp.write(arquivo.read())
                name_temp = temp.name
            return carrega_pdf(name_temp)
        if tiposArquivo == '.CSV':
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
                temp.write(arquivo.read())
                name_temp = temp.name
            return carrega_csv(name_temp)
        if tiposArquivo == '.TXT':
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
                temp.write(arquivo.read())
                name_temp = temp.name
            return carrega_txt(name_temp)
    except Exception as e:
        return f"(!) Erro ao carregar arquivo ({tiposArquivo}): {e}"
    return "(!) Tipo de arquivo não suportado."

# ----------------------------
# Chains (chat)
# ----------------------------
def _cria_cliente_chat(prov_info, modelo, api_key):
    extras = prov_info.get('extra', {})
    if prov_info.get('tipo') == 'openai_compat':
        if ChatOpenAI is None:
            st.error("langchain_openai não instalado.")
            st.stop()
        return ChatOpenAI(model=modelo, api_key=api_key, base_url=extras.get('base_url'))
    elif prov_info.get('tipo') == 'hf_inference':
        if HuggingFaceEndpoint is None:
            st.error("langchain_community não instalado.")
            st.stop()
        llm = HuggingFaceEndpoint(repo_id=modelo, huggingface_api_key=api_key)
        class HFChatAdapter:
            def __init__(self, llm): self.llm = llm
            def invoke(self, messages): return self.llm(messages.get('input', ''))
            def stream(self, messages): yield self.invoke(messages)
        return HFChatAdapter(llm)
    else:
        chat_cls = prov_info['chat_cls']
        return chat_cls(model=modelo, api_key=api_key, **extras)

def monta_chain_unificada(provedor: str, modelo: str, api_key: str, tiposArquivo: str, arquivo):
    """Chat principal: usa contexto.txt + (opcional) arquivo/URL salvo no tab Upload."""
    doc_texto = carrega_arquivos(tiposArquivo, arquivo) if tiposArquivo else ""
    contexto_raiz = carrega_contexto_raiz("contexto.txt")

    system_message = f"""Você é um assistente amigável chamado Oráculo.

Você possui acesso a DUAS fontes de informação:

1) **ARQUIVO DE CONTEXTO FIXO (raiz do projeto)**:
### CONTEXTO_FIXO
{contexto_raiz}
###

2) **CONTEÚDO CARREGADO PELO USUÁRIO** (tipo: {tiposArquivo or '-'}):
### DOCUMENTO_USUARIO
{doc_texto}
###

Instruções:
- Priorize coerência entre CONTEXTO_FIXO e DOCUMENTO_USUARIO. Se houver conflito explícito, peça orientação ao usuário.
- Utilize as informações fornecidas para basear suas respostas; não invente dados fora do contexto.
- Sempre que houver um "$" na sua saída, substitua por "S".
- Se aparecer "Just a moment...Enable JavaScript and cookies to continue", sugira recarregar o Oráculo.
"""
    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])

    prov_info = st.session_state['provedores'][provedor]
    chat = _cria_cliente_chat(prov_info, modelo, api_key)
    return template | chat

def monta_chain_upload(provedor: str, modelo: str, api_key: str, tiposArquivo: str, arquivo, incluir_contexto_raiz: bool):
    """Chat do tab Upload (isolado): usa o arquivo/URL do próprio tab (opcionalmente com contexto.txt)."""
    doc_texto = carrega_arquivos(tiposArquivo, arquivo)  # pode ser ""
    contexto_raiz = carrega_contexto_raiz("contexto.txt") if incluir_contexto_raiz else ""

    bloco_ctx = f"### CONTEXTO_FIXO (raiz)\n{contexto_raiz}\n###\n" if incluir_contexto_raiz else ""
    system_message = f"""Você é um assistente chamado Oráculo.

Fontes:
{bloco_ctx}### DOCUMENTO_UPLOAD (tipo: {tiposArquivo or '-'})
{doc_texto or '(vazio)'}
###

Instruções:
- Responda com base nas fontes acima (priorize o DOCUMENTO_UPLOAD).
- Se faltar informação, diga claramente o que falta em vez de inventar.
- Sempre troque "$" por "S".
- Se aparecer "Just a moment...Enable JavaScript and cookies to continue", sugira recarregar o Oráculo.
"""
    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])

    prov_info = st.session_state['provedores'][provedor]
    chat = _cria_cliente_chat(prov_info, modelo, api_key)
    return template | chat

# ----------------------------
# OCR do pedido (PDF/Imagem)
# ----------------------------
def extrair_texto_pedido(pedido_file) -> str:
    """OCR unificado: PDF (texto/ocr) e imagens (png/jpg)."""
    name = (getattr(pedido_file, "name", "") or "").lower()
    if name.endswith(".pdf"):
        if pdfplumber is None:
            return "(!) pdfplumber não instalado para leitura de PDF."
        partes = []
        try:
            if hasattr(pedido_file, "read"):
                pedido_file.seek(0)
                bio = io.BytesIO(pedido_file.read())
            else:
                bio = pedido_file
            with pdfplumber.open(bio) as pdf:
                for page in pdf.pages:
                    txt = page.extract_text() or ""
                    if txt.strip():
                        partes.append(txt)
                    else:
                        if pytesseract is not None and Image is not None:
                            img = page.to_image(resolution=300).original
                            partes.append(pytesseract.image_to_string(img, lang="por+eng"))
                        else:
                            partes.append("[Página sem texto legível e OCR indisponível]")
        except Exception as e:
            return f"(!) Erro ao processar PDF: {e}"
        return "\n".join(partes).strip()
    else:
        if Image is None or pytesseract is None:
            return "(!) OCR indisponível (instale pillow e pytesseract)."
        try:
            if hasattr(pedido_file, "seek"):
                pedido_file.seek(0)
            img = Image.open(pedido_file)
            return pytesseract.image_to_string(img, lang="por+eng")
        except Exception as e:
            return f"(!) Erro no OCR de imagem: {e}"

# ----------------------------
# Planilha do Rol (múltiplas abas) + normalização/prazos
# ----------------------------
def carregar_planilha_upload(file_obj) -> pd.DataFrame:
    """
    Lê planilha (CSV/XLS/XLSX).
    - Para Excel com múltiplas abas, concatena tudo.
    - Infere 'auditoria' / 'prazo_dias' pelo nome da aba:
        Sem auditoria -> auditoria=False, prazo=None
        Auditoria     -> auditoria=True,  prazo=5 (se ausente)
        OPME          -> auditoria=True,  prazo>=10 (se ausente, 10)
    - Normaliza: codigo, procedimento, auditoria, prazo_dias
    """
    if file_obj is None:
        return pd.DataFrame()

    name = (getattr(file_obj, "name", "") or "").lower()
    try:
        if name.endswith(".csv"):
            dfs = {"csv": pd.read_csv(file_obj)}
        else:
            dfs = pd.read_excel(file_obj, sheet_name=None)  # todas as abas
    except Exception as e:
        st.error(f"Falha ao ler planilha: {e}")
        return pd.DataFrame()

    def _norm(df_one: pd.DataFrame) -> pd.DataFrame:
        d = df_one.copy()
        d.columns = [c.strip().lower() for c in d.columns]
        # mapear variações comuns
        rename_map = {}
        for c in list(d.columns):
            if c in ['código','cod','codigo tuss','tuss','code']: rename_map[c] = 'codigo'
            if c in ['nome','descricao','descrição']:             rename_map[c] = 'procedimento'
            if c in ['auditoria?','precisa_auditoria','necessita_auditoria','auditoria']: rename_map[c] = 'auditoria'
            if c in ['prazo','prazo (dias)','sla','sla_dias','prazo_dias']:    rename_map[c] = 'prazo_dias'
        d = d.rename(columns=rename_map)

        for need in ['codigo','procedimento','auditoria','prazo_dias']:
            if need not in d.columns: d[need] = None

        def to_bool(x):
            s = str(x).strip().lower()
            if s in ['true','sim','yes','1']: return True
            if s in ['false','não','nao','no','0']: return False
            return None
        def to_int(x):
            try: return int(float(x))
            except: return None

        d['codigo'] = d['codigo'].astype(str).str.extract(r"(\d+)", expand=False)
        d['procedimento'] = d['procedimento'].astype(str).str.strip()
        d['auditoria'] = d['auditoria'].apply(to_bool)
        d['prazo_dias'] = d['prazo_dias'].apply(to_int)
        return d

    frames = []
    for sheet_name, df_sheet in dfs.items():
        d = _norm(df_sheet)
        sname = str(sheet_name).lower()

        if 'opme' in sname:
            d['auditoria'] = d['auditoria'].fillna(True)
            d['prazo_dias'] = d['prazo_dias'].apply(lambda x: 10 if x is None or x < 10 else x)
            d['_origem_aba'] = 'OPME'
        elif 'auditor' in sname:
            d['auditoria'] = d['auditoria'].fillna(True)
            d['prazo_dias'] = d['prazo_dias'].apply(lambda x: 5 if x is None else x)
            d['_origem_aba'] = 'AUDITORIA'
        elif 'sem' in sname or 'nao' in sname or 'não' in sname:
            d['auditoria'] = d['auditoria'].fillna(False)
            d['prazo_dias'] = d['prazo_dias'].apply(lambda x: None)
            d['_origem_aba'] = 'SEM_AUDITORIA'
        else:
            d['auditoria'] = d['auditoria'].fillna(False)
            d['_origem_aba'] = sheet_name

        frames.append(d[['codigo','procedimento','auditoria','prazo_dias','_origem_aba']])

    if not frames:
        return pd.DataFrame(columns=['codigo','procedimento','auditoria','prazo_dias'])

    df_final = pd.concat(frames, ignore_index=True)
    df_final = df_final.drop_duplicates(subset=['codigo','procedimento'], keep='first')
    return df_final  # mantém _origem_aba visível

# ----------------------------
# Cruzamento Pedido x Planilha (código + nome + fuzzy)
# ----------------------------
def _normalize_text(s: str) -> str:
    if s is None: return ""
    s = s.lower().strip()
    s = ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')
    return s

def cruzar_pedido_com_planilha(texto_pedido: str, df_plan: pd.DataFrame, fuzzy_cutoff: int = 85) -> pd.DataFrame:
    """
    Cruza o texto do pedido (OCR) com a planilha do Rol.
    - Match por código (exato)
    - Match por nome: substring + fuzzy (difflib), cutoff default 85
    Retorna DataFrame com colunas: encontrado, tipo, codigo, procedimento, auditoria, prazo_dias, criterio, _origem_aba
    """
    if not texto_pedido or df_plan.empty:
        return pd.DataFrame(columns=['encontrado','tipo','codigo','procedimento','auditoria','prazo_dias','criterio','_origem_aba'])

    # extrair suspeitas do texto
    codigos = list(set(re.findall(r"\b\d{4,8}\b", texto_pedido)))
    termos_raw = re.findall(r"[A-Za-zÀ-ÿ]{4,}", texto_pedido, flags=re.IGNORECASE)
    termos = list(set(_normalize_text(t) for t in termos_raw if len(t) >= 5))

    # preparar planilha
    df = df_plan.copy()
    if '_origem_aba' not in df.columns:
        df['_origem_aba'] = ''
    df['_proc_norm'] = df['procedimento'].astype(str).apply(_normalize_text)

    resultados = []

    # 1) por código
    if codigos:
        df_codes = df[df['codigo'].isin(codigos)]
        for _, row in df_codes.iterrows():
            resultados.append({
                'encontrado': row.get('codigo'),
                'tipo': 'codigo',
                'codigo': row.get('codigo'),
                'procedimento': row.get('procedimento'),
                'auditoria': row.get('auditoria'),
                'prazo_dias': row.get('prazo_dias'),
                'criterio': 'match:codigo',
                '_origem_aba': row.get('_origem_aba')
            })

    # 2) por nome (contains + fuzzy)
    base_nomes = df['_proc_norm'].tolist()
    for termo in termos:
        # contains
        mask = df['_proc_norm'].str.contains(termo, na=False)
        df_cont = df[mask]
        for _, row in df_cont.iterrows():
            resultados.append({
                'encontrado': termo,
                'tipo': 'termo',
                'codigo': row.get('codigo'),
                'procedimento': row.get('procedimento'),
                'auditoria': row.get('auditoria'),
                'prazo_dias': row.get('prazo_dias'),
                'criterio': 'contains:nome',
                '_origem_aba': row.get('_origem_aba')
            })

        # fuzzy (best match por termo)
        best = difflib.get_close_matches(termo, base_nomes, n=1, cutoff=fuzzy_cutoff/100.0)
        if best:
            alvo = best[0]
            row = df[df['_proc_norm'] == alvo].iloc[0]
            resultados.append({
                'encontrado': termo,
                'tipo': 'termo_fuzzy',
                'codigo': row.get('codigo'),
                'procedimento': row.get('procedimento'),
                'auditoria': row.get('auditoria'),
                'prazo_dias': row.get('prazo_dias'),
                'criterio': f'fuzzy>={fuzzy_cutoff}',
                '_origem_aba': row.get('_origem_aba')
            })

    if not resultados:
        return pd.DataFrame(columns=['encontrado','tipo','codigo','procedimento','auditoria','prazo_dias','criterio','_origem_aba'])

    df_res = pd.DataFrame(resultados)
    df_res = df_res.drop_duplicates(subset=['codigo','procedimento','tipo','criterio'], keep='first')
    return df_res

# ----------------------------
# Utilitários diversos
# ----------------------------
def gerar_token(prefixo: str) -> str:
    import secrets
    rand = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    return f"{prefixo}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{rand}"

def gerar_documento_download(texto: str, filename: str):
    st.download_button(
        label=f"⬇️ Baixar {filename}",
        data=texto.encode('utf-8'),
        file_name=filename,
        mime="text/plain"
    )

# ----------------------------
# Sidebar (somente Modelos)
# ----------------------------
def sidebar_modelos():
    st.subheader("Modelos", divider=True)
    sub = st.tabs(['Prontos', 'Adicionar LLM'])

    # PRONTOS
    with sub[0]:
        provedores_names = list(st.session_state['provedores'].keys())
        provedor = st.selectbox('Provedor', provedores_names, index=0)
        modelos = st.session_state['provedores'][provedor]['modelos']
        modelo = st.selectbox('Modelo', modelos, index=0)

        default_key = st.session_state['provedores'][provedor].get('api_key', '')
        usar_chave_salva = st.checkbox('Usar chave salva automaticamente', value=bool(default_key))
        if usar_chave_salva and default_key:
            api_key = default_key
            st.caption('✅ Usando chave salva em `st.secrets`/memória.')
        else:
            api_key = st.text_input('API Key', type='password')

        st.session_state['selecionado'] = {
            'provedor': provedor,
            'modelo': modelo,
            'api_key': api_key
        }

    # ADICIONAR LLM
    with sub[1]:
        st.write("Cadastre um novo LLM (sem .env, sem JSON manual).")
        novo_nome = st.text_input("Nome do provedor (ex.: 'Meu OpenAI', 'EmpresaX')")
        tipo = st.selectbox("Tipo do provedor", ['OpenAI-compatível', 'HuggingFace Inference'])
        novo_modelo = st.text_input("Modelo (ex.: 'gpt-4o-mini' ou repo_id do HF, p.ex. 'mistralai/Mistral-7B-Instruct')")
        nova_api = st.text_input("API Key do novo provedor", type='password')

        base_url = None
        if tipo == 'OpenAI-compatível':
            base_url = st.text_input("Base URL (opcional) – ex.: https://api.openai.com/v1 ou endpoint compatível")

        if st.button("Adicionar provedor", use_container_width=True):
            if not novo_nome or not novo_modelo or not nova_api:
                st.error("Preencha Nome, Modelo e API Key.")
            else:
                provs = st.session_state['provedores']
                if novo_nome in provs:
                    st.warning("Já existe um provedor com esse nome. Ele será atualizado.")
                entry = {
                    'modelos': [novo_modelo],
                    'api_key': nova_api,
                    'extra': {},
                }
                if tipo == 'OpenAI-compatível':
                    entry['tipo'] = 'openai_compat'
                    entry['chat_cls'] = None
                    entry['extra']['base_url'] = base_url or None
                else:
                    entry['tipo'] = 'hf_inference'
                    entry['chat_cls'] = None

                provs[novo_nome] = entry
                st.session_state['provedores'] = provs
                st.success(f"Provedor '{novo_nome}' adicionado!")

# ----------------------------
# Tab: Upload (Chat isolado + Autorização sem SQL)
# ----------------------------
def pagina_upload():
    st.header('📎 Upload de Arquivos / URLs', divider=True)
    up_tabs = st.tabs(["Envio & Chat do Upload", "Autorização de Exames"])

    # --- SUBTAB 1: Envio & Chat do Upload ---
    with up_tabs[0]:
        tipo = st.selectbox('Tipo de entrada', TIPOS_ARQUIVOS, key='upload_tipo_select')
        arquivo = None
        if tipo == 'Site':
            arquivo = st.text_input('URL do site:', key='upload_site')
        elif tipo == 'Link Youtube':
            arquivo = st.text_input('URL do vídeo:', key='upload_yt')
        elif tipo == '.PDF':
            arquivo = st.file_uploader('Arquivo `.PDF`', type=['pdf'], key='upload_pdf')
        elif tipo == '.CSV':
            arquivo = st.file_uploader('Arquivo `.CSV`', type=['csv'], key='upload_csv')
        elif tipo == '.TXT':
            arquivo = st.file_uploader('Arquivo `.TXT`', type=['txt'], key='upload_txt')

        col_s, col_i = st.columns([1,1])
        with col_s:
            if st.button("Salvar entrada para o Chat principal", use_container_width=True):
                st.session_state['upload'] = {'tipo': tipo, 'arquivo': arquivo}
                st.success("Entrada salva para o Chat principal!")
        with col_i:
            if st.button("Limpar entrada salva", use_container_width=True):
                st.session_state['upload'] = {'tipo': None, 'arquivo': None}
                st.success("Entrada salva limpa.")

        saved = st.session_state.get('upload', {})
        st.caption(f"Salvo p/ Chat principal → tipo={saved.get('tipo') or '-'} | anexado={'sim' if saved.get('arquivo') else 'não'}")

        st.markdown("---")
        st.subheader("💬 Chat do Upload (isolado)", divider=True)

        sel = st.session_state.get('selecionado')
        if not sel:
            st.info("Selecione um provedor/modelo na **sidebar**.")
            st.stop()

        incluir_ctx = st.checkbox("Incluir também o contexto.txt (raiz) neste chat", value=False)

        if st.button("Inicializar Chat do Upload", use_container_width=True):
            if not sel.get('api_key'):
                st.error("API Key vazia. Informe/salve na **sidebar**.")
            else:
                st.session_state['chain_upload'] = monta_chain_upload(
                    provedor=sel['provedor'],
                    modelo=sel['modelo'],
                    api_key=sel['api_key'],
                    tiposArquivo=tipo,
                    arquivo=arquivo,
                    incluir_contexto_raiz=incluir_ctx
                )
                st.success("Chat do Upload inicializado!")

        chain_upload = st.session_state.get('chain_upload')
        if chain_upload is not None:
            mem_up = st.session_state['memoria_upload']
            for msg in mem_up.buffer_as_messages:
                st.chat_message(msg.type).markdown(msg.content)

            entrada = st.chat_input("Converse sobre o arquivo/URL deste tab (chat isolado)")
            if entrada:
                st.chat_message('human').markdown(entrada)
                caixa_ai = st.chat_message('ai')
                try:
                    resposta = caixa_ai.write_stream(chain_upload.stream({
                        'input': entrada,
                        'chat_history': mem_up.buffer_as_messages
                    }))
                except Exception:
                    resp = chain_upload.invoke({
                        'input': entrada,
                        'chat_history': mem_up.buffer_as_messages
                    })
                    resposta = getattr(resp, 'content', resp)
                    caixa_ai.markdown(resposta)

                mem_up.chat_memory.add_user_message(entrada)
                mem_up.chat_memory.add_ai_message(resposta)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Limpar histórico (Chat do Upload)", use_container_width=True):
                    st.session_state['memoria_upload'] = ConversationBufferMemory()
                    st.success("Histórico do chat do upload limpo.")
            with c2:
                if st.button("Reinicializar Chat do Upload", use_container_width=True):
                    st.session_state.pop('chain_upload', None)
                    st.success("Reinicialize clicando em 'Inicializar Chat do Upload'.")

    # --- SUBTAB 2: Autorização de Exames (somente com planilha enviada)
    with up_tabs[1]:
        st.write("Envie o **pedido** (PDF/PNG/JPG) e a **planilha oficial** (CSV/XLSX).")

        col_ped, col_plan = st.columns([1,1])
        with col_ped:
            pedido_file = st.file_uploader("Pedido de exame/procedimento (PDF/PNG/JPG)", type=['pdf', 'png', 'jpg', 'jpeg'])
        with col_plan:
            planilha_file = st.file_uploader("Planilha oficial (CSV/XLSX)", type=['csv', 'xls', 'xlsx'])

        paciente = st.text_input("Paciente (opcional):")
        doc_paciente = st.text_input("Documento/CPF (opcional):")

        if st.button("▶️ Processar Pedido", use_container_width=True):
            if pedido_file is None or planilha_file is None:
                st.error("Envie o **pedido** e a **planilha oficial**.")
            else:
                # 1) OCR/extrair texto do pedido
                texto_pedido = extrair_texto_pedido(pedido_file)

                st.subheader("Texto extraído do pedido")
                st.code((texto_pedido or "").strip()[:5000], language="text")

                # 2) Carregar e normalizar planilha (todas as abas)
                df_plan = carregar_planilha_upload(planilha_file)
                if df_plan.empty:
                    st.error("Não foi possível processar a planilha. Verifique colunas: codigo, procedimento, auditoria, prazo_dias.")
                else:
                    st.success("Planilha carregada.")
                    with st.expander("Visualizar primeiras linhas da planilha"):
                        st.dataframe(df_plan.head(20), use_container_width=True)

                    # 3) Cruzar informações pedido x rol
                    df_res = cruzar_pedido_com_planilha(texto_pedido, df_plan, fuzzy_cutoff=85)
                    if df_res.empty:
                        st.warning("Nenhum procedimento do pedido foi encontrado na planilha. Ajuste os padrões ou confira o pedido.")
                    else:
                        st.subheader("Procedimentos identificados")
                        st.dataframe(df_res, use_container_width=True)

                        # 4) Decisão: se QUALQUER item exigir auditoria → protocolo; caso contrário → autorização automática
                        precisa_auditoria = df_res['auditoria'].fillna(False).any()

                        if not precisa_auditoria:
                            # Autorização automática (sem prazo)
                            codigo_aut = gerar_token("AUT")
                            resumo = []
                            resumo.append("AUTORIZAÇÃO AUTOMÁTICA\n")
                            resumo.append(f"Código: {codigo_aut}")
                            resumo.append(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
                            if paciente:
                                resumo.append(f"Paciente: {paciente}")
                            if doc_paciente:
                                resumo.append(f"Documento: {doc_paciente}")
                            resumo.append("\nProcedimentos autorizados:")
                            for _, r in df_res.iterrows():
                                resumo.append(f"- {r.get('codigo') or '-'} | {r.get('procedimento') or '-'} (sem auditoria)")
                            resumo.append("\nMotivo: Procedimentos não requerem auditoria conforme planilha oficial.")
                            texto_aut = "\n".join(resumo)

                            st.success(f"✅ Autorizado automaticamente. Código: {codigo_aut}")
                            gerar_documento_download(texto_aut, f"autorizacao_{codigo_aut}.txt")

                        else:
                            # Necessita auditoria → protocolo e prazo baseado no máximo dos itens (OPME ≥10; Auditoria 5)
                            protocolo = gerar_token("AUD")
                            prazos = df_res['prazo_dias'].dropna().astype(int)
                            prazo = int(prazos.max()) if not prazos.empty else 5  # fallback 5
                            itens = "; ".join([f"{r.get('codigo') or '-'}|{r.get('procedimento') or '-'}"
                                               for _, r in df_res.iterrows()])

                            st.session_state['auditorias_pendentes'].append({
                                'protocolo': protocolo,
                                'paciente': paciente or "",
                                'documento': doc_paciente or "",
                                'prazo_dias': prazo,
                                'itens': itens,
                                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })

                            linhas = []
                            linhas.append("PROTOCOLADO PARA AUDITORIA\n")
                            linhas.append(f"Protocolo: {protocolo}")
                            linhas.append(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
                            if paciente:
                                linhas.append(f"Paciente: {paciente}")
                            if doc_paciente:
                                linhas.append(f"Documento: {doc_paciente}")
                            linhas.append(f"Prazo estimado de retorno: {prazo} dia(s) úteis.")
                            linhas.append("\nProcedimentos em auditoria:")
                            for _, r in df_res.iterrows():
                                flag = "AUDITORIA" if r.get('auditoria') else "SEM AUDITORIA"
                                origem = r.get('_origem_aba') or '-'
                                linhas.append(f"- {r.get('codigo') or '-'} | {r.get('procedimento') or '-'} [{flag}] (aba: {origem})")
                            texto_aud = "\n".join(linhas)

                            st.warning(f"⏳ Necessita auditoria. Protocolo: {protocolo} | Prazo estimado: {prazo} dia(s) úteis")
                            gerar_documento_download(texto_aud, f"auditoria_{protocolo}.txt")

        st.markdown("---")
        st.subheader("✅ Aprovar Auditoria (memória da sessão) e Enviar Autorização", divider=True)
        pend = st.session_state.get('auditorias_pendentes', [])
        if not pend:
            st.caption("Nenhuma auditoria pendente nesta sessão.")
        else:
            df_pend = pd.DataFrame(pend)
            st.dataframe(df_pend[['protocolo','paciente','documento','prazo_dias','created_at','itens']], use_container_width=True)

            prot_list = [p['protocolo'] for p in pend]
            protocolo_sel = st.selectbox("Escolha um protocolo para aprovar:", prot_list)
            if st.button("Aprovar e Gerar Autorização", use_container_width=True):
                # localizar e remover da fila
                item = None
                for i, it in enumerate(st.session_state['auditorias_pendentes']):
                    if it['protocolo'] == protocolo_sel:
                        item = st.session_state['auditorias_pendentes'].pop(i)
                        break

                codigo_aut = gerar_token("AUT")
                linhas = [
                    "AUTORIZAÇÃO APÓS AUDITORIA",
                    f"Código: {codigo_aut}",
                    f"Protocolo auditoria: {protocolo_sel}",
                    f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                    f"Paciente: {item.get('paciente') if item else '-'}",
                    f"Documento: {item.get('documento') if item else '-'}",
                    "\nProcedimentos autorizados:",
                    item.get('itens') if item else "-"
                ]
                texto_aut = "\n".join(linhas)
                st.success(f"🎉 Auditoria aprovada. Autorização gerada: {codigo_aut}")
                gerar_documento_download(texto_aut, f"autorizacao_{codigo_aut}.txt")

        with st.expander("ℹ️ Observações & OCR"):
            st.markdown(
                "- **OCR**: este app usa **Tesseract** via `pytesseract` (gratuito). Em aplicações web puras JS, pode-se usar **Tesseract.js**.\n"
                "- **Planilha do Rol**: a decisão é feita **somente** com base no arquivo enviado (CSV/XLSX) nesta tela, lendo todas as abas (Sem auditoria / Auditoria / OPME).\n"
                "- **Regras de prazo**: Sem auditoria = *sem dias*; Auditoria = **5**; OPME = **≥10** (se não houver na planilha, força 10).\n"
                "- **Aprovação**: a fila de auditorias fica **apenas na sessão atual**. Ao recarregar o app, ela zera."
            )

# ----------------------------
# Tab: Chat principal (unificado)
# ----------------------------
def pagina_chat_unificada():
    st.header('💬 Oráculo — Chat', divider=True)

    sel = st.session_state.get('selecionado')
    up = st.session_state.get('upload', {})

    colA, colB, colC = st.columns(3)
    with colA:
        st.write(f"**Provedor:** {sel['provedor'] if sel else '-'}")
    with colB:
        st.write(f"**Modelo:** {sel['modelo'] if sel else '-'}")
    with colC:
        st.write(f"**Entrada salva:** {up.get('tipo') or '-'} {'✅' if up.get('arquivo') else '—'}")

    if st.button('Inicializar Oráculo (Unificado)', use_container_width=True):
        if not sel:
            st.error("Selecione um provedor/modelo na sidebar.")
        elif not sel.get('api_key'):
            st.error("API Key vazia. Informe/salve na sidebar.")
        else:
            st.session_state['chain_unificada'] = monta_chain_unificada(
                provedor=sel['provedor'],
                modelo=sel['modelo'],
                api_key=sel['api_key'],
                tiposArquivo=up.get('tipo'),
                arquivo=up.get('arquivo')
            )
            st.success("Oráculo inicializado! (Arquivo + contexto.txt)")

    chain = st.session_state.get('chain_unificada')
    if chain is None:
        st.info("Configure modelos na sidebar e (opcional) salve um arquivo/URL no tab 📎 Upload. Depois clique em **Inicializar Oráculo (Unificado)**.")
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA_PADRAO)
    for mensagem in memoria.buffer_as_messages:
        st.chat_message(mensagem.type).markdown(mensagem.content)

    entrada = st.chat_input('Pergunte algo para o Oráculo')
    if entrada:
        st.chat_message('human').markdown(entrada)
        caixa_ai = st.chat_message('ai')
        try:
            resposta = caixa_ai.write_stream(chain.stream({
                'input': entrada,
                'chat_history': memoria.buffer_as_messages
            }))
        except Exception:
            resp = chain.invoke({
                'input': entrada,
                'chat_history': memoria.buffer_as_messages
            })
            resposta = getattr(resp, 'content', resp)
            caixa_ai.markdown(resposta)

        memoria.chat_memory.add_user_message(entrada)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Apagar histórico de conversa', use_container_width=True):
            st.session_state['memoria'] = ConversationBufferMemory()
            st.success("Histórico apagado.")
    with col2:
        if st.button('Reinicializar Oráculo', use_container_width=True):
            st.session_state.pop('chain_unificada', None)
            st.success("Reinicialize usando o botão acima.")

# ----------------------------
# main
# ----------------------------
def main():
    with st.sidebar:
        sidebar_modelos()

    t_chat, t_upload = st.tabs(["💬 Chat", "📎 Upload"])
    with t_chat:
        pagina_chat_unificada()
    with t_upload:
        pagina_upload()

if __name__ == '__main__':
    main()
import os
import io
import re
import difflib
import sqlite3
import tempfile
import unicodedata
import string  # necessário p/ gerar tokens
from pathlib import Path
from datetime import datetime

import googleapiclient
import streamlit as st
import pandas as pd

from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI  # ajuste de import
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
    import pytesseract  # requer Tesseract instalado no sistema (CLI)
except Exception:
    pytesseract = None

# Se você já tem utilidades de loaders, mantenha:
try:    
    from loaders import carrega_sites, carrega_youtube, carrega_pdf, carrega_csv, carrega_txt
except Exception:
    # stubs simples (evita quebrar se não existir)
    def carrega_sites(url): return f"[Site] {url}"
    def carrega_youtube(url): return f"[YouTube] {url}"
    def carrega_pdf(path): return f"[PDF em {path}]"
    def carrega_csv(path): return f"[CSV em {path}]"
    def carrega_txt(path): return f"[TXT em {path}]"

# ----------------------------
# Função utilitária global – gerar tokens
# ----------------------------
def gerar_token(prefixo: str) -> str:
    """Gera identificadores únicos como AUD-YYYYMMDDHHMMSS-ABC123"""
    import secrets
    rand = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    return f"{prefixo}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{rand}"

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
    st.session_state['auditorias_pendentes'] = []

# ----------------------------
# Banco de Dados (SQLite)
# ----------------------------
DB_DIR = Path("data")
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "oraculo.db"

def get_conn():
    # conexão segura para Streamlit (threads) + timeout maior
    return sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)

def init_db():
    with get_conn() as conn:
        # cria se não existir (já com a coluna elegivel)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS procedimentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            codigo TEXT,
            procedimento TEXT,
            auditoria INTEGER,
            prazo_dias INTEGER,
            origem_aba TEXT,
            elegivel INTEGER,
            UNIQUE(codigo, procedimento)
        );
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS agendamentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            protocolo TEXT,
            nome TEXT,
            nascimento TEXT,
            especialidade TEXT,
            motivo TEXT,
            profissional TEXT,
            calendar_id TEXT,
            inicio TEXT,
            fim TEXT,
            event_id TEXT,
            event_link TEXT,
            created_at TEXT
        );
        """)
        
        conn.commit()
        _migrate_schema(conn)

def _migrate_schema(conn):
    """
    Garante que as colunas necessárias existam mesmo se a tabela foi criada
    numa versão antiga (sem 'elegivel' etc.). Adiciona colunas que faltarem.
    """
    cur = conn.execute("PRAGMA table_info(procedimentos);")
    cols = {row[1] for row in cur.fetchall()}  # nome da coluna é índice 1

    required = {
        "codigo": "TEXT",
        "procedimento": "TEXT",
        "auditoria": "INTEGER",
        "prazo_dias": "INTEGER",
        "origem_aba": "TEXT",
        "elegivel": "INTEGER",
    }

    changed = False
    for name, decl in required.items():
        if name not in cols:
            conn.execute(f"ALTER TABLE procedimentos ADD COLUMN {name} {decl};")
            changed = True

    if changed:
        conn.commit()

def _to_bool_int(x):
    """Converte True/False/None para 1/0/None compatível SQLite."""
    if x is None: return None
    return 1 if bool(x) else 0

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    rename_map = {}
    for c in list(d.columns):
        if c in ['código','cod','codigo tuss','tuss','code']: rename_map[c] = 'codigo'
        if c in ['nome','descricao','descrição']:             rename_map[c] = 'procedimento'
        if c in ['auditoria?','precisa_auditoria','necessita_auditoria','auditoria']: rename_map[c] = 'auditoria'
        if c in ['prazo','prazo (dias)','sla','sla_dias','prazo_dias']:               rename_map[c] = 'prazo_dias'
        if c in ['elegivel','elegível','elegibilidade','autorizavel','autorizável','autorizavel?','autorizável?']:
            rename_map[c] = 'elegivel'
    d = d.rename(columns=rename_map)

    # garante colunas mínimas
    for need in ['codigo','procedimento','auditoria','prazo_dias','elegivel']:
        if need not in d.columns: d[need] = None

    def to_bool(x):
        s = str(x).strip().lower()
        if s in ['true','sim','yes','1']: return True
        if s in ['false','não','nao','no','0','nao elegivel','não elegível','inelegivel','inelegível']: return False
        return None
    def to_int(x):
        try: return int(float(x))
        except: return None

    d['codigo'] = d['codigo'].astype(str).str.extract(r"(\d+)", expand=False)
    d['procedimento'] = d['procedimento'].astype(str).str.strip()
    d['auditoria'] = d['auditoria'].apply(to_bool)
    d['prazo_dias'] = d['prazo_dias'].apply(to_int)
    d['elegivel'] = d['elegivel'].apply(to_bool)

    # default elegibilidade: True (quando coluna ausente/vazia)
    d['elegivel'] = d['elegivel'].apply(lambda v: True if v is None else v)
    return d

def read_planilha_to_df(file_obj) -> pd.DataFrame:
    """
    Lê CSV/XLS/XLSX (todas as abas se Excel) e aplica regra de prazos por aba:
      - OPME: auditoria=True, prazo >= 10
      - Auditoria: auditoria=True, prazo=5 se ausente
      - Sem auditoria: auditoria=False, prazo=None
      - elegivel: default True; respeita valor se vier na planilha
    """
    if file_obj is None:
        return pd.DataFrame()

    name = (getattr(file_obj, "name", "") or "").lower()
    try:
        if name.endswith(".csv"):
            dfs = {"csv": pd.read_csv(file_obj)}
        else:
            dfs = pd.read_excel(file_obj, sheet_name=None)
    except Exception as e:
        st.error(f"Falha ao ler planilha: {e}")
        return pd.DataFrame()

    frames = []
    for sheet_name, df_sheet in dfs.items():
        d = _normalize_cols(df_sheet)
        sname = str(sheet_name).lower()

        # Regras por aba (não mexe em 'elegivel' além do default já aplicado)
        if 'opme' in sname:
            d['auditoria'] = d['auditoria'].fillna(True)
            d['prazo_dias'] = d['prazo_dias'].apply(lambda x: 10 if x is None or (pd.notna(x) and int(float(x)) < 10) else (None if pd.isna(x) else int(float(x))))
            d['_origem_aba'] = 'OPME'
        elif 'auditor' in sname:
            d['auditoria'] = d['auditoria'].fillna(True)
            d['prazo_dias'] = d['prazo_dias'].apply(lambda x: 5 if x is None or pd.isna(x) else int(float(x)))
            d['_origem_aba'] = 'AUDITORIA'
        elif 'sem' in sname or 'nao' in sname or 'não' in sname:
            d['auditoria'] = d['auditoria'].fillna(False)
            d['prazo_dias'] = d['prazo_dias'].apply(lambda x: None)
            d['_origem_aba'] = 'SEM_AUDITORIA'
        else:
            d['auditoria'] = d['auditoria'].fillna(False)
            # normaliza prazo com int ou None
            d['prazo_dias'] = d['prazo_dias'].apply(lambda x: (None if pd.isna(x) else int(float(x))) )
            d['_origem_aba'] = sheet_name

        # limpa procedimentos vazios
        d['procedimento'] = d['procedimento'].astype(str).str.strip()
        d = d[d['procedimento'] != ""]

        # garante colunas e ordem
        d = d[['codigo','procedimento','auditoria','prazo_dias','elegivel','_origem_aba']]

        # converte NaN -> None antes de concatenar
        d = d.where(pd.notnull(d), None)
        frames.append(d)

    if not frames:
        return pd.DataFrame(columns=['codigo','procedimento','auditoria','prazo_dias','elegivel','_origem_aba'])

    df_final = pd.concat(frames, ignore_index=True)

    # remove duplicados
    df_final = df_final.drop_duplicates(subset=['codigo','procedimento'], keep='first')

    # última passada: NaN -> None
    df_final = df_final.where(pd.notnull(df_final), None)
    return df_final

def upsert_planilha_df_into_db(df: pd.DataFrame) -> int:
    """Insere/atualiza (UPSERT) a planilha no banco. Retorna qtd de linhas afetadas."""
    if df.empty:
        return 0

    init_db()

    # função auxiliar de normalização segura
    def _safe_int(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            return int(v)
        except Exception:
            try:
                return int(float(v))
            except Exception:
                return None

    data = []
    skipped = 0
    for _, r in df.iterrows():
        codigo = r.get('codigo')
        if codigo is not None:
            codigo = str(codigo).strip()
            if codigo == "nan" or codigo == "":
                codigo = None

        procedimento = r.get('procedimento')
        if procedimento is not None:
            procedimento = str(procedimento).strip()
        if not procedimento:
            skipped += 1
            continue  # pula linhas sem procedimento

        auditoria = _to_bool_int(r.get('auditoria'))
        prazo = _safe_int(r.get('prazo_dias'))
        origem = r.get('_origem_aba')
        if origem is not None:
            origem = str(origem)
        elegivel = _to_bool_int(r.get('elegivel'))

        data.append((codigo, procedimento, auditoria, prazo, origem, elegivel))

    if not data:
        st.warning("Nenhuma linha válida para inserir (todas com 'procedimento' vazio?).")
        return 0

    rows = 0
    sql = """
        INSERT INTO procedimentos (codigo, procedimento, auditoria, prazo_dias, origem_aba, elegivel)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(codigo, procedimento) DO UPDATE SET
            auditoria=excluded.auditoria,
            prazo_dias=excluded.prazo_dias,
            origem_aba=excluded.origem_aba,
            elegivel=excluded.elegivel;
    """
    try:
        with get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executemany(sql, data)
            rows = conn.total_changes
            conn.commit()
        if skipped:
            st.info(f"⚠️ {skipped} linha(s) ignorada(s) por 'procedimento' vazio.")
        return rows
    except Exception as e:
        # debug: mostra a primeira linha que causou problema
        st.error(f"Erro ao inserir no banco: {e}")
        st.caption("Exemplo de linha preparada (primeira):")
        st.code(repr(data[0]), language="text")
        raise

def query_procedures_from_db(codigos: list, termos_norm: list, fuzzy_cutoff: int = 85) -> pd.DataFrame:
    """
    Busca no DB por:
      - códigos exatos (IN)
      - nomes contendo termos (LIKE)
      - fuzzy (carrega candidatos e aplica difflib)
    Retorna DataFrame com: codigo, procedimento, auditoria (bool), prazo_dias, origem_aba, elegivel (bool)
    """
    init_db()
    with get_conn() as conn:
        parts = []
        params = []

        # por códigos
        if codigos:
            q_marks = ",".join("?" for _ in codigos)
            parts.append(f"(codigo IN ({q_marks}))")
            params.extend(codigos)

        # por termos (LIKE)
        like_sqls = []
        for t in termos_norm:
            like_sqls.append("LOWER(procedimento) LIKE ?")
            params.append(f"%{t}%")
        if like_sqls:
            parts.append("(" + " OR ".join(like_sqls) + ")")

        where = " WHERE " + " OR ".join(parts) if parts else ""
        df = pd.read_sql_query(
            f"SELECT codigo, procedimento, auditoria, prazo_dias, origem_aba, elegivel FROM procedimentos{where};",
            conn, params=params
        )

        # fuzzy extra (compara termos contra todos os procedimentos, útil quando LIKE falha)
        if not df.empty and termos_norm:
            pass
        else:
            df_all = pd.read_sql_query(
                "SELECT codigo, procedimento, auditoria, prazo_dias, origem_aba, elegivel FROM procedimentos;",
                conn
            )
            if not df_all.empty and termos_norm:
                df_all["_proc_norm"] = df_all["procedimento"].astype(str).apply(_normalize_text)
                base = df_all["_proc_norm"].tolist()
                rows = []
                for termo in termos_norm:
                    best = difflib.get_close_matches(termo, base, n=1, cutoff=fuzzy_cutoff/100.0)
                    if best:
                        alvo = best[0]
                        row = df_all[df_all["_proc_norm"] == alvo].iloc[0]
                        rows.append(row[["codigo","procedimento","auditoria","prazo_dias","origem_aba","elegivel"]])
                if rows:
                    df_fz = pd.DataFrame(rows).drop_duplicates()
                    df = pd.concat([df, df_fz], ignore_index=True).drop_duplicates()

        if not df.empty:
            df["auditoria"] = df["auditoria"].apply(lambda v: bool(v) if v is not None else None)
            df["elegivel"] = df["elegivel"].apply(lambda v: bool(v) if v is not None else True)

        return df

# ============================
# Google Calendar Helpers
# ============================
from dateutil import tz
from dateutil.parser import isoparse
from datetime import timedelta
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow

# Escopos: ler/editar calendário
GCAL_SCOPES = ["https://www.googleapis.com/auth/calendar"]

def get_gcal_service() -> "googleapiclient.discovery.Resource":
    """
    Cria/recupera credenciais e devolve o serviço do Google Calendar.
    Requer client_secret.json na raiz do projeto. Salva token.json após OAuth.
    """
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", GCAL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())  # type: ignore
            except Exception:
                pass
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", GCAL_SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    service = build("calendar", "v3", credentials=creds)
    return service

# Mapeamento de profissionais → calendarId (real do Google Calendar)
# Preencha com os emails/IDs dos calendários dos médicos
PROFISSIONAIS = [
    {"nome": "Dra. Ana Souza", "especialidade": "Cardiologia", "calendar_id": "ana.souza@seudominio.com"},
    {"nome": "Dr. Bruno Lima", "especialidade": "Dermatologia", "calendar_id": "bruno.lima@seudominio.com"},
    {"nome": "Dra. Carla Reis", "especialidade": "Clínica Geral", "calendar_id": "carla.reis@seudominio.com"},
]

def listar_profissionais_por_especialidade(especialidade: str):
    return [p for p in PROFISSIONAIS if p["especialidade"].lower() == (especialidade or "").lower()]

def gcal_freebusy_slots(service, calendar_id: str, data: datetime, duracao_min=30,
                        inicio_dia="08:00", fim_dia="17:00", timezone="America/Sao_Paulo"):
    """
    Retorna janelas livres (start, end) para 'data' usando freeBusy do Google Calendar,
    dentro do horário comercial dado.
    """
    tzinfo = tz.gettz(timezone)
    dia_ini = datetime(data.year, data.month, data.day,
                       int(inicio_dia.split(":")[0]), int(inicio_dia.split(":")[1]), tzinfo=tzinfo)
    dia_fim = datetime(data.year, data.month, data.day,
                       int(fim_dia.split(":")[0]), int(fim_dia.split(":")[1]), tzinfo=tzinfo)

    fb = service.freebusy().query(body={
        "timeMin": dia_ini.isoformat(),
        "timeMax": dia_fim.isoformat(),
        "timeZone": timezone,
        "items": [{"id": calendar_id}],
    }).execute()

    busy = fb["calendars"][calendar_id].get("busy", [])
    # construir slots livres
    cursor = dia_ini
    livres = []
    while cursor + timedelta(minutes=duracao_min) <= dia_fim:
        slot_ini = cursor
        slot_fim = cursor + timedelta(minutes=duracao_min)
        conflito = False
        for b in busy:
            b_ini = isoparse(b["start"]).astimezone(tzinfo)
            b_fim = isoparse(b["end"]).astimezone(tzinfo)
            # overlap?
            if not (slot_fim <= b_ini or slot_ini >= b_fim):
                conflito = True
                break
        if not conflito:
            livres.append((slot_ini, slot_fim))
        cursor += timedelta(minutes=duracao_min)
    return livres

def gcal_criar_evento(service, calendar_id: str, resumo: str, descricao: str,
                      inicio: datetime, fim: datetime, timezone="America/Sao_Paulo",
                      convidado_email: str | None = None):
    """
    Cria evento no calendário. Retorna o objeto do evento criado.
    """
    event_body = {
        "summary": resumo,
        "description": descricao,
        "start": {"dateTime": inicio.isoformat(), "timeZone": timezone},
        "end": {"dateTime": fim.isoformat(), "timeZone": timezone},
        "reminders": {"useDefault": True},
    }
    if convidado_email:
        event_body["attendees"] = [{"email": convidado_email}]
    ev = service.events().insert(calendarId=calendar_id, body=event_body, sendUpdates="all").execute()
    return ev

# ----------------------------
# Utilidades gerais (chat)
# ----------------------------
def carrega_contexto_raiz(path_padrao: str = "contexto.txt") -> str:
    try:
        with open(path_padrao, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "(!) Arquivo 'contexto.txt' não encontrado na raiz do projeto."
    except Exception as e:
        return f"(!) Erro ao ler 'contexto.txt': {e}"

def carrega_arquivos(tiposArquivo, arquivo):
    if not arquivo:
        return ""
    try:
        if hasattr(arquivo, "seek"):
            try: arquivo.seek(0)
            except Exception: pass

        if tiposArquivo == 'Site':
            return carrega_sites(arquivo)
        if tiposArquivo == 'Link Youtube':
            return carrega_youtube(arquivo)
        if tiposArquivo == '.PDF':
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
                temp.write(arquivo.read()); name_temp = temp.name
            return carrega_pdf(name_temp)
        if tiposArquivo == '.CSV':
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
                temp.write(arquivo.read()); name_temp = temp.name
            return carrega_csv(name_temp)
        if tiposArquivo == '.TXT':
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
                temp.write(arquivo.read()); name_temp = temp.name
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
            st.error("langchain_openai não instalado."); st.stop()
        return ChatOpenAI(model=modelo, api_key=api_key, base_url=extras.get('base_url'))
    elif prov_info.get('tipo') == 'hf_inference':
        if HuggingFaceEndpoint is None:
            st.error("langchain_community não instalado."); st.stop()
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
    chat = _cria_cliente_chat(st.session_state['provedores'][provedor], modelo, api_key)
    return template | chat

def monta_chain_upload(provedor: str, modelo: str, api_key: str, tiposArquivo: str, arquivo, incluir_contexto_raiz: bool):
    doc_texto = carrega_arquivos(tiposArquivo, arquivo)
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
    chat = _cria_cliente_chat(st.session_state['provedores'][provedor], modelo, api_key)
    return template | chat

# ----------------------------
# OCR do pedido (PDF/Imagem)
# ----------------------------
def _normalize_text(s: str) -> str:
    if s is None: return ""
    s = s.lower().strip()
    s = ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')
    return s

def extrair_texto_pedido(pedido_file) -> str:
    """
    Retorna texto OCR do arquivo enviado (PDF/PNG/JPG).
    - PDF: tenta extrair texto; se a página não tiver texto, tenta OCR com Tesseract (se disponível).
    - Imagem: OCR direto com Tesseract.
    """
    if pedido_file is None:
        return ""

    name = (getattr(pedido_file, "name", "") or "").lower()

    # PDF
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
                            # fallback OCR da página como imagem
                            img = page.to_image(resolution=300).original
                            partes.append(pytesseract.image_to_string(img, lang="por+eng"))
                        else:
                            partes.append("[Página sem texto legível e OCR indisponível]")
        except Exception as e:
            return f"(!) Erro ao processar PDF: {e}"
        return "\n".join(partes).strip()

    # Imagens
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

def extrair_procedimentos_do_pedido(texto_ocr: str) -> str:
    """
    Pega o texto bruto do OCR e retorna somente a seção de exames/procedimentos.
    Heurística:
      - começa quando aparecer uma linha com palavras-chave de INÍCIO (ex.: "exames", "procedimentos", "solicito"...)
      - termina quando aparecer uma linha com palavras-chave de FIM (ex.: "CRM", "Dr", "Assinatura", "Data"...)
    """
    if not texto_ocr:
        return ""

    import unicodedata
    def norm(s: str) -> str:
        s = s.lower().strip()
        s = ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')
        return s

    START_KEYS = {
        "exames", "exame", "procedimentos", "procedimento",
        "exames laboratoriais", "laboratoriais", "laboratorio", "laboratorial",
        "solicito", "solicitacao", "solicitação", "solicitados", "solicitado",
        "itens", "pedidos"
    }
    STOP_KEYS = {
        "crm", "dr.", "dr ", "dra", "assinatura", "carimbo",
        "medico", "médico", "responsavel", "responsável",
        "data:", "observacoes", "observações", "diagnostico", "diagnóstico",
        "cid", "conselho", "crm:"
    }

    linhas = texto_ocr.splitlines()
    capturando = False
    coletadas = []

    for raw in linhas:
        n = norm(raw)

        # decide começar
        if not capturando:
            if any(k in n for k in START_KEYS):
                capturando = True
                # pula a linha de título, normalmente não é um item
                continue
            else:
                continue

        # já capturando → ver se é hora de parar
        if any(k in n for k in STOP_KEYS):
            break

        # limpeza simples de ruído e bullets
        limpo = raw.strip().lstrip("-•–—").strip()
        if not limpo or len(limpo) < 3 or norm(limpo) in START_KEYS:
            continue

        coletadas.append(limpo)

        # trava de segurança para não pegar o documento inteiro
        if len(coletadas) > 100:
            break

    texto = "\n".join(coletadas)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()

# ----------------------------
# Cruzamento Pedido x Banco
# ----------------------------
def detectar_codigos_e_termos(texto: str):
    if not texto: return [], []
    codigos = list(set(re.findall(r"\b\d{4,8}\b", texto)))
    termos_raw = re.findall(r"[A-Za-zÀ-ÿ]{4,}", texto, flags=re.IGNORECASE)
    termos_norm = list(set(_normalize_text(t) for t in termos_raw if len(t) >= 5))
    return codigos, termos_norm

# ----------------------------
# UI: Sidebar (somente Modelos)
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
# Tab: Upload (Chat isolado + Autorização + Banco dentro do Upload)
# ----------------------------
def pagina_upload():
    st.header('📎 Upload de Arquivos / URLs', divider=True)
    up_tabs = st.tabs(["Envio & Chat do Upload", "Autorização de Exames", "🗄️ Banco de Dados"])

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

    # --- SUBTAB 2: Autorização de Exames (consulta SOMENTE o Banco)
    with up_tabs[1]:
        st.write("Envie o **pedido** (PDF/PNG/JPG). A decisão consulta **somente** o banco (carregue a planilha no subtab 🗄️ Banco de Dados).")

        pedido_file = st.file_uploader("Pedido de exame/procedimento (PDF/PNG/JPG)", type=['pdf', 'png', 'jpg', 'jpeg'])
        paciente = st.text_input("Paciente (opcional):")
        doc_paciente = st.text_input("Documento/CPF (opcional):")

        if st.button("▶️ Processar Pedido", use_container_width=True):
            if pedido_file is None:
                st.error("Envie o **pedido**.")
            else:
                # 1) OCR pedido
                texto_ocr = extrair_texto_pedido(pedido_file)

                # 1.1) valida se houve erro de OCR
                if texto_ocr.startswith("(!) "):
                    st.error(texto_ocr)
                    st.stop()

                # 1.2) manter apenas a seção de exames/procedimentos
                texto_pedido = extrair_procedimentos_do_pedido(texto_ocr)

                # *** VALIDAÇÃO NOVA: documento sem seção de exames/procedimentos ***
                if not texto_pedido or texto_pedido.strip() == "" or len(texto_pedido.splitlines()) == 0:
                    st.error("❌ Documento inválido para autorização: não encontramos uma seção de **exames/procedimentos** no arquivo enviado.")
                    st.info("Dica: envie um pedido médico contendo a lista de exames/procedimentos solicitados.")
                    st.stop()

                st.subheader("Texto extraído do pedido (seção de exames/procedimentos)")
                st.code((texto_pedido or "").strip()[:5000], language="text")

                # 2) Extrair códigos/termos e consultar banco
                codigos, termos_norm = detectar_codigos_e_termos(texto_pedido)
                df_res = query_procedures_from_db(codigos, termos_norm, fuzzy_cutoff=85)

                # Caminho A: não encontrado no banco => NÃO AUTORIZADO
                if df_res.empty:
                    codigo_na = gerar_token("NAO-AUT")
                    linhas = [
                        "NÃO AUTORIZADO",
                        f"Código: {codigo_na}",
                        f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                        "Motivo: Procedimento(s) não encontrados no Rol (banco de dados)."
                    ]
                    if paciente: linhas.append(f"Paciente: {paciente}")
                    if doc_paciente: linhas.append(f"Documento: {doc_paciente}")
                    texto_na = "\n".join(linhas)
                    st.error("❌ Não autorizado: procedimento(s) não encontrados no banco.")
                    st.download_button("⬇️ Baixar não autorização", data=texto_na.encode("utf-8"),
                                       file_name=f"nao_autorizado_{codigo_na}.txt", mime="text/plain")
                else:
                    st.subheader("Procedimentos identificados (Banco)")
                    st.dataframe(df_res, use_container_width=True)

                    # Caminho B: encontrado, mas há item(es) não elegíveis => NÃO AUTORIZADO
                    nao_elegiveis = df_res[df_res['elegivel'] == False]
                    if not nao_elegiveis.empty:
                        codigo_na = gerar_token("NAO-AUT")
                        linhas = [
                            "NÃO AUTORIZADO",
                            f"Código: {codigo_na}",
                            f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                            "Motivo: Um ou mais procedimentos NÃO elegíveis conforme Rol (banco).",
                            "\nItens não elegíveis:"
                        ]
                        for _, r in nao_elegiveis.iterrows():
                            linhas.append(f"- {r.get('codigo') or '-'} | {r.get('procedimento') or '-'}")
                        if paciente: linhas.append(f"\nPaciente: {paciente}")
                        if doc_paciente: linhas.append(f"Documento: {doc_paciente}")
                        texto_na = "\n".join(linhas)
                        st.error("❌ Não autorizado: há procedimento(s) não elegíveis no Rol.")
                        st.download_button("⬇️ Baixar não autorização", data=texto_na.encode("utf-8"),
                                           file_name=f"nao_autorizado_{codigo_na}.txt", mime="text/plain")

                    else:
                        # Caminho C: todos elegíveis -> avaliar auditoria
                        precisa_auditoria = df_res['auditoria'].fillna(False).any()

                        if not precisa_auditoria:
                            codigo_aut = gerar_token("AUT")
                            linhas = [
                                "AUTORIZAÇÃO AUTOMÁTICA",
                                f"Código: {codigo_aut}",
                                f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
                            ]
                            if paciente: linhas.append(f"Paciente: {paciente}")
                            if doc_paciente: linhas.append(f"Documento: {doc_paciente}")
                            linhas.append("\nProcedimentos autorizados:")
                            for _, r in df_res.iterrows():
                                linhas.append(f"- {r.get('codigo') or '-'} | {r.get('procedimento') or '-'} (sem auditoria)")
                            linhas.append("\nMotivo: Procedimentos elegíveis e sem exigência de auditoria conforme Rol (banco).")
                            texto_aut = "\n".join(linhas)
                            st.success(f"✅ Autorizado automaticamente. Código: {codigo_aut}")
                            st.download_button("⬇️ Baixar autorização", data=texto_aut.encode("utf-8"),
                                               file_name=f"autorizacao_{codigo_aut}.txt", mime="text/plain")
                        else:
                            # prazo = máximo encontrado
                            prazos = df_res['prazo_dias'].dropna().astype(int)
                            prazo = int(prazos.max()) if not prazos.empty else 5
                            protocolo = gerar_token("AUD")
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

                            linhas = [
                                "PROTOCOLADO PARA AUDITORIA",
                                f"Protocolo: {protocolo}",
                                f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                                f"Prazo estimado de retorno: {prazo} dia(s) úteis.",
                                "\nProcedimentos em auditoria:"
                            ]
                            for _, r in df_res.iterrows():
                                flag = "AUDITORIA" if r.get('auditoria') else "SEM AUDITORIA"
                                origem = r.get('origem_aba') or '-'
                                linhas.append(f"- {r.get('codigo') or '-'} | {r.get('procedimento') or '-'} [{flag}] (aba: {origem})")
                            texto_aud = "\n".join(linhas)

                            st.warning(f"⏳ Necessita auditoria. Protocolo: {protocolo} | Prazo estimado: {prazo} dia(s) úteis")
                            st.download_button("⬇️ Baixar protocolo de auditoria", data=texto_aud.encode("utf-8"),
                                               file_name=f"auditoria_{protocolo}.txt", mime="text/plain")

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
                st.download_button("⬇️ Baixar autorização", data=texto_aut.encode("utf-8"),
                                   file_name=f"autorizacao_{codigo_aut}.txt", mime="text/plain")

    # --- SUBTAB 3: Banco de Dados (dentro do Upload) ---
    with up_tabs[2]:
        st.header("🗄️ Banco de Dados (SQLite)", divider=True)
        init_db()

        st.markdown("Carregue a **planilha oficial** (CSV/XLSX) para inserir/atualizar no banco.")
        plan_db_file = st.file_uploader("Planilha (CSV/XLSX) → Banco", type=['csv','xls','xlsx'], key="db_upload_inside")

        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("⬆️ Inserir/Atualizar planilha no Banco", use_container_width=True, type="primary"):
                if plan_db_file is None:
                    st.error("Envie um arquivo de planilha.")
                else:
                    df = read_planilha_to_df(plan_db_file)
                    if df.empty:
                        st.error("Planilha vazia ou inválida.")
                    else:
                        n = upsert_planilha_df_into_db(df)
                        st.success(f"Planilha inserida/atualizada no banco. Registros afetados: {n}")
        with c2:
            if st.button("🧹 Limpar Tabela de Procedimentos", use_container_width=True):
                with get_conn() as conn:
                    conn.execute("DELETE FROM procedimentos;"); conn.commit()
                st.success("Tabela procedimentos limpa.")

        st.markdown("### 🔎 Amostra do Banco")

        def _table_exists(conn, name: str) -> bool:
            try:
                cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,))
                return cur.fetchone() is not None
            except Exception:
                return False

        try:
            with get_conn() as conn:
                init_db()
                if _table_exists(conn, "procedimentos"):
                    df_preview = pd.read_sql_query(
                        "SELECT codigo, procedimento, auditoria, prazo_dias, elegivel, origem_aba FROM procedimentos LIMIT 50;",
                        conn
                    )
                else:
                    df_preview = pd.DataFrame(columns=["codigo","procedimento","auditoria","prazo_dias","elegivel","origem_aba"])
        except Exception:
            st.info("Banco ainda vazio ou indisponível. Insira a planilha para criar/atualizar os dados.")
            df_preview = pd.DataFrame(columns=["codigo","procedimento","auditoria","prazo_dias","elegivel","origem_aba"])

        st.dataframe(df_preview, use_container_width=True)

# ----------------------------
# Tab: Chat principal (unificado)
# ----------------------------
def pagina_chat_unificada():
    st.header('💬 Oráculo — Chat', divider=True)

    sel = st.session_state.get('selecionado')
    up = st.session_state.get('upload', {})

    colA, colB, colC = st.columns(3)
    with colA: st.write(f"**Provedor:** {sel['provedor'] if sel else '-'}")
    with colB: st.write(f"**Modelo:** {sel['modelo'] if sel else '-'}")
    with colC: st.write(f"**Entrada salva:** {up.get('tipo') or '-'} {'✅' if up.get('arquivo') else '—'}")

    if st.button('Inicializar Oráculo (Unificado)', use_container_width=True):
        if not sel:
            st.error("Selecione um provedor/modelo na sidebar.")
        elif not sel.get('api_key'):
            st.error("API Key vazia. Informe/salve na sidebar.")
        else:
            st.session_state['chain_unificada'] = monta_chain_unificada(
                provedor=sel['provedor'], modelo=sel['modelo'], api_key=sel['api_key'],
                tiposArquivo=up.get('tipo'), arquivo=up.get('arquivo')
            )
            st.success("Oráculo inicializado! (Arquivo + contexto.txt)")

    chain = st.session_state.get('chain_unificada')
    if chain is None:
        st.info("Configure modelos na sidebar e (opcional) salve um arquivo/URL no tab 📎 Upload. Depois clique em **Inicializar Oráculo (Unificado)**.")
        st.stop()

 # historico 
    memoria = st.session_state.get('memoria', MEMORIA_PADRAO)
    for mensagem in memoria.buffer_as_messages:
        st.chat_message(mensagem.type).markdown(mensagem.content)

    entrada = st.chat_input('Pergunte algo para o Oráculo')
    if entrada:
        st.chat_message('human').markdown(entrada)
        caixa_ai = st.chat_message('ai')
        try:
            resposta = caixa_ai.write_stream(chain.stream({
                'input': entrada, 'chat_history': memoria.buffer_as_messages
            }))
        except Exception:
            resp = chain.invoke({'input': entrada, 'chat_history': memoria.buffer_as_messages})
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
    # garante que a tabela exista antes de qualquer SELECT
    init_db()

    with st.sidebar:
        sidebar_modelos()

    # Duas tabs: Chat e Upload (Banco está dentro do Upload)
    t_chat, t_upload = st.tabs(["💬 Chat", "📎 Upload"])
    with t_chat:
        pagina_chat_unificada()
    with t_upload:
        pagina_upload()

if __name__ == '__main__':
    main()
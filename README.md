# Oráculo 🔮 — Chat + Autorização de Exames + Agendamento

> ⚠️ **Requisito de versão**: este projeto requer **Python 3.12.3**.
> Use exatamente essa versão para evitar incompatibilidades de dependências.

## Estrutura do projeto

```
HACKATON/
├── __pycache__/           # cache automático do Python
├── .streamlit/            # configs locais do Streamlit (secrets.toml)
├── .venv/                 # ambiente virtual Python
├── data/                  # banco e exportações locais (não versionado)
├── .gitignore             # arquivos/pastas ignorados no Git
├── contexto.txt           # contexto raiz usado no chat (modo estrito)
├── hackaton.db            # banco SQLite local
├── hackaton.py            # app principal (Streamlit)
├── loaders.py             # funções de carregamento de arquivos/URLs
├── requirements.txt       # dependências do projeto
└── README.md              # este documento
```

## Funcionalidades

* 💬 **Chat Unificado** (Groq, Gemini, OpenAI-compat, HF)
* 📎 **Autorização de Exames** via OCR + banco SQLite
* 🗄️ **Banco de Dados** (rol de procedimentos carregado via planilha)
* 🗓️ **Agendamento de Consultas** com bloqueio automático de horários
* 📆 **Integração opcional com Google Calendar**

## Requisitos

* Python **3.12.3**
* Tesseract OCR instalado no sistema (`pytesseract` depende dele)
* Credenciais do Google (`client_secret.json` na raiz, `token.json` será gerado no primeiro login OAuth)

## Instalação

```bash
# criar ambiente virtual
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
source .venv/bin/activate    # Linux/Mac

# instalar dependências
pip install -r requirements.txt
```

## Execução

```bash
streamlit run hackaton.py
```

O app irá rodar no navegador em `http://localhost:8501`.

## Configuração de chaves

Crie o arquivo `.streamlit/secrets.toml` com suas chaves:

```toml
GROQ_API_KEY = "sua_chave_groq"
GOOGLE_API_KEY = "sua_chave_gemini"
```

Para o Google Calendar:

* `client_secret.json` deve estar na raiz.
* O app criará `token.json` após o fluxo OAuth.

## Observações

* A pasta `data/` contém `oraculo.db` e exportações (txts de autorização/protocolo).
* Arquivos sensíveis (`client_secret.json`, `token.json`, `.db`, `.venv`) estão no `.gitignore`.

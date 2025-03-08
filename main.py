import os
import sys
import asyncio
import time
import base64
import pyperclip
import streamlit as st
from PIL import Image
from pydantic import SecretStr
# langchain
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import HumanMessage,SystemMessage
# browser_use
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.controller.service import Controller
# ログ
from logging import getLogger, Formatter, INFO, shutdown
from logging.handlers import RotatingFileHandler
loggers = {}

# 環境変数の読込
from dotenv import load_dotenv
load_dotenv()

# システムプロンプトを含むディレクトリまでパスを通す
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# セッションステート
if "base64_data" not in st.session_state:
    st.session_state["base64_data"] = None
    
if "image_data" not in st.session_state:
    st.session_state["image_data"] = None

if "uploaded_pic" in st.session_state and st.session_state["uploaded_pic"]:
    st.toast("画像をアップロードしました", icon="📥")
    del st.session_state["uploaded_pic"]

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ロガー生成
def getModuleLogger(moduleName):
    
    if moduleName is None:
        moduleName = __name__
        
    if loggers.get(moduleName):
        return loggers.get(moduleName)
    
    formatter = Formatter('[%(asctime)s | '
                          '%(name)s | '
                          '%(levelname)s] '
                          '%(message)s')
    fileHandler = RotatingFileHandler("feedback.log", maxBytes=5000, backupCount=3)    
    fileHandler.setFormatter(formatter)
    fileHandler.setLevel(INFO)
    logger = getLogger(moduleName)
    logger.setLevel(INFO)
    logger.addHandler(fileHandler)
    logger.propagate = False
    loggers[moduleName] = logger
    return logger


# ロガーKILL
def killLoggers():
    for l in loggers:
        logger = loggers.get(l)
        for h in logger.handlers:
            logger.removeHandler(h)
    shutdown()
    return


# フィードバックログ
def log_feedback(icon,messages):
    logger = getModuleLogger(__name__)
    # フィードバックありがとう表示
    st.toast("フィードバックありがとう！", icon="👌")
    # いいね/いくない判断
    activity = "👍:positive" if icon == "👍" else "👎:negative"
    # ログ出力
    logger.info(activity + "," + messages)
    killLoggers()


# llm初期化
def init_llm(llm_model: str):
    if llm_model == 'gemini':
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                      api_key=SecretStr(os.getenv('GOOGLE_API_KEY')),
                                      temperature=0,)

    else:
        st.error(f'サポートされてないLLMモデルです: {llm_model}',icon="✖")



# エージェント初期化
def init_agent(message, llm):
    controller = Controller()
    browser = Browser(config=BrowserConfig(
                            headless=False,
                            disable_security=True, 
                        )
                      )

    return Agent(
        task=message,
        llm=llm,
        controller=controller,
        browser=browser,
        max_actions_per_step=3,
    ), browser


# エージェント実行    
async def run_agent(agent,max_steps):
    return await agent.run(max_steps=max_steps)


# ページ設定
st.set_page_config(page_title="デモ", page_icon="🤩",layout="wide")
# deployボタン非表示
HIDE_ST_STYLE = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
				        .appview-container .main .block-container{
                            padding-top: 1rem;
                            padding-right: 3rem;
                            padding-left: 3rem;
                            padding-bottom: 1rem;
                        }  
                        .reportview-container {
                            padding-top: 0rem;
                            padding-right: 3rem;
                            padding-left: 3rem;
                            padding-bottom: 0rem;
                        }
                        header[data-testid="stHeader"] {
                            z-index: -1;
                        }
                        div[data-testid="stToolbar"] {
                        z-index: 100;
                        }
                        div[data-testid="stDecoration"] {
                        z-index: 100;
                        }
                </style>
"""
st.markdown(HIDE_ST_STYLE, unsafe_allow_html=True)


# 画像アップロード
@st.dialog("🎨 画像アップロード")
def upload_document():
    st.warning("アップロードしたい画像を選択してください",icon="💡",)
    picture = st.file_uploader("ファイル選択", 
                               type=["jpg", "png", "bmp"], 
                               label_visibility="hidden") 
    
    # 画像アップロード
    if picture:   
        
        # 画像をバイナリに変換
        bytes_data = picture.getvalue()
        # 画像を表示
        image = Image.open(picture)
        st.session_state["uploaded_pic"] = True
        st.image(image, caption='プレビュー') 
        
        #　画像アップロードボタン押す
        if st.button("画像アップロード"):
            # 画像をbase64にエンコード
            base64_data = base64.b64encode(bytes_data).decode('utf-8')
            st.session_state["base64_data"] = base64_data
            st.session_state["image_data"] = image
            st.rerun()

# チャットのアバター
user_avatar = "👩‍💻"
assistant_avatar = "🤖"

# タイトル
st.markdown("### AI Chat Browser-Use 対応版")

# LLMモデル選択
llm_model = st.sidebar.radio("LLMモデル選択:",
                             ["gemini"], 
                             index=0)

# ブラウザ利用モード
browser_use_mode = st.sidebar.radio("Browser-Use:", ["利用しない","利用する"], index=0)

# チャット表示コンテナ
main_container = st.container(height=500)

# チャット表示
for i,message in st.session_state["messages"]:
    
    with main_container.chat_message(
        message["role"],
        avatar=assistant_avatar if message["role"] == "assistant" else user_avatar,
    ):
        
        st.markdown(message["content"],unsafe_allow_html=True)
        col1,col2,col3,col4 = st.columns([1,1,1,8])
        
        if col1.button(label="📋",key="copy_button_" + str(i)):
            pyperclip.copy(message["content"])
            st.success("クリップボードにコピーしました",icon="📋")
            time.sleep(0.5)
            st.rerun()
            
        if message["role"] == "assistant":
            if col2.button(label="👍",key="good_button_" + str(i)):
                log_feedback("👍",message["content"])
            if col3.button(label="👎",key="bag_button_" + str(i)):
                log_feedback("👎",message["content"])


# ボタンコンテナ
buttons_container = st.container()

# チャット入力
prompt = buttons_container.text_area(label="プロンプトを入力してください:", 
                                     placeholder="東京の明日の天気を日本語で教えて", 
                                     height=90)

# 利用しない場合
col1,col2,col3,col4,col5 = buttons_container.columns([2,1.2,1.2,9,3.8],
                                                     vertical_alignment="bottom") 

if browser_use_mode =="利用する":
    # ブラウザ利用モード
    max_step = st.sidebar.radio("max_step:", 
                                [10,15,20,25], 
                                index=0)
    # 利用する場合
    buttons_container.write("画像貼り付け未対応")
    st.session_state["image_data"] = None
    st.session_state["base64_data"] = None     
else:
    with col2:
        if st.button(label="🎨"):
            upload_document()
    with col3:
        if st.button(label="🗑️"):
            st.session_state["image_data"] = None
            st.session_state["base64_data"] = None
            st.rerun()
    with col4:
        # チャットにプレビュー画像を表示
        if st.session_state["image_data"] != None:
            col4.image(st.session_state["image_data"],width=100)
 
# 実行ボタン 
if col1.button(label="実行",type="primary"):
    if prompt == "":
        main_container.error("プロンプトを入力してください",icon="✖")
        time.sleep(2)
        st.rerun()

    # ユーザー
    with main_container.chat_message("user", avatar=user_avatar):  
        if st.session_state["base64_data"] != None:
            base64_data = st.session_state["base64_data"]
            disp_msg = f"""{prompt}<br><img src="data:image/jpeg;base64,{st.session_state["base64_data"]}"/>"""
            # システムプロンプト 
            system_prompt = SystemMessage(content="あなたは親切なAIアシスタントです。ユーザーの質問に日本語でこたえてください。")
            # ユーザープロンプト
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": prompt,
                    },  
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,${base64_data}",
                    },
                ]
            )
        else:
            disp_msg = f"""{prompt}"""
            # システムプロンプト 
            system_prompt = SystemMessage(content="あなたは親切なAIアシスタントです。ユーザーの質問に日本語でこたえてください。")
            # ユーザープロンプト
            message = HumanMessage(
                content=[
                        {
                            "type": "text",
                            "text": prompt,
                        },  
                    ]
            )
        
        st.session_state["messages"].append({"role": "user", "content":  disp_msg})
        st.markdown(disp_msg,unsafe_allow_html=True) 
                
        # AI    
        with main_container.chat_message("assistant", avatar=assistant_avatar):
            llm = init_llm(llm_model)
            # LLM実行
            with st.spinner(text="AI実行中...",show_time=True):
                result = ""
                if browser_use_mode =="利用する":
                    # エージェント初期化
                    agent, browser = init_agent([system_prompt,message], llm)
                    # エージェント実行
                    response = asyncio.run(run_agent(agent,max_step))
                    result = response.final_result()  
                else: 
                    response = llm.invoke([system_prompt,message]) 
                    result = response.content

                st.session_state["messages"].append({"role": "assistant", "content": result})
                st.markdown(result,unsafe_allow_html=True) 
                st.session_state["base64_data"] = None
                st.session_state["image_data"] = None
                st.rerun()

  
# チャット履歴クリア
if col5.button("新しいチャット"):
    st.session_state["base64_data"] = None
    st.session_state["image_data"] = None
    st.session_state["messages"] = []
    st.rerun()
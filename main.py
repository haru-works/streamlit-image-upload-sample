import json
import logging
import streamlit as st  # 1.34.0
import extra_streamlit_components as stx
import time
from datetime import datetime
from streamlit_float import *
from PIL import Image
import io 
import base64

logger = logging.getLogger()
logging.basicConfig(encoding="UTF-8", level=logging.INFO)

st.set_page_config(page_title="デモ", page_icon="🤩")

st.title("🤩 デモ")

cookie_manager = stx.CookieManager(key="cookie_manager")

float_init()




# This function logs the last question and answer in the chat messages
def log_feedback(icon):
    # We display a nice toast
    st.toast("フィードバックありがとう！", icon="👌")

    # We retrieve the last question and answer
    last_messages = json.dumps(st.session_state["messages"][-2:])

    # We record the timestamp
    activity = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": "

    # And include the messages
    activity += "positive" if icon == "👍" else "negative"
    #activity += ": " + last_messages

    # And log everything
    logger.info(activity)


# メッセージセッション
#if "image_data" not in st.session_state:
#    st.session_state["image_data"] = None


@st.dialog("🎨 画像アップロード")
def upload_document():
    st.warning(
        "アップロードしたい画像を選択してください",
        icon="💡",
    )
    picture = st.file_uploader(
        "ファイル選択", type=["jpg", "png", "bmp"], label_visibility="hidden"
    )
    if picture:   
        bytes_data = picture.getvalue()
        image = Image.open(picture)
        st.session_state["uploaded_pic"] = True
        st.image(image, caption='プレビュー') 
        chat_input = st.text_input("画像説明コメント")
        if st.button("画像説明"):
            data = base64.b64encode(bytes_data).decode('utf-8')
            #st.session_state["messages"].append({"role": "user", "content": chat_input})
            st.session_state["messages"].append({"role": "user", "content":  f"""{chat_input}<br><img src="data:image/jpeg;base64,{data}"/>"""})
            #st.session_state["image_data"] = image
            st.rerun()


if "uploaded_pic" in st.session_state and st.session_state["uploaded_pic"]:
    st.toast("画像をアップロードしました", icon="📥")
    del st.session_state["uploaded_pic"]

# メッセージセッション
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# チャットのアバター
user_avatar = "👩‍💻"
assistant_avatar = "🤖"


# コンテナ
main_container = st.container()

# チャット表示
for message in st.session_state["messages"]:
    with main_container.chat_message(
        message["role"],
        avatar=assistant_avatar if message["role"] == "assistant" else user_avatar,
    ):
        st.markdown(message["content"],unsafe_allow_html=True)


# 画像表示
#if "image_data" in st.session_state:
#    if st.session_state["image_data"] != None:
#        main_container.image(st.session_state["image_data"], caption='メイン',use_column_width=True) 


# チャット入力
if prompt := st.chat_input("チャットを入力してください"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with main_container.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)
        with main_container.chat_message("assistant", avatar=assistant_avatar):
            response = "AI:" + prompt
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.markdown(response)




# フロートボタン
action_buttons_container = st.container()
action_buttons_container.float(
    "bottom: 7.2rem;background-color: var(--default-backgroundColor); padding-top: 1rem;"
)
cols_dimensions = [1,1,1]
col3, col5, col6 = action_buttons_container.columns(cols_dimensions)
with col3:
    if st.button("🎨"):
        upload_document()

with col5:
    icon = "👍"
    if st.button(icon):
        log_feedback(icon)

with col6:
    icon = "👎"
    if st.button(icon):
        log_feedback(icon)



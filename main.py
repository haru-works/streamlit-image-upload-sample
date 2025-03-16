import os
import sys
import asyncio
import time
import base64
import shutil
import streamlit as st
from streamlit_scroll_navigation import scroll_navbar
from PIL import Image
from pydantic import SecretStr
# langchain
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import HumanMessage,SystemMessage
# video
from moviepy.editor import VideoFileClip
import cv2
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect import VideoManager
from scenedetect import SceneManager
import tempfile
import math
# audio
#import whisper


# システムプロンプトを含むディレクトリまでパスを通す
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 環境変数の読込
from dotenv import load_dotenv
load_dotenv()


# ページ設定
st.set_page_config(page_title="streamlit demo",layout="wide")

# セッションステート
if "base64_data" not in st.session_state:
    st.session_state["base64_data"] = None
if "image_data" not in st.session_state:
    st.session_state["image_data"] = None 
    
if "video_scene_file_list" not in st.session_state:
    st.session_state["video_scene_file_list"] = []
if "tmp_audio_file_path" not in st.session_state:
    st.session_state["tmp_audio_file_path"] = None
    
if "uploaded_pic" in st.session_state and st.session_state["uploaded_pic"]:
    st.toast("画像をアップロードしました", icon="📥")
    del st.session_state["uploaded_pic"]
if "uploaded_video" in st.session_state and st.session_state["uploaded_video"]:
    st.toast("動画をアップロードしました", icon="📥")
    del st.session_state["uploaded_video"]
    
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    

if "anchor_ids" not in st.session_state:
    st.session_state["anchor_ids"] = []
if "anchor_icons" not in st.session_state:
    st.session_state["anchor_icons"] = []
if "anchor_labels" not in st.session_state:
    st.session_state["anchor_labels"] = []



# 削除用コールバック関数
def delete_scene(target_no):
    if len(st.session_state["video_scene_file_list"]) > 0:
        # 新しい初期値のリスト
        new_values = []
        for j in range(len(st.session_state["video_scene_file_list"])):
            # 削除対象のtarget_noはスキップ
            if j != target_no:
                #print( "削除対象以外を追加 j:" + str(j))
                # 初期値に現在の値を追加
                new_values.append(st.session_state["video_scene_file_list"][j])
            else:
                print("削除対象 target_no:" + str(target_no) + "を除外")     
                os.remove(st.session_state["video_scene_file_list"][j]["video_scene_file_path"])
    
        # リストを更新
        st.session_state["video_scene_file_list"] = new_values

  
  
    
# llm初期化
def init_llm(llm_model: str,temperature):
    if llm_model == 'gemini-1.5-flash':
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                      api_key=SecretStr(os.getenv('GOOGLE_API_KEY')),
                                      temperature=temperature,)
    elif llm_model == 'gemini-2.0-flash':
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                      api_key=SecretStr(os.getenv('GOOGLE_API_KEY')),
                                      temperature=temperature,)
    else:
        st.error(f'サポートされてないLLMモデルです: {llm_model}',icon="✖")



# 画像アップロード
@st.dialog("🎨画像アップロード")
def upload_image():
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



# 動画アップロード
@st.dialog("🎥動画アップロード")
def upload_video():
    st.warning("アップロードしたい動画を選択してください",icon="💡",)
    video = st.file_uploader("ファイル選択", 
                               type=["mp4", "avi"], 
                               label_visibility="hidden") 
    
    # 動画アップロード
    if video:   
        # 一時ファイルとして保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=video.name) as temp_file:
            temp_file.write(video.read())
            temp_file_path = temp_file.name
            # 動画を表示
            st.session_state["uploaded_video"] = True
            st.video(video) 
            
            selected_mode = st.radio(label="モード選択",options=["手動","自動"],horizontal=True)
            
            if selected_mode == "手動":
                seconds_per_frame = st.number_input(label="フレーム数/秒",value=10,step=1) 
            else:
                threshold = st.number_input(label="閾値",value=30,step=1) 
                 
            if st.button("シーン分割"):
                # 初期化
                st.session_state["video_scene_file_list"] = []
                st.session_state["tmp_audio_file_path"] = None 
                st.session_state["anchor_ids"] = []
                st.session_state["anchor_labels"] = []
                st.session_state["anchor_icons"] = [] 
                st.session_state["delete_scene_no"] = None
                shutil.rmtree("tmp_mp3/tmp")
                os.mkdir("tmp_mp3/tmp")       
                shutil.rmtree("tmp_video")
                os.mkdir("tmp_video")   
                
                
                if selected_mode == "手動":
                    audio_path,json_data_list = split_video_into_scenes_manual(video_path=temp_file_path,seconds_per_frame=seconds_per_frame)
                else:
                    audio_path ,json_data_list = split_video_into_scenes_auto(video_path=temp_file_path,threshold=threshold)

                st.session_state["tmp_audio_file_path"] = audio_path
                st.session_state["video_scene_file_list"] = json_data_list
                st.rerun()        
                


# 動画フレーム解析
def split_video_into_scenes_auto(video_path, threshold=27.0):
    with st.spinner("シーン分割中．．．",show_time=True):
        # VideoManager オブジェクトを作成
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        # ビデオのフレームレートを取得
        framerate = video_manager.get_framerate()
        # ContentDetectorを閾値、最小シーン長、フレームスキップと共に追加
        scene_manager.add_detector(ContentDetector(threshold))
        # ビデオとシーンマネージャを関連付け
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        # 検出されたシーンのリストを取得
        scene_list = scene_manager.get_scene_list(base_timecode=video_manager.get_base_timecode(),start_in_scene=True)
        # 各シーンの最初のフレームを画像として保存し、シーンの秒数を計算
        cap = cv2.VideoCapture(video_path)
        json_data_list = []
        base64Frames = []
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            scene_duration = (end_frame - start_frame) / framerate
            procedure = ""
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = cap.read()
            
            if ret:
                scene_no = i+1
                img_path = f'tmp_video/scene_{scene_no}.jpg'
                scene_time_start = scene[0].get_timecode()
                scene_time_end = scene[1].get_timecode()
                _, buffer = cv2.imencode(".jpg", frame) # 文字列にエンコーディング
                base64_data = (base64.b64encode(buffer).decode("utf-8"))
                cv2.imwrite(img_path, frame)
                json_data = {
                    "scene_no":scene_no,
                    "scene_time_start":scene_time_start,
                    "scene_time_end":scene_time_end,
                    "scene_duration":scene_duration,
                    "video_scene_file_path":img_path,
                    "procedure":procedure,
                    "base64_data":base64_data,
                }
                json_data_list.append(json_data)
            cap.release()
            
        # Extract audio from video
        audio_path = f"tmp_mp3/{video_path}.mp3"
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, bitrate="32k")
        clip.audio.close()
        clip.close()

    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")
    return audio_path, json_data_list



# 動画フレーム解析&音声抽出
def split_video_into_scenes_manual(video_path, seconds_per_frame=2):
    
    with st.spinner("シーン分割中…",show_time=True):
        
        base_video_path, _ = os.path.splitext(video_path)
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        print("fps=", fps)
        frames_to_skip = int(fps * seconds_per_frame)
        curr_frame=0
        i=1
        json_data_list = []
        base64Frames = []
        while curr_frame < total_frames - 1:
            scene_no = i
            scene_time_start = ""
            scene_time_end = ""
            scene_duration = ""
            procedure = ""
            img_path = f'tmp_video/scene_{scene_no}.jpg'          
            video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64_data = base64.b64encode(buffer).decode("utf-8")
            cv2.imwrite(img_path, frame)
            curr_frame += frames_to_skip
            json_data = {
                    "scene_no":scene_no,
                    "scene_time_start":scene_time_start,
                    "scene_time_end":scene_time_end,
                    "scene_duration":scene_duration,
                    "video_scene_file_path":img_path,
                    "procedure":procedure,
                    "base64_data":base64_data,
                }
            json_data_list.append(json_data) 
            i=i+1    
        video.release()

        audio_path = f"tmp_mp3/{base_video_path}.mp3"
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, bitrate="32k")
        clip.audio.close()
        clip.close()

        print(f"Extracted {len(base64Frames)} frames")
        print(f"Extracted audio to {audio_path}")
        return audio_path,json_data_list



# 手順書自動作成
def generate_procedure(llm_model,
                       temperature,
                       video_scene_file_list,
                       transcription_text):

    num_requests = len(video_scene_file_list)
    procedure_steps = []
    
    print("video_scene_file_list_len:" + str(len(video_scene_file_list)))

    llm = init_llm(llm_model,temperature)

    for i in range(num_requests):    
        frames_subset = [st.session_state["video_scene_file_list"][i]["base64_data"]]
        # システムプロンプト 
        system_prompt = SystemMessage(content=f"与えられた動画データのシーン{i+1}/{num_requests}と文字起こしデータを元に、説明されている作業の手順をできるだけ詳細に日本語で箇条書きで作成してください。")   
        
        # ユーザープロンプト
        message = HumanMessage(
                        content=[
                            {"type":"text", "text":f"こちらが動画シーン{i+1}/{num_requests}のフレーム画像です。"},
                            *map(lambda x: {"type": "image_url", 
                                "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, frames_subset),
                            {"type": "text", "text": f"音声の文字起こしデータは以下の通りです: {transcription_text}"}
                        ]
                    )
        
        response = llm.invoke([system_prompt,message]) 
        procedure = response.content
        procedure_steps.append(procedure)
        st.session_state["video_scene_file_list"][i]["procedure"] = procedure
        
        print(f"シーン {i+1}/{num_requests} の作業手順:")
        print(procedure)
        print("=" * 40)  # 区切り線を表示

    # システムプロンプト 
    system_prompt = SystemMessage(content=f"""
これまでに生成された各シーンの作業手順をもとに、動画全体の詳細な作業手順書を章建てて構成し、#出力フォーマットで作成してください。
出力フォーマットの前後に余計な説明をつけないでください。
#出力フォーマット:
|No|作業名|詳細手順|
|xxx|xxxx|xxxx|
""")                                        
    # ユーザープロンプト
    message = HumanMessage(
                    content=[
                        {"type":"text", "text":"\n".join(procedure_steps)},
                    ]
                )    
    final_response = llm.invoke([system_prompt,message])     

    main_container2.markdown(final_response.content,unsafe_allow_html=True)



# 説明編集
@st.dialog("📝説明編集")
def delete_scene_note(target_no,scene_note):
    # チャット入力
    update_scene_note = st.text_area(label="📝説明編集", 
                        value=scene_note,
                        key="input_scene_note",
                        height=500)

    scene_note_1,scene_note_2 = st.columns((1,1))
    if scene_note_1.button("更新"):
        st.session_state["video_scene_file_list"][target_no]["procedure"]  = update_scene_note 
        st.rerun()

    if scene_note_2.button("閉じる"):
        st.rerun()




# タブを作成
tab_titles = ['💭通常チャット', '🎥動画解析']
tab1, tab2 = st.tabs(tab_titles)
# LLMモデル選択
llm_model = st.sidebar.radio("🤖LLM Model:",["gemini-1.5-flash","gemini-2.0-flash"], index=0)
temperature = st.sidebar.slider("🌡Temperature:", 0.0, 1.0, 0.0,step=0.1)
# チャットのアバター
user_avatar = "👩‍💻"
assistant_avatar = "🤖"

     
# 各タブにコンテンツを追加
with tab1:
    # タイトル
    #st.markdown("### AI Chat") 
    # メインコンテナ
    main_container1 = st.container(height=680)
    # ボタンコンテナ
    buttons_container1 = st.container()
    # チャット表示
    for message in st.session_state["messages"]:
        with main_container1.chat_message(
            message["role"],
            avatar=assistant_avatar if message["role"] == "assistant" else user_avatar,
        ):
            st.markdown(message["content"],unsafe_allow_html=True)
            
    # システムプロンプト 
    system_prompt = SystemMessage(content="あなたは親切なAIアシスタントです。ユーザーの質問に日本語でこたえてください。")
    # チャット入力
    prompt = buttons_container1.text_area(label="プロンプトを入力してください:", 
                                        placeholder="プロンプトを入力してください", 
                                        key="input_prompt_1",
                                        height=80)
    
    # ボタン
    col1,col2,col3,col4,col7 = buttons_container1.columns([1,1,1,4,1],vertical_alignment="bottom") 
    
    with col1:
        if st.button(label="🎨画像アップロード",key="button_upload_image",type="primary"):
            upload_image()                  
    with col3:
        # チャットにプレビュー画像を表示
        if st.session_state["image_data"] != None:
            col3.image(st.session_state["image_data"],width=100)


    # 実行ボタン 
    if col2.button(label="💭チャット実行",key="button_exec_chat",type="primary"):
        if prompt == "":
            main_container1.error("プロンプトを入力してください",icon="✖")
            st.stop()

        with main_container1.chat_message("user", avatar=user_avatar):  
            if st.session_state["base64_data"] != None:
                base64_data = st.session_state["base64_data"]
                disp_msg = f"""{prompt}<br><img src="data:image/jpeg;base64,{st.session_state["base64_data"]}"/>"""
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
                
                st.session_state["messages"].append({"role": "user", "content":  disp_msg})
                st.markdown(disp_msg,unsafe_allow_html=True) 
                        
                # AI    
                with main_container1.chat_message("assistant", avatar=assistant_avatar):
                    llm = init_llm(llm_model,temperature)
                    # LLM実行
                    with st.spinner(text="AI実行中...",show_time=True):
                        result = ""
                        response = llm.invoke([system_prompt,message]) 
                        result = response.content
                        st.session_state["messages"].append({"role": "assistant", "content": result})
                        st.markdown(result,unsafe_allow_html=True) 
            else:        
                disp_msg = f"""{prompt}"""
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
                with main_container1.chat_message("assistant", avatar=assistant_avatar):
                    llm = init_llm(llm_model,temperature)
                    # LLM実行
                    with st.spinner(text="AI実行中...",show_time=True):
                        response = llm.invoke([system_prompt,message]) 
                        result = response.content
                        st.session_state["messages"].append({"role": "assistant", "content": result})
                        st.markdown(result,unsafe_allow_html=True) 


    # チャット履歴クリア
    if col7.button("🆕新しいチャット",key="button_chat_clear_1"):
        st.session_state["base64_data"] = None
        st.session_state["image_data"] = None
        st.session_state["messages"] = []
        st.rerun()




# 動画解析
with tab2:
    # タイトル
    #st.markdown("### 動画解析")
    # メインコンテナ
    main_container2 = st.container(height=680)
    # ボタンコンテナ
    buttons_container2 = st.container()
    # チャット入力
    prompt = buttons_container2.text_area(label="プロンプトを入力してください:", 
                                        placeholder="プロンプトを入力してください", 
                                        key="prompt_input_2",
                                        height=80)

    # ボタン
    col1,col2,col3,col4,col7 = buttons_container2.columns([1,1,1,4,1],vertical_alignment="bottom") 

    with col1:
        if st.button(label="🎥動画アップロード",key="button_upload_video",type="primary"):
            upload_video()  
    with col4:
        if st.session_state["tmp_audio_file_path"] != None:
            st.audio(st.session_state["tmp_audio_file_path"],autoplay=False)
      
    if len(st.session_state["video_scene_file_list"]) > 0:  
          
        st.session_state["anchor_ids"] = []   
        st.session_state["anchor_labels"] = []       
        st.session_state["anchor_icons"] = []             
        for json_data in st.session_state["video_scene_file_list"]:   
            st.session_state["anchor_ids"].append("scene_no_" + str(json_data["scene_no"]))
            st.session_state["anchor_labels"].append("シーンNO:" + str(json_data["scene_no"]))
            st.session_state["anchor_icons"].append("🖼️")
         
        with st.sidebar:    
            st.subheader("🎥動画シーン一覧")
            scroll_navbar(
                anchor_ids=st.session_state["anchor_ids"],
                anchor_labels=st.session_state["anchor_labels"],
                anchor_icons=st.session_state["anchor_icons"])              
                
        for i,anchor_id,anchor_label in zip(range(len(st.session_state["video_scene_file_list"])),st.session_state["anchor_ids"],st.session_state["anchor_labels"]):   
            main_container2.subheader(anchor_label,anchor=anchor_id)     
            sub_col1,sub_col2 = main_container2.columns((1,1))  
            # シーン画像表示
            with sub_col1:
                sub_col1.image(image=st.session_state["video_scene_file_list"][i]["video_scene_file_path"]) 
                sub_col1.button(label="🗑️シーン削除",key="scene_delete_" + anchor_id,on_click=delete_scene, args=(i, ))
            # シーン説明表示
            with sub_col2:
                scene_text_container = sub_col2.container(border=1,height=450)
                scene_text_container.markdown(st.session_state["video_scene_file_list"][i]["procedure"],unsafe_allow_html=True)
                sub_col2.button(label="📝説明編集",key="scene_note_update_" + anchor_id,on_click=delete_scene_note, args=(i,st.session_state["video_scene_file_list"][i]["procedure"]))

 


    # 実行ボタン 
    if col2.button(label="🎥動画解析",key="button_video",type="primary"):
    
        # ユーザー            
        if len(st.session_state["video_scene_file_list"]) > 0:
            disp_msg = "動画解析を行います。\n\n" + prompt

            # LLM実行
            with col3:
                with st.spinner(text="AI実行中...",show_time=True):  
                    # audio文字起こし
                    transcription_text = ""
                    # 作業手順書を生成
                    procedure = generate_procedure(llm_model=llm_model,
                                                    temperature=temperature,
                                                    video_scene_file_list=st.session_state["video_scene_file_list"] , 
                                                    transcription_text=transcription_text)
                    st.rerun()


        else:
            st.warning("動画をアップロードしてください")



    # チャット履歴クリア
    if col7.button("🆕クリア",key="button_clear_2"):
        st.session_state["video_scene_file_list"] = []
        st.session_state["tmp_audio_file_path"] = None 
        st.session_state["anchor_ids"] = []
        st.session_state["anchor_labels"] = []
        st.session_state["anchor_icons"] = [] 
        
        shutil.rmtree("tmp_mp3/tmp")
        os.mkdir("tmp_mp3/tmp")       
        shutil.rmtree("tmp_video")
        os.mkdir("tmp_video")
        st.rerun()
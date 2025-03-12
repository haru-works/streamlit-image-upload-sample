import os
import sys
import asyncio
import time
import base64
import streamlit as st
from PIL import Image
from pydantic import SecretStr
# langchain
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import HumanMessage,SystemMessage
# video
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from scenedetect import detect,open_video, AdaptiveDetector,SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
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
st.set_page_config(page_title="デモ", page_icon="🤩",layout="wide")

# セッションステート
if "base64_data" not in st.session_state:
    st.session_state["base64_data"] = None
    
if "image_data" not in st.session_state:
    st.session_state["image_data"] = None
     
if "video_scene_file_list" not in st.session_state:
    st.session_state["video_scene_file_list"] = []
    
if "base64_frames" not in st.session_state:
    st.session_state["base64_frames"] = None

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


# llm初期化
def init_llm(llm_model: str):
    if llm_model == 'gemini':
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                      api_key=SecretStr(os.getenv('GOOGLE_API_KEY')),
                                      temperature=0,)

    else:
        st.error(f'サポートされてないLLMモデルです: {llm_model}',icon="✖")


# 画像アップロード
@st.dialog("🎨 画像アップロード")
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
@st.dialog("🎥 動画アップロード")
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
            # 画像アップロードボタン押す
            if st.button("シーン検出"):
                base64Frames, audio_path ,json_data_list = split_video_into_scenes(video_path=temp_file_path,threshold=30)
                #base64Frames, audio_path = process_video(temp_file_path,2)
                st.session_state["base64_frames"] = base64Frames
                st.session_state["tmp_audio_file_path"] = audio_path
                st.session_state["video_scene_file_list"] = json_data_list
                st.rerun()


# 動画フレーム解析
def split_video_into_scenes(video_path, threshold=27.0):
    with st.spinner("シーン検出中…",show_time=True):
        # VideoManager オブジェクトを作成
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
    
        # ビデオのフレームレートを取得
        framerate = video_manager.get_framerate()

        # # ContentDetectorを閾値、最小シーン長、フレームスキップと共に追加
        scene_manager.add_detector(ContentDetector(threshold))

        # ビデオとシーンマネージャを関連付け
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        # 検出されたシーンのリストを取得
        scene_list = scene_manager.get_scene_list(base_timecode=video_manager.get_base_timecode(),start_in_scene=True)
        # 各シーンの最初のフレームを画像として保存し、シーンの秒数を計算
        cap = cv2.VideoCapture(video_path)
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            scene_duration = (end_frame - start_frame) / framerate
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = cap.read()
            base64Frames = []
            if ret:
                img_path = f'tmp_video/scene_{i+1}.jpg'
                scene_no = i+1
                scene_time_start = scene[0].get_timecode()
                scene_time_end = scene[1].get_timecode()
                _, buffer = cv2.imencode(".jpg", frame) # 文字列にエンコーディング
                base64_data = (base64.b64encode(buffer).decode("utf-8"))
                base64Frames.append(base64_data)
                json_data_list = []
                cv2.imwrite(img_path, frame)
                json_data = {
                    "scene_no":scene_no,
                    "scene_time_start":scene_time_start,
                    "scene_time_end":scene_time_end,
                    "scene_duration":scene_duration,
                    "video_scene_file_path":img_path,
                    "base64_data":base64_data
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
    return base64Frames, audio_path, json_data_list



# 動画フレーム解析&音声抽出
def process_video(video_path, seconds_per_frame=2):
    with st.spinner("シーン検出中…",show_time=True):
        base64Frames = []
        base_video_path, _ = os.path.splitext(video_path)

        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        print("fps=", fps)
        frames_to_skip = int(fps * seconds_per_frame)
        curr_frame=0

        while curr_frame < total_frames - 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            curr_frame += frames_to_skip
        video.release()

        audio_path = f"tmp_mp3/{base_video_path}.mp3"
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, bitrate="32k")
        clip.audio.close()
        clip.close()

        print(f"Extracted {len(base64Frames)} frames")
        print(f"Extracted audio to {audio_path}")
        return base64Frames, audio_path


# # 音声解析 streamlitだと使えない API化必要
# async def transcribe_audio(audio_path):
#     model = whisper.load_model("medium")  # 使用するモデルを指定する
#     #start_time = time.time()
#     result = asyncio.run(model.transcribe(audio_path, fp16=False))  # 音声データの文字起こし
#     #end_time = time.time()
#     print(result["text"])  # 文字起こし結果の表示
#     #print(f'実行時間: {end_time - start_time} seconds')
#     return result["text"]


# 手順書自動作成
def generate_procedure(llm_model,base64_frames, transcription_text, max_frames_per_request=30):
    num_requests = math.ceil(len(base64_frames) / max_frames_per_request)
    procedure_steps = []

    llm = init_llm(llm_model)

    for i in range(num_requests):
        start_index = i * max_frames_per_request
        end_index = min((i + 1) * max_frames_per_request, len(base64_frames))
        frames_subset = base64_frames[start_index:end_index]

        # システムプロンプト 
        system_prompt = SystemMessage(content=f"与えられた動画データのセグメント{i+1}/{num_requests}と文字起こしデータを元に、説明されている作業の手順をできるだけ詳細に日本語で箇条書きで作成してください。")   
        # ユーザープロンプト
        message = HumanMessage(
                        content=[
                            {"type":"text", "text":f"こちらが動画セグメント{i+1}/{num_requests}のフレーム画像です。"},
                            *map(lambda x: {"type": "image_url", 
                                "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, frames_subset),
                            {"type": "text", "text": f"音声の文字起こしデータは以下の通りです: {transcription_text}"}
                        ]
                    )
        response = llm.invoke([system_prompt,message]) 
        procedure = response.content
        procedure_steps.append(procedure)

        print(f"セグメント {i+1}/{num_requests} の作業手順:")
        print(procedure)
        print("=" * 40)  # 区切り線を表示
        
        # システムプロンプト 
        system_prompt = SystemMessage(content=f"""
これまでに生成された各セグメントの作業手順をもとに、動画全体の詳細な作業手順書を章建てて構成し、#出力フォーマットで作成してください。
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
        
    return final_response.content


# チャットのアバター
user_avatar = "👩‍💻"
assistant_avatar = "🤖"

# タイトル
st.markdown("### AI Chat Test")

# LLMモデル選択
llm_model = st.sidebar.radio("LLMモデル選択:",["gemini"], index=0)

# チャット表示コンテナ
main_container = st.container(height=500)

# チャット表示
for message in st.session_state["messages"]:
    with main_container.chat_message(
        message["role"],
        avatar=assistant_avatar if message["role"] == "assistant" else user_avatar,
    ):
        st.markdown(message["content"],unsafe_allow_html=True)

if st.session_state["video_scene_file_list"] != []:
    for json_data in st.session_state["video_scene_file_list"]:
        disp_msg = f"""scene_no{json_data["scene_no"]} 開始{json_data["scene_time_start"]} 終了{json_data["scene_time_end"]} 動画の長さ{json_data["scene_duration"]} <br><img  width="300" height="200" src="data:image/jpeg;base64,{json_data["base64_data"]}"/>"""
        main_container.markdown(disp_msg,unsafe_allow_html=True)
 
if st.session_state["tmp_audio_file_path"] != None:
    main_container.audio(st.session_state["tmp_audio_file_path"])
      
        
# ボタンコンテナ
buttons_container = st.container()

# チャット入力
prompt = buttons_container.text_area(label="プロンプトを入力してください:", 
                                     placeholder="東京の明日の天気を日本語で教えて", 
                                     height=90)

# 利用しない場合
col1,col2,col3,col4,col5 = buttons_container.columns([2,1.2,1.2,9,3.8],
                                                     vertical_alignment="bottom") 
with col2:
    if st.button(label="🎨"):
        upload_image()
with col3:
    if st.button(label="🎥"):
        upload_video()                     
with col4:
    if st.button(label="🗑️"):
        st.session_state["image_data"] = None
        st.session_state["base64_data"] = None
        st.session_state["video_scene_file_list"] = []
        st.session_state["base64_frames"] = None
        st.session_state["tmp_audio_file_path"] = None
        st.rerun()
with col5:
    # チャットにプレビュー画像を表示
    if st.session_state["image_data"] != None:
        col5.image(st.session_state["image_data"],width=100)
 
# 実行ボタン 
if col1.button(label="実行",type="primary"):
  
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
             
            st.session_state["messages"].append({"role": "user", "content":  disp_msg})
            st.markdown(disp_msg,unsafe_allow_html=True) 
                    
            # AI    
            with main_container.chat_message("assistant", avatar=assistant_avatar):
                llm = init_llm(llm_model)
                # LLM実行
                with st.spinner(text="AI実行中...",show_time=True):
                    result = ""
                    response = llm.invoke([system_prompt,message]) 
                    result = response.content
                    st.session_state["messages"].append({"role": "assistant", "content": result})
                    st.markdown(result,unsafe_allow_html=True) 

                          
        elif st.session_state["tmp_audio_file_path"] != None and st.session_state["base64_frames"] !=None:
            disp_msg = "動画から手順書作成を行います"
            st.session_state["messages"].append({"role": "user", "content":  disp_msg})
            st.markdown(disp_msg,unsafe_allow_html=True) 
            if st.session_state["video_scene_file_list"] != []:
                for json_data in st.session_state["video_scene_file_list"]:
                    disp_msg = f"""scene_no{json_data["scene_no"]} 開始{json_data["scene_time_start"]} 終了{json_data["scene_time_end"]} 動画の長さ{json_data["scene_duration"]} <br><img  width="300" height="200" src="data:image/jpeg;base64,{json_data["base64_data"]}"/>"""
                    main_container.markdown(disp_msg,unsafe_allow_html=True)
                    st.session_state["messages"].append({"role": "user", "content":  disp_msg})
                    
                        
            
            # LLM実行
            with st.spinner(text="AI実行中...",show_time=True):  
                # 関数を実行して文字起こしを行う
                #transcription_text = asyncio.run(transcribe_audio(st.session_state["tmp_audio_file_path"]))
                transcription_text = ""
                # 作業手順書を生成
                procedure = generate_procedure(llm_model,st.session_state["base64_frames"] , transcription_text, max_frames_per_request=34)
                #回答結果の表示
                st.session_state["messages"].append({"role": "assistant", "content": procedure})
                st.markdown(procedure,unsafe_allow_html=True) 
              
        else:
            if prompt == "":
                main_container.error("プロンプトを入力してください",icon="✖")
                time.sleep(2)
                st.rerun()
                
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
                    response = llm.invoke([system_prompt,message]) 
                    result = response.content
                    st.session_state["messages"].append({"role": "assistant", "content": result})
                    st.markdown(result,unsafe_allow_html=True) 
                    
        # セッションクリア       
        st.session_state["base64_data"] = None
        st.session_state["image_data"] = None
        st.session_state["video_scene_file_list"] = []
        st.session_state["base64_frames"] = None
        st.session_state["tmp_audio_file_path"] = None
        st.rerun()

# チャット履歴クリア
if col5.button("新しいチャット"):
    st.session_state["base64_data"] = None
    st.session_state["image_data"] = None
    st.session_state["messages"] = []
    st.session_state["video_scene_file_list"] = []
    st.session_state["base64_frames"] = None
    st.session_state["tmp_audio_file_path"] = None 
    st.rerun()
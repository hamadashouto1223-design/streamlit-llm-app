from dotenv import load_dotenv
import os
import streamlit as st

# スクリプトのディレクトリを基準にapi.envを読み込む
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, 'api.env')
load_dotenv(env_path)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 環境変数の確認
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error(f"OPENAI_API_KEYが設定されていません。{env_path}ファイルを確認してください。")
    st.stop()

# LLMの初期化
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

def get_llm_response(input_text: str, mode: str) -> str:
    """
    LLMからの回答を取得する関数
    
    Args:
        input_text: ユーザーの入力テキスト
        mode: 選択されたモード（健康 or 動物）
    
    Returns:
        LLMからの回答テキスト
    """
    if mode == "健康に関する専門家とのチャット":
        system_message = SystemMessage(
            content="あなたは健康と医療に関する専門家です。ユーザーの健康に関する質問に、専門的な知識を用いて丁寧に回答してください。ただし、深刻な症状については医療機関の受診を勧めてください。"
        )
    else:
        system_message = SystemMessage(
            content="あなたは動物に関する専門家です。動物の生態、習性、飼育方法などについて、専門的な知識を用いて丁寧に回答してください。"
        )
    
    messages = [system_message, HumanMessage(content=input_text)]
    response = llm.invoke(messages)
    return response.content

st.title("サンプルアプリ③: 専門家LLMチャット")

st.write("##### 動作モード1: 健康に関する専門家LLMとのチャット")
st.write("健康や医療に関する質問に、専門知識を持つLLMが回答します。")
st.write("##### 動作モード2: 動物に関する専門家LLMとのチャット")
st.write("動物の生態や飼育方法などに関する質問に、専門知識を持つLLMが回答します。")

selected_item = st.radio(
    "動作モードを選択してください。",
    ["健康に関する専門家とのチャット", "動物に関する専門家とのチャット"]
)

st.divider()

input_message = st.text_input(label="LLMに送信するメッセージを入力してください。")

if st.button("実行"):
    st.divider()

    if input_message:
        with st.spinner("LLMが回答を生成中..."):
            try:
                response_text = get_llm_response(input_message, selected_item)
                st.write("**LLMの回答:**")
                st.write(response_text)
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
    else:
        st.error("メッセージを入力してから「実行」ボタンを押してください。")

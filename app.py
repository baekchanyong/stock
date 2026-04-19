import streamlit as st
from dotenv import load_dotenv
import os
import sys

# Streamlit Cloud 경로 인식 문제를 방지하기 위해 파일 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 모듈 임포트 (추후 생성할 파일들)
from ui.rule_chat import render_rule_builder
from ui.simulation_dashboard import render_simulation_dashboard

# 환경 변수 로드
load_dotenv()

st.set_page_config(
    page_title="AI Game Tester MVP",
    page_icon="🎲",
    layout="wide"
)

def main():
    st.sidebar.title("⚙️ 설정 및 백업")
    engine_choice = st.sidebar.radio(
        "시뮬레이션 엔진 선택",
        ["LLM 기반 (소셜/대화형 게임)", "수학적 강화학습 (업데이트 예정)"]
    )
    
    st.sidebar.divider()
    st.sidebar.subheader("💾 게임 데이터 입출력")
    
    # 데이터 불러오기
    import json
    uploaded_file = st.sidebar.file_uploader("저장된 게임 파일 불러오기 (.json)", type="json")
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            if "messages" in data:
                st.session_state.messages = data["messages"]
            if "final_rules" in data:
                st.session_state.final_rules = data["final_rules"]
            if "sim_logs_history" in data:
                st.session_state.sim_logs_history = data["sim_logs_history"]
            if "analysis_feedback" in data:
                st.session_state.analysis_feedback = data["analysis_feedback"]
            st.sidebar.success("성공적으로 불러왔습니다!")
        except Exception as e:
            st.sidebar.error("파일 업로드 중 오류가 발생했습니다.")
            
    # 데이터 저장하기
    export_data = {
        "messages": st.session_state.get("messages", []),
        "final_rules": st.session_state.get("final_rules", ""),
        "sim_logs_history": st.session_state.get("sim_logs_history", ""),
        "analysis_feedback": st.session_state.get("analysis_feedback", "")
    }
    json_string = json.dumps(export_data, ensure_ascii=False, indent=2)
    st.sidebar.download_button(
        label="📥 현재 작업 내역 저장하기 (.json)",
        data=json_string,
        file_name="game_tester_backup.json",
        mime="application/json"
    )
    
    tab1, tab2 = st.tabs(["💬 게임 규칙 빌더", "📊 시뮬레이션 및 분석"])
    
    with tab1:
        st.header("게임 규칙 설계 (Rule Builder)")
        st.caption("AI를 통해 게임의 제목, 참가자 수, 직업, 승리 조건을 정리하세요.")
        render_rule_builder()
        
    with tab2:
        st.header("시뮬레이션 결과 및 밸런스 검증")
        if engine_choice == "LLM 기반 (소셜/대화형 게임)":
            render_simulation_dashboard()
        else:
            st.info("수학적 시뮬레이션 엔진은 현재 준비 중입니다.")

if __name__ == "__main__":
    main()

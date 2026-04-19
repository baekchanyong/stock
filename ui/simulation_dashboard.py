import streamlit as st
import time
from core.llm_engine import stream_simulation_match, stream_analyze_simulation_results

def render_simulation_dashboard():
    if "final_rules" not in st.session_state:
        st.warning("👉 아직 확정된 게임 룰이 없습니다! 좌측의 [💬 게임 규칙 빌더] 탭에서 AI와 대화하며 게임을 먼저 완성해주세요.")
        return
        
    st.write("확정된 게임 규칙을 바탕으로 AI 에이전트들이 시뮬레이션을 진행합니다.")
    
    # 기존 백업된 기록이 있다면 화면 상단에 표시
    if "sim_logs_history" not in st.session_state:
        st.session_state.sim_logs_history = ""
    if "analysis_feedback" not in st.session_state:
        st.session_state.analysis_feedback = ""
        
    if st.session_state.sim_logs_history != "":
        with st.expander("이전에 진행된(불러온) 시뮬레이션 전체 로그 보기"):
            st.markdown(st.session_state.sim_logs_history)
            
    if st.session_state.analysis_feedback != "":
        st.subheader("📈 통계 및 밸런스 분석 피드백 (백업 기반)")
        st.info("아래는 불러온 파일에 내장되어 있던 규칙 분석 피드백입니다.")
        st.markdown(st.session_state.analysis_feedback)
        st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        num_games = st.number_input("시뮬레이션 진행 판수", min_value=1, max_value=50, value=5)
    with col2:
        st.write("")
        st.write("")
        start_btn = st.button("🚀 시뮬레이션 시작", type="primary")
        
    st.divider()
    
    # 상태 관리 (진행/중단)
    if "simulating" not in st.session_state:
        st.session_state.simulating = False
    
    # 멈춤 처리용 변수
    stop_signal = False
    
    if start_btn:
        st.session_state.simulating = True
        
    if st.session_state.simulating:
        stop_btn = st.button("🛑 중단 (진행된 결과 표시)")
        if stop_btn:
            stop_signal = True
            st.session_state.simulating = False
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_box = st.empty()
        
        log_box = st.empty()
        
        # 새 시뮬레이션이 시작될 경우 이번 회차의 로그를 관리할 변수
        current_session_logs = ""
        
        for i in range(num_games):
            if stop_signal:
                status_text.warning("시뮬레이션이 사용자에 의해 중단되었습니다.")
                break
                
            progress_bar.progress((i + 1) / num_games)
            status_text.text(f"시뮬레이션 진행 중... {i+1}번째 판 실시간 중계 중입니다. 화면을 확인해주세요!")
            
            # 실제 LLM 시뮬레이션 호출 (실시간 스트리밍)
            with st.expander(f"🎲 [신규 게임 {i+1}] 실시간 중계 중...", expanded=True):
                match_result_text = st.write_stream(stream_simulation_match(st.session_state.final_rules))
            
            # 새 세션 로그 및 전역(Session State) 로그에 동시 반영
            current_session_logs += f"### [신규 게임 {i+1} 요약]\n" + match_result_text + "\n\n"
            st.session_state.sim_logs_history += f"### [게임 {i+1} 요약]\n" + match_result_text + "\n\n"
            
        if not stop_signal:
            progress_bar.progress(1.0)
            status_text.success("선택한 모든 판수의 테스트가 완료되었습니다!")
            st.session_state.simulating = False
            
        # 1판 이상 진행되었으면 최종 밸런스 분석 실행
        if current_session_logs != "":
            st.subheader("📈 최신 시뮬레이션 결과 및 밸런스 분석 피드백")
            st.info("AI가 그동안 진행된 플레이 기록을 분석하여 밸런스 구멍이나 필승법을 도출하고 있습니다.")
            st.session_state.analysis_feedback = st.write_stream(
                stream_analyze_simulation_results(st.session_state.final_rules, st.session_state.sim_logs_history)
            )

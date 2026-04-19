import streamlit as st
from core.llm_engine import get_chat_response, stream_chat_response, stream_generate_content

def render_rule_builder():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # 초기 환영 메시지
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "안녕하세요! 저는 게임 디자인 마스터입니다. 어떤 종류의 게임(예: 마피아, 먹이사슬 등)을 만드려고 하시나요? 참가 인원수와 대략적인 아이디어를 먼저 말씀해 주시면 체계적으로 룰을 세팅해 드릴게요."
        })
        
    # 기존 채팅 내역 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # 사용자 입력 처리
    prompt = st.chat_input("방어막 아이템을 가진 역할도 하나 추가하고 싶은데 룰을 어떻게 짜는게 좋을까?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("규칙을 분석하고 답변을 준비 중입니다... 👀"):
                response = st.write_stream(stream_chat_response(st.session_state.messages))
            st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("📝 현재 룰 조율 완료 및 시뮬레이터로 전송", type="primary"):
        st.info("🔄 진행 알림: 흩어진 대화를 모아 최종 '공식 룰북'으로 정리하여 작성 중입니다. (10초~30초 소요)")
        with st.container(border=True):
            summary_prompt = "지금까지 우리가 대화한 내용을 바탕으로, 이 게임의 전체 규칙(제목, 인원수, 직업과 능력, 낮과 밤 등 턴 진행 방식, 최종 승리 조건)을 완전무결한 '공식 룰북 매뉴얼' 형태로 출력해줘. 시뮬레이션 AI가 이 룰북만 보고 게임을 빈틈없이 진행할 수 있어야 해."
            
            temp_context = st.session_state.messages.copy()
            temp_context.append({"role": "user", "content": summary_prompt})
            
            # 여기서 스트리밍으로 룰을 작성하는 과정을 사용자에게 시각적으로 보여줍니다.
            final_rules = st.write_stream(stream_chat_response(temp_context))
            
            # 최종 룰을 세션에 저장
            st.session_state.final_rules = final_rules
            st.session_state.messages.append({"role": "user", "content": "여기까지의 룰을 확정해줘."})
            st.session_state.messages.append({"role": "assistant", "content": "완성된 게임 룰북은 다음과 같습니다:\n\n" + final_rules})
            
        st.success("✅ 규칙이 성공적으로 확정되었습니다! 상단의 [📊 시뮬레이션 및 분석] 탭으로 이동하세요.")

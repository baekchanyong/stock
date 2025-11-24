import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
import time
import requests
from datetime import datetime, timedelta

# --- 설정 ---
DB_FILE = "stock_analysis_v42.csv"

st.set_page_config(page_title="V42 가치투자 분석기", page_icon="📡", layout="wide")

# --- 헬퍼 함수 ---
def to_float(val):
    try:
        if pd.isna(val) or val == '' or str(val).strip() == '-': return 0.0
        return float(str(val).replace(',', '').replace('%', ''))
    except: return 0.0

# --- [핵심 수정] 금리 크롤링 엔진 강화 ---
def get_current_bond_yield():
    """
    네이버 금융에서 BBB- 회사채 금리를 3단계로 집요하게 찾아냅니다.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    
    # 시도할 URL 목록 (메인 -> 금리상세)
    urls = [
        "https://finance.naver.com/marketindex/",
        "https://finance.naver.com/marketindex/interestList.naver"
    ]
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=5)
            # 인코딩 자동 감지 및 설정 (cp949 or euc-kr)
            response.encoding = 'cp949' 
            
            # 테이블 파싱
            dfs = pd.read_html(response.text)
            
            for df in dfs:
                # 데이터프레임을 문자열로 변환해 '회사채' 키워드 확인
                if '회사채' in df.to_string() or 'BBB' in df.to_string():
                    for idx, row in df.iterrows():
                        # 라벨 컬럼(보통 첫번째)
                        label = str(row.iloc[0])
                        
                        # 'BBB-' 키워드가 포함된 행 찾기
                        if 'BBB-' in label or ('회사채' in label and 'BBB' in label):
                            # 보통 두 번째 컬럼이 현재 금리
                            val = to_float(row.iloc[1])
                            if val > 0:
                                return val
        except:
            continue # 다음 URL 시도
            
    return None # 모든 시도 실패

# --- 펀더멘털 크롤링 (기존 유지) ---
def get_fundamentals(code):
    try:
        target_code = code
        if len(code) == 6 and code.isdigit() and not code.endswith('0'):
            target_code = code[:-1] + '0'
        
        url = f"https://finance.naver.com/item/main.naver?code={target_code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=3)
        dfs = pd.read_html(response.text, encoding='cp949')
        
        eps, bps = 0.0, 0.0
        for df in dfs:
            df_str = df.to_string()
            if 'EPS' in df_str or 'BPS' in df_str:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if c[0] == c[1] else f"{c[0]}_{c[1]}" for c in df.columns]
                for idx, row in df.iterrows():
                    row_str = str(row.iloc[0])
                    if 'EPS' in row_str or '주당순이익' in row_str:
                        values = row.iloc[1:].tolist()
                        for v in reversed(values):
                            val = to_float(v)
                            if val > 0: 
                                eps = val
                                break
                    if 'BPS' in row_str or '주당순자산' in row_str:
                        values = row.iloc[1:].tolist()
                        for v in reversed(values):
                            val = to_float(v)
                            if val > 0: 
                                bps = val
                                break
                if eps > 0 and bps > 0: break
        return eps, bps
    except: return 0, 0

# --- 공포탐욕지수 ---
def calculate_fear_greed(df):
    if len(df) < 30: return 50
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    ma20 = df['Close'].rolling(window=20).mean()
    disparity = (df['Close'] / ma20) * 100
    disparity_score = disparity.apply(lambda x: 0 if x < 90 else (100 if x > 110 else (x - 90) * 5))
    try:
        val = (rsi.iloc[-1] * 0.5) + (disparity_score.iloc[-1] * 0.5)
        return 50 if pd.isna(val) else val
    except: return 50

# --- CSV 저장 ---
def save_to_csv(data):
    df = pd.DataFrame(data)
    if not os.path.exists(DB_FILE):
        df.to_csv(DB_FILE, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(DB_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

# --- 분석 실행 함수 ---
def run_analysis_core(target_stocks, applied_rate, status_text, progress_bar):
    today_str = datetime.now().strftime('%Y-%m-%d')
    chart_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    
    total = len(target_stocks)
    new_data = []
    
    for step, (idx, row) in enumerate(target_stocks.iterrows()):
        code = str(row['Code'])
        name = row['Name']
        
        if name in ["맥쿼리인프라", "SK리츠"]: continue
        
        progress_bar.progress(min((step + 1) / total, 1.0))
        status_text.text(f"⏳ [{step+1}/{total}] {name} 정밀 분석 중...")
        
        try:
            current_price = to_float(row.get('Close', 0))
            
            eps, bps = get_fundamentals(code)
            if eps == 0: eps = to_float(row.get('EPS', 0))
            if bps == 0: bps = to_float(row.get('BPS', 0))
            
            roe = 0
            if bps > 0: roe = (eps / bps) * 100
            
            time.sleep(0.05)
            fg_score = 50
            try:
                df_chart = fdr.DataReader(code, chart_start, today_str)
                if not df_chart.empty:
                    fg_score = calculate_fear_greed(df_chart)
            except: pass

            # S-RIM 계산
            k = applied_rate / 100
            target_pbr = max(0.3, roe / applied_rate)
            sentiment_factor = 1 + ((50 - fg_score) / 50 * 0.1)
            fair_price = bps * target_pbr * sentiment_factor
            
            gap = 0
            if current_price > 0:
                gap = (fair_price - current_price) / current_price * 100
            
            data_row = {
                '종목코드': code,
                '종목명': name,
                '현재가': round(current_price, 0),
                '적정주가': round(fair_price, 0),
                '괴리율': round(gap, 2),
                'ROE(%)': round(roe, 2),
                'EPS': round(eps, 0),
                'BPS': round(bps, 0),
                '공포지수': round(fg_score, 1)
            }
            new_data.append(data_row)
            
            if len(new_data) >= 5:
                save_to_csv(new_data)
                new_data = []
        except: continue

    if new_data: save_to_csv(new_data)
    progress_bar.empty()
    return True

# --- 메인 UI ---

st.title("📡 V42 가치투자 분석기 (금리수집 강화)")

with st.expander("📘 **적정주가 산출 방식 및 금리 안내 (Click)**", expanded=True):
    st.info("💡 **분석 시작**을 누르면 실시간 금리를 3단계로 정밀 조회합니다.")
    # 수식 오류 방지를 위해 안전하게 분리
    latex_formula = r"\text{적정주가} = \text{BPS} \times \frac{\text{ROE}}{\text{실시간금리}} \times \text{심리보정}"
    st.latex(latex_formula)

st.divider()

# --- 1. 설정 영역 ---
st.header("1. 분석 대상 설정")

# 분석 모드 선택
mode = st.radio("분석 모드 선택", ["🏆 시가총액 상위 종목 분석", "🔍 특정 종목 검색/추천 분석"], horizontal=True)

target_stocks = pd.DataFrame()

# 모드 1: 시가총액 상위
if mode == "🏆 시가총액 상위 종목 분석":
    st.write("📊 **분석할 상위 종목 수 설정**")
    
    if 'stock_count' not in st.session_state:
        st.session_state.stock_count = 200

    def update_slider():
        st.session_state.stock_count = st.session_state.slider_widget
    
    def apply_manual_input():
        st.session_state.stock_count = st.session_state.num_input

    c1, c2 = st.columns([3, 1])
    with c1:
        st.slider("슬라이더", 10, 500, key='slider_widget', value=st.session_state.stock_count, on_change=update_slider)
    with c2:
        st.number_input("직접 입력", 10, 500, key='num_input', value=st.session_state.stock_count, on_change=apply_manual_input)
        
    if st.button("✅ 위 수치 적용"):
        apply_manual_input()
        st.session_state.slider_widget = st.session_state.stock_count # 동기화
        st.success(f"상위 {st.session_state.stock_count}개 종목으로 설정되었습니다.")

# 모드 2: 검색
elif mode == "🔍 특정 종목 검색/추천 분석":
    search_query = st.text_input("분석하고 싶은 종목명을 입력하세요 (예: 삼성, 현대)", placeholder="종목명 입력 후 Enter")
    
    if search_query:
        with st.spinner("종목 리스트 검색 중..."):
            try:
                df_krx = fdr.StockListing('KRX')
                search_results = df_krx[df_krx['Name'].str.contains(search_query, case=False)]
                
                if search_results.empty:
                    st.error(f"❌ '{search_query}'에 대한 검색 결과가 없습니다.")
                else:
                    st.success(f"🔎 총 {len(search_results)}개의 종목을 찾았습니다.")
                    selected_stocks = st.multiselect(
                        "분석할 종목을 선택해주세요",
                        search_results['Name'].tolist(),
                        default=search_results['Name'].tolist()[:5]
                    )
                    target_stocks = search_results[search_results['Name'].isin(selected_stocks)]
                    if not target_stocks.empty:
                        st.write("👇 선택된 종목 리스트")
                        st.dataframe(target_stocks[['Code', 'Name', 'Market', 'Close']])
            except Exception as e:
                st.error(f"오류 발생: {e}")

# --- 2. 실행 및 결과 ---
st.divider()
st.header("2. 분석 실행")

if st.button("▶️ 분석 시작 (Start Analysis)", type="primary", use_container_width=True):
    
    if mode == "🏆 시가총액 상위 종목 분석":
        with st.spinner("상위 종목 리스트 가져오는 중..."):
            df_krx = fdr.StockListing('KRX')
            df_krx = df_krx[df_krx['Market'].isin(['KOSPI'])]
            df_krx = df_krx.sort_values(by='Marcap', ascending=False)
            final_target = df_krx.head(st.session_state.stock_count)
    else:
        if target_stocks.empty:
            st.warning("⚠️ 분석할 종목이 없습니다. 종목을 선택해주세요.")
            st.stop()
        final_target = target_stocks

    # [금리 크롤링]
    status_box = st.empty()
    status_box.info("📡 네이버 금융에서 실시간 금리(BBB-) 정밀 조회 중...")
    
    real_rate = get_current_bond_yield()
    applied_rate = 8.0
    
    if real_rate:
        applied_rate = real_rate
        status_box.success(f"✅ 조회 성공! 현재 시장 금리 **{applied_rate}%**를 적용합니다.")
    else:
        status_box.error(f"❌ 실시간 금리 조회 실패! 부득이하게 **기본값 {applied_rate}%**를 적용합니다.")
    
    time.sleep(1.5)
    
    p_bar = st.progress(0)
    run_analysis_core(final_target, applied_rate, status_box, p_bar)
    
    if real_rate:
        status_box.success(f"✅ 분석 완료! (적용된 실시간 금리: {applied_rate}%)")
    else:
        status_box.warning(f"⚠️ 분석 완료! (적용된 기본 금리: {applied_rate}%)")

st.divider()

# 결과 표
st.header("🏆 분석 결과")

sort_option = st.radio("정렬 기준", ["괴리율 높은 순", "ROE 높은 순", "공포지수 낮은 순"], horizontal=True)

if st.button("🔄 결과 새로고침"): st.rerun()

if os.path.exists(DB_FILE):
    try:
        df_res = pd.read_csv(DB_FILE)
        for col in ['현재가', '적정주가', '괴리율', 'EPS', 'BPS', 'ROE(%)', '공포지수']:
            if col in df_res.columns: df_res[col] = df_res[col].apply(to_float)

        df_res = df_res.drop_duplicates(['종목코드'], keep='last')
        
        if not df_res.empty:
            if "괴리율" in sort_option:
                df_res = df_res.sort_values(by='괴리율', ascending=False)
            elif "ROE" in sort_option:
                df_res = df_res.sort_values(by='ROE(%)', ascending=False)
            elif "공포지수" in sort_option:
                df_res = df_res.sort_values(by='공포지수', ascending=True)

            df_res = df_res.reset_index(drop=True)
            df_res.index += 1
            
            top = df_res.iloc[0]
            st.info(f"🥇 **1위: {top['종목명']}** | 괴리율: {top['괴리율']}% | 적정가: {top['적정주가']:,.0f}원")

            st.dataframe(
                df_res[['종목명', '현재가', '적정주가', '괴리율', 'ROE(%)', 'EPS', 'BPS', '공포지수']].style.applymap(
                    lambda x: 'color: red; font-weight: bold;' if x > 20 else ('color: blue;' if x < 0 else 'color: black;'), 
                    subset=['괴리율']
                ).format("{:,.0f}", subset=['현재가', '적정주가', 'EPS', 'BPS']),
                height=600,
                use_container_width=True
            )
        else: st.warning("결과 데이터가 없습니다.")
    except Exception as e: st.error(f"결과 로드 중 오류: {e}")
else: st.info("👈 위에서 [분석 시작] 버튼을 눌러주세요.")

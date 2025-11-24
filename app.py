import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
import time
import requests
from datetime import datetime, timedelta

# --- 설정 ---
DB_FILE = "stock_analysis_v38.csv"

st.set_page_config(page_title="V38 실시간 금리 연동 분석기", page_icon="📡", layout="wide")

# --- 헬퍼 함수 ---
def to_float(val):
    try:
        if pd.isna(val) or val == '' or str(val).strip() == '-': return 0.0
        return float(str(val).replace(',', '').replace('%', ''))
    except: return 0.0

# --- [NEW] 실시간 채권 금리 크롤링 ---
def get_current_bond_yield():
    """
    네이버 금융 시장지표에서 'BBB- 회사채 금리'를 가져옵니다.
    실패 시 기본값 8.0% 반환
    """
    try:
        url = "https://finance.naver.com/marketindex/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        dfs = pd.read_html(response.text, encoding='cp949')
        
        # 보통 금리 표는 뒤쪽에 위치함. '회사채' 키워드 찾기
        for df in dfs:
            if '회사채' in df.to_string() or 'CD' in df.to_string():
                # 데이터프레임 순회
                for idx, row in df.iterrows():
                    # 라벨 컬럼(보통 0번) 확인
                    label = str(row.iloc[0])
                    if '회사채' in label and 'BBB-' in label:
                        yield_val = to_float(row.iloc[1])
                        if yield_val > 0:
                            return yield_val
        return 8.0 # 못 찾으면 기본값
    except:
        return 8.0

# --- 네이버 금융 펀더멘털 크롤링 ---
def get_fundamentals(code):
    try:
        target_code = code
        if len(code) == 6 and code.isdigit() and not code.endswith('0'):
            target_code = code[:-1] + '0'
        
        url = f"https://finance.naver.com/item/main.naver?code={target_code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        dfs = pd.read_html(response.text, encoding='cp949')
        
        eps = 0.0
        bps = 0.0
        
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

# --- 분석 프로세스 ---
def run_srim_analysis(target_num, applied_rate, status_text, progress_bar):
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    status_text.info(f"📡 적용 금리 {applied_rate}%를 기준으로 S-RIM 적정주가를 계산합니다...")

    try:
        df_krx = fdr.StockListing('KRX')
        df_krx = df_krx[df_krx['Market'].isin(['KOSPI'])]
        df_krx = df_krx.sort_values(by='Marcap', ascending=False)
        target_stocks = df_krx.head(target_num)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return

    if os.path.exists(DB_FILE): os.remove(DB_FILE)

    total = len(target_stocks)
    new_data = []
    chart_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    for step, (idx, row) in enumerate(target_stocks.iterrows()):
        code = str(row['Code'])
        name = row['Name']
        
        if name in ["맥쿼리인프라", "SK리츠"]: continue
        
        progress_bar.progress(min((step + 1) / total, 1.0))
        status_text.text(f"⏳ [{step+1}/{total}] {name} 분석 중...")
        
        try:
            current_price = to_float(row.get('Close', 0))
            
            # 1. 펀더멘털 (실시간 크롤링)
            eps, bps = get_fundamentals(code)
            if eps == 0: eps = to_float(row.get('EPS', 0))
            if bps == 0: bps = to_float(row.get('BPS', 0))
            
            roe = 0
            if bps > 0: roe = (eps / bps) * 100
            
            # 2. 공포지수
            time.sleep(0.05)
            fg_score = 50
            try:
                df_chart = fdr.DataReader(code, chart_start, today_str)
                if not df_chart.empty:
                    fg_score = calculate_fear_greed(df_chart)
            except: pass

            # 3. S-RIM 적정주가 계산
            # k = 요구수익률 (실시간 금리 반영)
            k = applied_rate / 100
            
            # 적정 PBR = ROE / k (이익률이 금리보다 높아야 PBR 1배 이상 받음)
            # 최소 0.3배 방어 (망하지 않을 기업 가정)
            target_pbr = max(0.3, roe / applied_rate)
            
            # 심리 보정
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
            
            if len(new_data) >= 10:
                save_to_csv(new_data)
                new_data = []
        except: continue

    if new_data: save_to_csv(new_data)
    progress_bar.empty()
    return True

# --- 메인 UI ---

st.title("📡 V38 실시간 금리 연동 가치투자 분석기")

# 실시간 금리 가져오기 (캐싱)
if 'market_rate' not in st.session_state:
    with st.spinner("실시간 시장 금리(BBB-)를 조회 중입니다..."):
        st.session_state.market_rate = get_current_bond_yield()

current_rate_display = st.session_state.market_rate

with st.expander("📘 **[필독] 실시간 금리 반영 원리 (Click)**", expanded=True):
    st.markdown(f"""
    ##### 1. 기준 지표: BBB- 등급 회사채 금리
    * **현재 조회된 시장 금리:** **{current_rate_display}%**
    * **의미:** 투자자가 주식 투자 시 감수하는 위험에 대해 요구하는 **최소한의 수익률**입니다.
    * 금리가 오르면 $\\rightarrow$ 요구수익률 상승 $\\rightarrow$ 적정주가 하락 (보수적 평가)
    * 금리가 내리면 $\\rightarrow$ 요구수익률 하락 $\\rightarrow$ 적정주가 상승 (공격적 평가)
    
    ##### 2. 산출 공식 (S-RIM 응용)
    $$ \\text{적정주가} = \\text{BPS} \\times \\frac{\\text{ROE}}{\\text{실시간금리}({current_rate_display}\\%)} \\times \\text{심리보정} $$
    """)

st.divider()

# 설정 영역
st.header("1. 분석 조건 설정")

col1, col2 = st.columns(2)
with col1:
    # 금리 선택 (자동 vs 수동)
    rate_option = st.radio("금리 설정 방식", ["실시간 시장 금리 사용", "수동 입력"], horizontal=True)
    
    if rate_option == "실시간 시장 금리 사용":
        final_rate = current_rate_display
        st.success(f"✅ 현재 시장 금리 **{final_rate}%**를 적용합니다.")
    else:
        final_rate = st.number_input("희망 기대수익률 (%)", 1.0, 30.0, 8.0, 0.1)
        st.info(f"사용자가 설정한 **{final_rate}%**를 적용합니다.")

with col2:
    target_count = st.slider("분석 종목 수", 10, 300, 200)

if st.button("▶️ 분석 시작 (Start)", type="primary", use_container_width=True):
    status_box = st.empty()
    p_bar = st.progress(0)
    is_done = run_srim_analysis(target_count, final_rate, status_box, p_bar)
    if is_done:
        status_box.success(f"✅ 분석 완료! (적용 금리: {final_rate}%)")

st.divider()

# 결과 영역
st.header("🏆 가치투자 추천 순위")

sort_option = st.radio(
    "🔀 정렬 기준", 
    ["괴리율 높은 순 (저평가)", "💎 ROE 높은 순 (고수익)", "📉 낙폭 과대 순 (공포)"],
    horizontal=True
)

if st.button("🔄 결과 표 새로고침"): st.rerun()

if os.path.exists(DB_FILE):
    try:
        df_res = pd.read_csv(DB_FILE)
        for col in ['현재가', '적정주가', '괴리율', 'EPS', 'BPS', 'ROE(%)', '공포지수']:
            if col in df_res.columns: df_res[col] = df_res[col].apply(to_float)

        df_res = df_res.drop_duplicates(['종목코드'], keep='last')
        df_res = df_res[df_res['적정주가'] > 0]
        
        if not df_res.empty:
            if "괴리율" in sort_option:
                df_res = df_res.sort_values(by='괴리율', ascending=False)
            elif "ROE" in sort_option:
                df_res = df_res.sort_values(by='ROE(%)', ascending=False)
            elif "낙폭" in sort_option:
                df_res = df_res.sort_values(by='공포지수', ascending=True)

            df_res = df_res.reset_index(drop=True)
            df_res.index += 1
            df_res.index.name = "순번"
            
            search_term = st.text_input("🔍 결과 내 검색", placeholder="종목명")
            if search_term:
                df_res = df_res[df_res['종목명'].str.contains(search_term, na=False)]

            if not df_res.empty:
                top = df_res.iloc[0]
                st.info(f"🥇 **1위: {top['종목명']}** | ROE: {top['ROE(%)']}% | 금리대비 초과수익: {top['ROE(%)'] - final_rate:.1f}%p")
            
            st.dataframe(
                df_res[['종목명', '현재가', '적정주가', '괴리율', 'ROE(%)', 'EPS', 'BPS', '공포지수']].style.applymap(
                    lambda x: 'color: red; font-weight: bold;' if x > 20 else ('color: blue;' if x < 0 else 'color: black;'), 
                    subset=['괴리율']
                ).format("{:,.0f}", subset=['현재가', '적정주가', 'EPS', 'BPS']),
                height=800,
                use_container_width=True
            )
        else: st.warning("데이터가 없습니다.")
    except Exception as e: st.error(f"오류: {e}")
else: st.info("👈 [분석 시작] 버튼을 눌러주세요.")

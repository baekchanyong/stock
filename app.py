import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
import time
import requests
import re
from datetime import datetime, timedelta

# --- 설정 ---
DB_FILE = "stock_analysis_v52.csv"

st.set_page_config(page_title="V52 가치투자 분석기 (주봉심리)", page_icon="⚖️", layout="wide")

# --- 헬퍼 함수 ---
def to_float(val):
    try:
        if pd.isna(val) or val == '' or str(val).strip() == '-': return 0.0
        clean_val = str(val).replace(',', '').replace('%', '')
        return float(clean_val)
    except: return 0.0

# --- [금리] 한국은행 기준금리 ---
def get_bok_base_rate():
    url = "https://finance.naver.com/marketindex/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=2)
        response.encoding = 'cp949'
        html = response.text
        match = re.search(r'한국은행 기준금리.*?([0-9]{1}\.[0-9]{2})', html, re.DOTALL)
        if match: return float(match.group(1))
        return 3.25 
    except: return 3.25

# --- 펀더멘털 정밀 크롤링 ---
def get_fundamentals(code):
    try:
        target_code = code
        if len(code) == 6 and code.isdigit() and not code.endswith('0'):
            target_code = code[:-1] + '0'
        
        url = f"https://finance.naver.com/item/main.naver?code={target_code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=2)
        
        html = response.text
        dfs = pd.read_html(html, encoding='cp949')
        
        eps, bps = 0.0, 0.0
        for df in dfs:
            if 'EPS' in df.to_string() or 'BPS' in df.to_string():
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                for idx, row in df.iterrows():
                    row_str = str(row.iloc[0])
                    if 'EPS' in row_str or '주당순이익' in row_str:
                        vals = row.iloc[1:].tolist()
                        for v in reversed(vals):
                            val = to_float(v)
                            if val > 0: 
                                eps = val
                                break
                    if 'BPS' in row_str or '주당순자산' in row_str:
                        vals = row.iloc[1:].tolist()
                        for v in reversed(vals):
                            val = to_float(v)
                            if val > 0: 
                                bps = val
                                break
                if eps > 0 and bps > 0: break
        return eps, bps
    except: return 0, 0

# --- [핵심 수정] 공포탐욕지수 (주봉 변환) ---
def calculate_fear_greed_weekly(df_daily):
    """
    일봉 데이터를 받아 주봉(Weekly)으로 변환한 뒤 공포지수를 산출합니다.
    """
    if df_daily.empty: return 50
    
    # 1. 주봉으로 리샘플링 (금요일 기준)
    try:
        df_weekly = df_daily.resample('W-FRI').agg({
            'Close': 'last'
        }).dropna()
    except:
        return 50

    # 데이터가 너무 적으면(20주 미만) 계산 불가
    if len(df_weekly) < 20: return 50
    
    # 2. 지표 계산 (주봉 기준 14주, 20주)
    delta = df_weekly['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    ma20 = df_weekly['Close'].rolling(window=20).mean()
    disparity = (df_weekly['Close'] / ma20) * 100
    
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

# --- 분석 실행 ---
def run_analysis_core(target_stocks, applied_rate, status_text, progress_bar):
    today_str = datetime.now().strftime('%Y-%m-%d')
    # 주봉 계산을 위해 넉넉하게 2년치 데이터 확보
    chart_start = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    
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
            
            # 펀더멘털
            eps, bps = get_fundamentals(code)
            if eps == 0: eps = to_float(row.get('EPS', 0))
            if bps == 0: bps = to_float(row.get('BPS', 0))
            
            roe = 0
            if bps > 0: roe = (eps / bps) * 100
            
            time.sleep(0.02) # 서버 부하 방지
            fg_score = 50
            try:
                df_chart = fdr.DataReader(code, chart_start, today_str)
                if not df_chart.empty:
                    # [수정] 주봉 기준 공포지수 사용
                    fg_score = calculate_fear_greed_weekly(df_chart)
            except: pass

            # V51 로직 유지: 수익가치(7) : 자산가치(3)
            earnings_value = 0
            if applied_rate > 0:
                earnings_value = eps / (applied_rate / 100)
            
            asset_value = bps
            base_fair_price = (earnings_value * 0.7) + (asset_value * 0.3)
            
            sentiment_factor = 1 + ((50 - fg_score) / 50 * 0.1)
            fair_price = base_fair_price * sentiment_factor
            
            gap = 0
            if current_price > 0:
                gap = (fair_price - current_price) / current_price * 100
            
            data_row = {
                '종목코드': code,
                '종목명': name,
                '현재가': round(current_price, 0),
                '적정가': round(fair_price, 0),
                '괴리율': round(gap, 2),
                '공포지수': round(fg_score, 1),
                'ROE(%)': round(roe, 2),
                'EPS': round(eps, 0),
                'BPS': round(bps, 0)
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

st.title("⚖️ V52 수익중심 가치투자 분석기 (주봉심리)")

# 설명서
with st.expander("📘 **[필독] 적정주가 & 공포지수 산출 공식**", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🧮 적정주가 산출식 (수익 7 : 자산 3)")
        st.latex(r"\text{수익가치} = \frac{\text{EPS}}{\text{한국은행 기준금리}}")
        st.latex(r"\text{적정가} = [(\text{수익가치} \times 0.7) + (\text{BPS} \times 0.3)] \times \text{심리보정}")
    
    with c2:
        st.markdown("##### 👻 공포탐욕지수 산출식 (주봉 기준)")
        st.latex(r"\text{Index} = (\text{RSI}_{14주} \times 0.5) + (\text{이격도}_{20주} \text{ 점수} \times 0.5)")
        st.caption("* **주봉(Weekly)**을 사용하여 일시적 등락을 배제하고 큰 추세를 봅니다.")

st.divider()

# --- 1. 설정 ---
st.header("1. 분석 설정")

mode = st.radio("분석 모드", ["🏆 시가총액 상위", "🔍 종목 검색"], horizontal=True)
target_stocks = pd.DataFrame()

if mode == "🏆 시가총액 상위":
    if 'stock_count' not in st.session_state:
        st.session_state.stock_count = 200

    def update_from_slider():
        st.session_state.stock_count = st.session_state.slider_key

    def apply_manual_input():
        st.session_state.stock_count = st.session_state.num_key

    c1, c2 = st.columns([3, 1])
    with c1:
        st.slider(
            "종목 수", 10, 500, 
            key='slider_key', 
            value=st.session_state.stock_count, 
            on_change=update_from_slider
        )
    with c2:
        st.number_input(
            "직접 입력", 10, 500, 
            key='num_key', 
            value=st.session_state.stock_count
        )
        if st.button("✅ 수치 적용", on_click=apply_manual_input):
            st.rerun()

elif mode == "🔍 종목 검색":
    query = st.text_input("종목명 검색", placeholder="예: 삼성")
    if query:
        try:
            with st.spinner("검색 중..."):
                df_krx = fdr.StockListing('KRX')
                res = df_krx[df_krx['Name'].str.contains(query, case=False)]
                if res.empty: st.error("결과 없음")
                else:
                    picks = st.multiselect("선택", res['Name'].tolist(), default=res['Name'].tolist()[:5])
                    target_stocks = res[res['Name'].isin(picks)]
        except: st.error("오류")

# --- 2. 실행 ---
st.divider()
if st.button("▶️ 분석 시작 (Start)", type="primary", use_container_width=True):
    
    if mode == "🏆 시가총액 상위":
        with st.spinner("리스트 로딩 중..."):
            df_krx = fdr.StockListing('KRX')
            df_krx = df_krx[df_krx['Market'].isin(['KOSPI'])]
            final_target = df_krx.sort_values(by='Marcap', ascending=False).head(st.session_state.stock_count)
    else:
        if target_stocks.empty:
            st.warning("종목을 선택해주세요.")
            st.stop()
        final_target = target_stocks

    status_box = st.empty()
    status_box.info("🇰🇷 한국은행 기준금리 조회 중...")
    
    bok_rate = get_bok_base_rate()
    applied_rate = bok_rate if bok_rate else 3.25
    
    status_box.success(f"✅ 기준금리 **{applied_rate}%** 적용 | 주봉 심리 데이터 분석 중...")
    time.sleep(0.5)
    
    p_bar = st.progress(0)
    run_analysis_core(final_target, applied_rate, status_box, p_bar)
    
    status_box.success(f"✅ 분석 완료!")

# --- 3. 결과 ---
st.divider()
st.header("🏆 분석 결과")

sort_opt = st.radio("정렬 기준", ["괴리율 높은 순", "ROE 높은 순", "공포지수 낮은 순"], horizontal=True)

if st.button("🔄 결과 새로고침"): st.rerun()

if os.path.exists(DB_FILE):
    try:
        df = pd.read_csv(DB_FILE)
        for c in ['현재가', '적정가', '괴리율', 'EPS', 'BPS', 'ROE(%)', '공포지수']:
            if c in df.columns: df[c] = df[c].apply(to_float)
            
        df = df.drop_duplicates(['종목코드'], keep='last')
        df = df[df['적정가'] > 0]
        
        if not df.empty:
            # 정렬
            if "괴리율" in sort_opt: df = df.sort_values(by='괴리율', ascending=False)
            elif "ROE" in sort_opt: df = df.sort_values(by='ROE(%)', ascending=False)
            else: df = df.sort_values(by='공포지수', ascending=True)
            
            df = df.reset_index(drop=True)
            df.index += 1
            
            # UI 고정 및 컬럼 순서
            df.index.name = "순위"
            df_display = df.set_index('종목명', append=True)
            cols = ['현재가', '적정가', '괴리율', '공포지수', 'ROE(%)', 'EPS', 'BPS']
            
            top = df.iloc[0]
            st.info(f"🥇 **1위: {top['종목명']}** | 괴리율: {top['괴리율']}% | ROE: {top['ROE(%)']}%")

            st.dataframe(
                df_display[cols].style.applymap(
                    lambda x: 'color: red; font-weight: bold;' if x > 20 else ('color: blue;' if x < 0 else 'color: black;'), 
                    subset=['괴리율']
                ).format("{:,.0f}", subset=['현재가', '적정가', 'EPS', 'BPS']),
                height=800,
                use_container_width=True
            )
        else: st.warning("결과 데이터가 없습니다.")
    except: st.error("파일 오류")
else: st.info("👈 위에서 [분석 시작] 버튼을 눌러주세요.")

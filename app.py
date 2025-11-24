import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta

# --- 설정 ---
DB_FILE = "stock_analysis_v35.csv"

st.set_page_config(page_title="V35 가치투자 분석기", page_icon="🎯", layout="wide")

# --- 헬퍼 함수 ---
def to_float(val):
    try:
        if pd.isna(val) or val == '' or str(val).strip() == '-': return 0.0
        return float(str(val).replace(',', '').replace('%', ''))
    except: return 0.0

# --- 공포탐욕지수 ---
def calculate_fear_greed_from_slice(df_slice):
    if len(df_slice) < 10: return 50
    delta = df_slice['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    ma20 = df_slice['Close'].rolling(window=20).mean()
    disparity = (df_slice['Close'] / ma20) * 100
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

# --- 분석 엔진 ---
def run_custom_analysis(target_date, period_years, target_num, status_text, progress_bar):
    quarter_count = period_years * 4
    dates = []
    for i in range(quarter_count): 
        d = target_date - timedelta(days=91 * i)
        dates.append(d.strftime('%Y-%m-%d'))
    
    target_str = dates[0]
    today_str = datetime.now().strftime('%Y-%m-%d')
    is_backtest = (target_str != today_str)

    status_text.info(f"📅 기준일 [{target_str}]로부터 과거 {period_years}년({quarter_count}분기) 데이터를 분석합니다...")

    try:
        df_main = fdr.StockListing('KRX', target_str)
        df_main = df_main[df_main['Market'].isin(['KOSPI'])]
        df_main = df_main.sort_values(by='Marcap', ascending=False)
        target_stocks = df_main.head(target_num)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return

    current_prices_map = {}
    if is_backtest:
        try:
            df_now = fdr.StockListing('KRX')
            current_prices_map = df_now.set_index('Code')['Close'].to_dict()
        except: pass

    if os.path.exists(DB_FILE): os.remove(DB_FILE)

    total = len(target_stocks)
    new_data = []
    chart_lookback_days = (period_years * 365) + 365
    chart_start_date = (datetime.strptime(dates[-1], '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')

    for step, (idx, row) in enumerate(target_stocks.iterrows()):
        code = str(row['Code'])
        name = row['Name']
        
        if name in ["맥쿼리인프라", "SK리츠"]: continue
        
        progress_bar.progress(min((step + 1) / total, 1.0))
        status_text.text(f"⏳ [{step+1}/{total}] {name} : {period_years}년치 흐름 분석 중...")
        
        try:
            time.sleep(0.01)
            df_chart_full = fdr.DataReader(code, chart_start_date, target_str)
            if df_chart_full.empty: continue

            historical_fair_prices = []
            for d in dates:
                end_dt = datetime.strptime(d, "%Y-%m-%d")
                start_dt = end_dt - timedelta(days=90)
                start_dt_str = start_dt.strftime("%Y-%m-%d")
                
                quarter_data = df_chart_full.loc[start_dt_str:d]
                if len(quarter_data) < 10: continue
                
                quarter_avg_price = quarter_data['Close'].mean()
                if quarter_avg_price <= 0: continue
                
                fg_score = calculate_fear_greed_from_slice(quarter_data)
                correction_factor = 1 + ((50 - fg_score) / 50 * 0.1)
                fair_price_at_quarter = quarter_avg_price * correction_factor
                historical_fair_prices.append(fair_price_at_quarter)

            if not historical_fair_prices: continue
            avg_fair_price = sum(historical_fair_prices) / len(historical_fair_prices)
            
            price_base = to_float(row.get('Close', 0))
            price_now = price_base
            if is_backtest and code in current_prices_map:
                price_now = to_float(current_prices_map[code])
            
            gap = 0
            if price_base > 0:
                gap = (avg_fair_price - price_base) / price_base * 100
            
            data_row = {
                '종목코드': code,
                '종목명': name,
                '기준일': target_str,
                '분석기간(년)': period_years,
                '기준일가격': round(price_base, 0),
                '현재가격': round(price_now, 0),
                '차이금액': round(price_now - price_base, 0),
                '평균적정주가': round(avg_fair_price, 0),
                '괴리율': round(gap, 2),
                '최근공포지수': round(fg_score, 1)
            }
            new_data.append(data_row)
            
            if len(new_data) >= 20:
                save_to_csv(new_data)
                new_data = []
        except: continue

    if new_data: save_to_csv(new_data)
    progress_bar.empty()
    return True

# --- 메인 UI ---

st.title("🎯 V35 맞춤형 가치투자 분석기")

with st.expander("📘 **[설명서] 기능 사용법 (Click)**", expanded=False):
    st.info("""
    1. **분석 기간 선택:** 1년~5년 중 선택 (해당 기간의 분기별 평균 주가로 적정가 산출)
    2. **주식 수 설정:** 슬라이더를 움직이거나, 숫자 입력 후 [적용]을 누르세요.
    3. **검색:** 결과 표 위에서 종목명을 입력하고 Enter를 치면 위치를 찾아줍니다.
    """)

st.divider()

# --- 1. 설정 영역 ---
st.header("1. 분석 조건 설정")

col_date, col_years = st.columns([2, 1])
with col_date:
    target_date = st.date_input("📅 분석 기준일", value=datetime.now(), min_value=datetime(2016, 1, 1), max_value=datetime.now())
with col_years:
    period_years = st.selectbox("⏳ 분석 기간 (년)", [1, 2, 3, 4, 5], index=4)

# [핵심 수정] 주식 수 입력 동기화 로직
st.write("📊 **분석할 종목 수 설정**")

if 'target_count' not in st.session_state:
    st.session_state.target_count = 200

# 슬라이더 콜백: 슬라이더 움직이면 값 업데이트
def update_from_slider():
    st.session_state.target_count = st.session_state.slider_widget

# 1. 슬라이더 (키: slider_widget)
st.slider(
    "슬라이더로 조절", 10, 300, 
    key='slider_widget', 
    value=st.session_state.target_count, 
    on_change=update_from_slider
)

# 2. 숫자 입력 + 버튼
c_input, c_btn = st.columns([3, 1])
with c_input:
    manual_val = st.number_input("직접 입력 (숫자)", 10, 500, value=st.session_state.target_count)
with c_btn:
    if st.button("✅ 수치 적용"):
        st.session_state.target_count = manual_val
        # [중요] 슬라이더의 내부 상태도 강제로 업데이트!
        st.session_state.slider_widget = manual_val
        st.rerun()

# 분석 시작 버튼
st.markdown("---")
if st.button("▶️ 분석 시작 (Start)", type="primary", use_container_width=True):
    status_box = st.empty()
    p_bar = st.progress(0)
    is_done = run_custom_analysis(target_date, period_years, st.session_state.target_count, status_box, p_bar)
    if is_done:
        status_box.success(f"✅ 분석 완료! ({period_years}년치 데이터 반영)")

st.divider()

# --- 2. 결과 영역 ---
st.header("🏆 분석 결과 리포트")

col_sort, col_search = st.columns([2, 1])

with col_sort:
    sort_option = st.radio(
        "🔀 정렬 기준", 
        ["괴리율 높은 순", "📈 가격 상승액 순", "📉 가격 하락액 순"],
        horizontal=True
    )

with col_search:
    search_term = st.text_input("🔍 종목 검색 (Enter)", placeholder="종목명 입력")

if st.button("🔄 결과 표 새로고침"): st.rerun()

if os.path.exists(DB_FILE):
    try:
        df_res = pd.read_csv(DB_FILE)
        for col in ['기준일가격', '현재가격', '차이금액', '평균적정주가', '괴리율', '최근공포지수']:
            if col in df_res.columns: df_res[col] = df_res[col].apply(to_float)

        df_res = df_res.drop_duplicates(['종목코드'], keep='last')
        df_res = df_res[df_res['평균적정주가'] > 0]
        
        if not df_res.empty:
            if "괴리율" in sort_option:
                df_res = df_res.sort_values(by='괴리율', ascending=False)
            elif "상승액" in sort_option:
                df_res = df_res.sort_values(by='차이금액', ascending=False)
            elif "하락액" in sort_option:
                df_res = df_res.sort_values(by='차이금액', ascending=True)

            df_res = df_res.reset_index(drop=True)
            df_res.index += 1
            df_res.index.name = "순번"
            
            if search_term:
                matches = df_res[df_res['종목명'].str.contains(search_term, na=False)]
                if not matches.empty:
                    match_row = matches.iloc[0]
                    st.success(f"🔎 **'{match_row['종목명']}'** 찾음! 현재 **{match_row.name}위**")
                else:
                    st.error("❌ 해당 종목을 찾을 수 없습니다.")

            def highlight_search(row):
                styles = [''] * len(row)
                if search_term and search_term in str(row['종목명']):
                    return ['background-color: #ffffcc; color: black; font-weight: bold; border: 2px solid orange;'] * len(row)
                if row.name == '괴리율':
                    val = row['괴리율']
                    if val > 20: return 'color: red; font-weight: bold;'
                    elif val < 0: return 'color: blue;'
                return styles

            st.dataframe(
                df_res[['기준일', '종목명', '기준일가격', '현재가격', '차이금액', '평균적정주가', '괴리율', '최근공포지수']].style.apply(
                    highlight_search, axis=1
                ).applymap(
                    lambda x: 'color: red; font-weight: bold;' if x > 0 else 'color: blue; font-weight: bold;',
                    subset=['차이금액']
                ).format("{:,.0f}", subset=['기준일가격', '현재가격', '차이금액', '평균적정주가']),
                height=800,
                use_container_width=True
            )
        else: st.warning("데이터가 없습니다.")
    except Exception as e: st.error(f"오류: {e}")
else: st.info("👈 [분석 시작] 버튼을 눌러주세요.")

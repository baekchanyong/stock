import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta

# --- 설정 ---
DB_FILE = "stock_analysis_v30.csv"

st.set_page_config(page_title="V30 심플 가치투자 분석기", page_icon="⚡", layout="wide")

# --- 헬퍼 함수 ---
def to_float(val):
    try:
        if pd.isna(val) or val == '' or str(val).strip() == '-': return 0.0
        return float(str(val).replace(',', '').replace('%', ''))
    except: return 0.0

# --- 공포탐욕지수 (차트 슬라이싱) ---
def calculate_fear_greed_from_slice(df_slice):
    if len(df_slice) < 20: return 50
    
    # RSI (14)
    delta = df_slice['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 이격도 (20)
    ma20 = df_slice['Close'].rolling(window=20).mean()
    disparity = (df_slice['Close'] / ma20) * 100
    disparity_score = disparity.apply(lambda x: 0 if x < 90 else (100 if x > 110 else (x - 90) * 5))
    
    try:
        last_rsi = rsi.iloc[-1]
        last_disp = disparity_score.iloc[-1]
        if pd.isna(last_rsi) or pd.isna(last_disp): return 50
        return (last_rsi * 0.5) + (last_disp * 0.5)
    except: return 50

# --- CSV 저장 ---
def save_to_csv(data):
    df = pd.DataFrame(data)
    if not os.path.exists(DB_FILE):
        df.to_csv(DB_FILE, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(DB_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

# --- 핵심 분석 엔진 ---
def run_simple_analysis(target_date, target_num, status_text, progress_bar):
    
    # 1. 5년(20분기) 날짜 생성
    dates = []
    for i in range(20): 
        d = target_date - timedelta(days=91 * i)
        dates.append(d.strftime('%Y-%m-%d'))
    
    target_str = dates[0]
    today_str = datetime.now().strftime('%Y-%m-%d')
    is_backtest = (target_str != today_str)

    status_text.info(f"⚡ 과거 5년(20분기)의 주가와 심리를 분석합니다...")

    # 2. 데이터 스냅샷 로딩
    df_krx_snapshots = {}
    try:
        df_main = fdr.StockListing('KRX', target_str)
        df_main = df_main[df_main['Market'].isin(['KOSPI'])]
        df_main = df_main.sort_values(by='Marcap', ascending=False)
        target_stocks = df_main.head(target_num)
        
        for i, d in enumerate(dates):
            if i % 5 == 0: status_text.text(f"📥 데이터 복원 중... ({d})")
            try:
                snapshot = fdr.StockListing('KRX', d)
                if not snapshot.empty:
                    df_krx_snapshots[d] = snapshot.set_index('Code')['Close']
            except: pass
            
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return

    # 현재가 로딩
    current_prices_map = {}
    if is_backtest:
        try:
            df_now = fdr.StockListing('KRX')
            current_prices_map = df_now.set_index('Code')['Close'].to_dict()
        except: pass

    if os.path.exists(DB_FILE): os.remove(DB_FILE)

    total = len(target_stocks)
    new_data = []
    
    chart_start_date = (datetime.strptime(dates[-1], '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')

    for step, (idx, row) in enumerate(target_stocks.iterrows()):
        code = str(row['Code'])
        name = row['Name']
        
        if name in ["맥쿼리인프라", "SK리츠"]: continue
        
        progress_bar.progress(min((step + 1) / total, 1.0))
        status_text.text(f"⏳ [{step+1}/{total}] {name} 분석 중...")
        
        try:
            time.sleep(0.01)
            df_chart_full = fdr.DataReader(code, chart_start_date, target_str)
            
            historical_fair_prices = []
            
            for d in dates:
                if d not in df_krx_snapshots or code not in df_krx_snapshots[d].index:
                    continue
                
                price_then = to_float(df_krx_snapshots[d][code])
                if price_then <= 0: continue
                
                fg_score = 50
                if not df_chart_full.empty:
                    chart_slice = df_chart_full.loc[:d].tail(60)
                    fg_score = calculate_fear_greed_from_slice(chart_slice)

                # 심플 적정주가 공식 (주가 * 심리보정)
                correction_factor = 1 + ((50 - fg_score) / 50 * 0.1)
                fair_price_at_moment = price_then * correction_factor
                historical_fair_prices.append(fair_price_at_moment)

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
                '기준일가격': round(price_base, 0),
                '현재가격': round(price_now, 0),
                '차이금액': round(price_now - price_base, 0),
                '5년평균적정가': round(avg_fair_price, 0),
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

# --- 메인 화면 구성 ---

st.title("⚡ V30 심플 가치투자 분석기")

# 설명 섹션
with st.expander("📘 **[NEW] 심플 적정주가 산출 원리 (Click)**", expanded=False):
    st.info("""
    **복잡한 재무제표 없이, '가격'과 '심리'의 역사적 평균으로 가치를 찾습니다.**
    
    1. **개별 시점 적정가:** $\\text{당시주가} \\times \\left( 1 + \\frac{50 - \\text{공포지수}}{50} \\times 0.1 \\right)$
       *(공포스러우면 주가보다 높게, 탐욕스러우면 주가보다 낮게 평가)*
    2. **최종 적정주가:** 위 계산을 **과거 5년(20분기)** 동안 반복하여 평균을 냅니다.
    """)

st.divider()

# [수정됨] 설정 및 실행 버튼을 메인 화면으로 이동 (사이드바 X)
st.header("1. 분석 조건 설정")

col1, col2 = st.columns(2)
with col1:
    target_date = st.date_input("📅 분석 기준일", value=datetime.now(), min_value=datetime(2016, 1, 1), max_value=datetime.now())
with col2:
    target_count = st.slider("분석 종목 수", 10, 300, 50)

# [중요] 버튼을 메인 화면에 크게 배치
if st.button("▶️ 분석 시작 (Start Analysis)", type="primary", use_container_width=True):
    status_box = st.empty()
    p_bar = st.progress(0)
    is_done = run_simple_analysis(target_date, target_count, status_box, p_bar)
    if is_done:
        status_box.success(f"✅ 분석 완료! 아래 순위를 확인하세요.")

st.divider()

# 결과 화면
st.header("🏆 5년 평균 가치투자 순위")

sort_option = st.radio(
    "🔀 정렬 기준", 
    ["괴리율 높은 순 (저평가)", "📈 가격 상승액 순 (수익)", "📉 가격 하락액 순 (손실)"],
    horizontal=True
)

if st.button("🔄 결과 표 새로고침"): st.rerun()

if os.path.exists(DB_FILE):
    try:
        df_res = pd.read_csv(DB_FILE)
        for col in ['기준일가격', '현재가격', '차이금액', '5년평균적정가', '괴리율', '최근공포지수']:
            if col in df_res.columns: df_res[col] = df_res[col].apply(to_float)

        df_res = df_res.drop_duplicates(['종목코드'], keep='last')
        df_res = df_res[df_res['5년평균적정가'] > 0]
        
        if not df_res.empty:
            # 정렬
            if "괴리율" in sort_option:
                df_res = df_res.sort_values(by='괴리율', ascending=False)
            elif "상승액" in sort_option:
                df_res = df_res.sort_values(by='차이금액', ascending=False)
            elif "하락액" in sort_option:
                df_res = df_res.sort_values(by='차이금액', ascending=True)

            df_res = df_res.reset_index(drop=True)
            df_res.index += 1
            
            # 모바일 뷰 (인덱스 고정)
            df_res.index.name = "순번"
            df_display = df_res.set_index('종목명', append=True)
            
            top = df_res.iloc[0]
            st.info(f"🥇 **1위: {top['종목명']}** | 5년평균적정가: {top['5년평균적정가']:,.0f}원 | 괴리율: {top['괴리율']}%")
            
            st.dataframe(
                df_display[['기준일', '기준일가격', '현재가격', '차이금액', '5년평균적정가', '괴리율', '최근공포지수']].style.applymap(
                    lambda x: 'color: red; font-weight: bold;' if x > 20 else ('color: blue;' if x < 0 else 'color: black;'), 
                    subset=['괴리율']
                ).applymap(
                    lambda x: 'color: red; font-weight: bold;' if x > 0 else 'color: blue; font-weight: bold;',
                    subset=['차이금액']
                ).format("{:,.0f}", subset=['기준일가격', '현재가격', '차이금액', '5년평균적정가']),
                height=800,
                use_container_width=True
            )
        else: st.warning("데이터가 없습니다. 위쪽의 [▶️ 분석 시작] 버튼을 눌러주세요.")
    except Exception as e: st.error(f"오류: {e}")
else: st.info("👈 위쪽의 **[▶️ 분석 시작]** 버튼을 눌러주세요.")

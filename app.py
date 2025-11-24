import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta

# --- 설정 ---
DB_FILE = "stock_analysis_v32.csv"

st.set_page_config(page_title="V32 정밀 가치투자 분석기", page_icon="⚖️", layout="wide")

# --- 헬퍼 함수 ---
def to_float(val):
    try:
        if pd.isna(val) or val == '' or str(val).strip() == '-': return 0.0
        return float(str(val).replace(',', '').replace('%', ''))
    except: return 0.0

# --- 공포탐욕지수 (차트 슬라이싱) ---
def calculate_fear_greed_from_slice(df_slice):
    if len(df_slice) < 10: return 50 # 데이터 너무 적으면 중립
    
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

# --- [핵심] 분기 평균 기반 분석 엔진 ---
def run_avg_price_analysis(target_date, target_num, status_text, progress_bar):
    
    # 1. 20개 분기 시점 생성
    dates = []
    for i in range(20): 
        d = target_date - timedelta(days=91 * i)
        dates.append(d.strftime('%Y-%m-%d'))
    
    target_str = dates[0]
    today_str = datetime.now().strftime('%Y-%m-%d')
    is_backtest = (target_str != today_str)

    status_text.info(f"⚖️ 과거 5년(20분기)의 [평균 주가]를 산출하여 정밀 분석 중입니다...")

    # 2. 종목 리스트 로딩
    try:
        df_main = fdr.StockListing('KRX', target_str)
        df_main = df_main[df_main['Market'].isin(['KOSPI'])]
        df_main = df_main.sort_values(by='Marcap', ascending=False)
        target_stocks = df_main.head(target_num)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return

    # 현재가 로딩 (검증용)
    current_prices_map = {}
    if is_backtest:
        try:
            df_now = fdr.StockListing('KRX')
            current_prices_map = df_now.set_index('Code')['Close'].to_dict()
        except: pass

    if os.path.exists(DB_FILE): os.remove(DB_FILE)

    total = len(target_stocks)
    new_data = []
    
    # 차트 데이터 시작일 (6년 전부터 넉넉하게)
    chart_start_date = (datetime.strptime(dates[-1], '%Y-%m-%d') - timedelta(days=400)).strftime('%Y-%m-%d')

    # --- 종목별 분석 ---
    for step, (idx, row) in enumerate(target_stocks.iterrows()):
        code = str(row['Code'])
        name = row['Name']
        
        if name in ["맥쿼리인프라", "SK리츠"]: continue
        
        progress_bar.progress(min((step + 1) / total, 1.0))
        status_text.text(f"⏳ [{step+1}/{total}] {name} : 분기 평균 주가 계산 중...")
        
        try:
            # 차트 데이터 로딩
            time.sleep(0.01)
            df_chart_full = fdr.DataReader(code, chart_start_date, target_str)
            
            if df_chart_full.empty: continue

            historical_fair_prices = []
            
            # 20개 분기 루프
            for d in dates:
                # [핵심 변경] 해당 날짜 기준 과거 3개월(약 90일) 데이터 추출
                end_dt = datetime.strptime(d, "%Y-%m-%d")
                start_dt = end_dt - timedelta(days=90) # 분기 시작일
                start_dt_str = start_dt.strftime("%Y-%m-%d")
                
                # 해당 분기의 데이터 슬라이싱
                quarter_data = df_chart_full.loc[start_dt_str:d]
                
                if len(quarter_data) < 10: continue # 거래일수 부족하면 스킵
                
                # 1. 그 분기의 [평균 주가] 계산 (Spot 가격 아님!)
                quarter_avg_price = quarter_data['Close'].mean()
                
                if quarter_avg_price <= 0: continue
                
                # 2. 공포지수 (그 분기 말 기준)
                # 공포지수는 추세를 봐야 하므로 분기 말 시점의 지표를 씁니다.
                fg_score = calculate_fear_greed_from_slice(quarter_data)

                # 3. 적정주가 계산 (평균주가 * 심리보정)
                # 주가가 평균적이더라도, 심리가 공포였다면 저평가 구간으로 해석하여 가치 상향
                correction_factor = 1 + ((50 - fg_score) / 50 * 0.1)
                
                fair_price_at_quarter = quarter_avg_price * correction_factor
                historical_fair_prices.append(fair_price_at_quarter)

            # 최종 5년 평균 적정가
            if not historical_fair_prices: continue
            avg_fair_price = sum(historical_fair_prices) / len(historical_fair_prices)
            
            # 기준일 당시의 실제 주가 (매수 기준가)
            # 주의: 적정가 계산엔 '평균'을 썼지만, 내가 사는 가격은 '그날 종가'입니다.
            price_base = to_float(row.get('Close', 0))
            
            # 현재 주가
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

# --- 메인 화면 ---

st.title("⚖️ V32 정밀 가치투자 분석기")

with st.expander("📘 **[NEW] 분기 평균 주가 적용 원리 (노이즈 제거)**", expanded=False):
    st.info("""
    **기존 방식과의 차이점**
    * **기존:** 3월 31일 딱 하루의 종가만 사용 → 그날 급등락하면 오차 발생
    * **V32:** 1월 1일 ~ 3월 31일의 **[평균 주가]** 사용 → 일시적 거품이나 폭락 노이즈 제거
    
    **산출 공식**
    $$ \text{분기별 적정가} = \text{해당분기 평균주가} \times \text{심리보정계수} $$
    *(이 값을 과거 20분기 동안 계산하여 평균)*
    """)

st.divider()

# 설정 UI
st.header("1. 분석 조건 설정")

col1, col2 = st.columns(2)
with col1:
    target_date = st.date_input("📅 분석 기준일", value=datetime.now(), min_value=datetime(2016, 1, 1), max_value=datetime.now())
with col2:
    target_count = st.slider("분석 종목 수", 10, 300, 50)

if st.button("▶️ 정밀 분석 시작 (Start)", type="primary", use_container_width=True):
    status_box = st.empty()
    p_bar = st.progress(0)
    is_done = run_avg_price_analysis(target_date, target_count, status_box, p_bar)
    if is_done:
        status_box.success(f"✅ 분석 완료! 분기 평균 가격으로 정밀하게 계산되었습니다.")

st.divider()

# 결과 UI
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
            if "괴리율" in sort_option:
                df_res = df_res.sort_values(by='괴리율', ascending=False)
            elif "상승액" in sort_option:
                df_res = df_res.sort_values(by='차이금액', ascending=False)
            elif "하락액" in sort_option:
                df_res = df_res.sort_values(by='차이금액', ascending=True)

            df_res = df_res.reset_index(drop=True)
            df_res.index += 1
            
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

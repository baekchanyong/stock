import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
import time
import requests
import re
import yfinance as yf
from datetime import datetime, timedelta

# --- 설정 ---
DB_FILE = "stock_analysis_v68.csv"

st.set_page_config(page_title="V68 가치투자 분석기 (주봉적용)", page_icon="⚖️", layout="wide")

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

# --- [과거 금리 추정] ---
def get_historical_base_rate(date_str):
    return 3.50

# --- [데이터 수집] 휴일 보정 ---
def get_stock_listing_with_retry(market, date_str, max_retries=5):
    curr_date = datetime.strptime(date_str, "%Y-%m-%d")
    for _ in range(max_retries):
        d_str = curr_date.strftime("%Y-%m-%d")
        try:
            df = fdr.StockListing(market, d_str)
            if not df.empty:
                return df
        except: pass
        curr_date -= timedelta(days=1)
    return pd.DataFrame()

# --- 3중 데이터 확보 전략 ---
def get_robust_metrics(code, row):
    current_price = to_float(row.get('Close', 0))
    eps = to_float(row.get('EPS', 0))
    bps = to_float(row.get('BPS', 0))
    
    if eps == 0 or bps == 0:
        try:
            ticker = yf.Ticker(f"{code}.KS")
            info = ticker.info
            if eps == 0 and info.get('trailingEps'): eps = float(info['trailingEps'])
            if bps == 0 and info.get('bookValue'): bps = float(info['bookValue'])
        except: pass
    
    if current_price > 0:
        per = to_float(row.get('PER', 0))
        pbr = to_float(row.get('PBR', 0))
        if eps == 0 and per > 0: eps = current_price / per
        if bps == 0 and pbr > 0: bps = current_price / pbr
        
    return eps, bps

# --- [핵심 수정] 공포탐욕지수 (주봉 변환 적용) ---
def calculate_fear_greed_weekly(df_daily):
    """
    일봉 데이터를 받아 주봉(Weekly)으로 변환한 뒤 공포지수를 산출합니다.
    """
    if df_daily.empty: return 50
    
    # 1. 주봉으로 리샘플링 (금요일 기준)
    # Open은 첫날, High는 최대, Low는 최소, Close는 마지막 날
    try:
        df_weekly = df_daily.resample('W-FRI').agg({
            'Close': 'last'
        }).dropna()
    except:
        return 50

    # 데이터가 너무 적으면(20주 미만) 계산 불가 -> 50점
    if len(df_weekly) < 20: return 50
    
    # 2. 지표 계산 (주봉 기준)
    # RSI (14주)
    delta = df_weekly['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 이격도 (20주 이동평균선)
    ma20 = df_weekly['Close'].rolling(window=20).mean()
    disparity = (df_weekly['Close'] / ma20) * 100
    
    disparity_score = disparity.apply(lambda x: 0 if x < 90 else (100 if x > 110 else (x - 90) * 5))
    
    try:
        # 가장 최근 주봉의 값 사용
        val = (rsi.iloc[-1] * 0.5) + (disparity_score.iloc[-1] * 0.5)
        return 50 if pd.isna(val) else val
    except: return 50

# --- CSV 저장 ---
def save_to_csv_flat(data_list):
    if not data_list: return
    df = pd.DataFrame(data_list)
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            new_cols.append(f"{col[0]}_{col[1]}")
        else:
            new_cols.append(str(col))
    df.columns = new_cols
    
    if not os.path.exists(DB_FILE):
        df.to_csv(DB_FILE, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(DB_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

# --- 분석 실행 ---
def run_history_analysis(target_stocks, applied_rate, status_text, progress_bar):
    today = datetime.now()
    quarters = []
    temp_date = today
    for _ in range(8):
        temp_date = temp_date - timedelta(days=95)
        q_date_str = temp_date.strftime('%Y-%m-%d')
        quarters.append(q_date_str)
    
    status_text.info(f"📅 과거 2년(8개 분기) 데이터를 복원 중입니다...")

    snapshot_dfs = {}
    try:
        for i, q_date in enumerate(quarters):
            status_text.text(f"📥 [{i+1}/8] {q_date} 기준 데이터 확보 중...")
            df = get_stock_listing_with_retry('KRX', q_date)
            if not df.empty:
                snapshot_dfs[q_date] = df.set_index('Code')
    except Exception as e:
        st.error(f"데이터 준비 실패: {e}")
        return

    if os.path.exists(DB_FILE): os.remove(DB_FILE)

    total = len(target_stocks)
    new_data = []
    
    # 주봉 생성을 위해 넉넉하게 과거 2년치 데이터를 더 가져옴 (총 4.5년)
    chart_start = (today - timedelta(days=365*4.5)).strftime('%Y-%m-%d')
    today_str = today.strftime('%Y-%m-%d')

    for step, (idx, row) in enumerate(target_stocks.iterrows()):
        code = str(row['Code'])
        name = row['Name']
        
        if name in ["맥쿼리인프라", "SK리츠"]: continue
        
        progress_bar.progress(min((step + 1) / total, 1.0))
        status_text.text(f"⏳ [{step+1}/{total}] {name} 주봉 분석 중...")
        
        try:
            current_price = to_float(row.get('Close', 0))
            eps_now, bps_now = get_robust_metrics(code, row)
            
            time.sleep(0.02)
            df_chart_full = fdr.DataReader(code, chart_start, today_str)
            
            # [수정] 주봉 기준 공포지수 산출
            fg_score_now = 50
            if not df_chart_full.empty:
                fg_score_now = calculate_fear_greed_weekly(df_chart_full)
            
            base_rate = applied_rate
            earnings_val = eps_now / (base_rate/100) if base_rate > 0 else 0
            base_fair = (earnings_val * 0.7) + (bps_now * 0.3)
            sentiment = 1 + ((50 - fg_score_now)/50 * 0.1)
            fair_now = base_fair * sentiment
            
            gap_now = 0
            if current_price > 0:
                gap_now = (fair_now - current_price) / current_price * 100
            
            data_row = {
                '기본정보_종목코드': code,
                '기본정보_종목명': name,
                '현재정보_현재가': round(current_price, 0),
                '현재정보_적정주가': round(fair_now, 0),
                '현재정보_괴리율': round(gap_now, 2),
                '지표_공포지수': round(fg_score_now, 1),
                '지표_EPS': round(eps_now, 0),
                '지표_BPS': round(bps_now, 0)
            }
            
            for q_date in quarters:
                q_end_dt = datetime.strptime(q_date, '%Y-%m-%d')
                q_start_dt = q_end_dt - timedelta(days=90)
                q_start_str = q_start_dt.strftime('%Y-%m-%d')
                
                yyyy = q_end_dt.year
                mm = q_end_dt.month
                q_num = (mm - 1) // 3 + 1
                if q_num == 0: q_num = 4; yyyy -= 1
                col_group = f"{str(yyyy)[2:]}년{q_num}Q"
                
                q_avg_price = 0
                q_fair = 0
                
                if not df_chart_full.empty:
                    # 해당 시점까지의 데이터 슬라이싱
                    q_chart = df_chart_full.loc[:q_date]
                    if not q_chart.empty:
                        # 분기 평균 주가 (일봉 평균) - 이건 그대로 둠 (가격 확인용)
                        # 분기 내 가격 흐름은 일봉으로 보는 게 맞음
                        q_slice_for_price = q_chart.loc[q_start_str:q_date]
                        if not q_slice_for_price.empty:
                            q_avg_price = q_slice_for_price['Close'].mean()
                        
                        # 스냅샷 데이터
                        found_snap = None
                        for snap_date in snapshot_dfs.keys():
                            diff = abs((datetime.strptime(snap_date, '%Y-%m-%d') - q_end_dt).days)
                            if diff < 10:
                                found_snap = snapshot_dfs[snap_date]
                                break
                        
                        if found_snap is not None and code in found_snap.index:
                            snap_row = found_snap.loc[code]
                            q_eps, q_bps = get_robust_metrics(code, snap_row)
                            
                            # [수정] 과거 시점의 공포지수도 '주봉'으로 계산
                            q_fg = calculate_fear_greed_weekly(q_chart)
                            q_rate = get_historical_base_rate(q_date)
                            
                            q_earn = q_eps / (q_rate/100)
                            q_base = (q_earn * 0.7) + (q_bps * 0.3)
                            q_sent = 1 + ((50 - q_fg)/50 * 0.1)
                            q_fair = q_base * q_sent
                
                data_row[f"{col_group}_평균주가"] = round(q_avg_price, 0)
                data_row[f"{col_group}_적정주가"] = round(q_fair, 0)

            new_data.append(data_row)
            
            if len(new_data) >= 5:
                save_to_csv_flat(new_data)
                new_data = []
        except: continue

    if new_data:
        save_to_csv_flat(new_data)
            
    progress_bar.empty()
    return True

# --- 메인 UI ---

st.title("⚖️ V68 가치투자 분석기 (주봉 심리)")

with st.expander("📘 **[필독] 적정주가 & 공포지수 산출 공식**", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🧮 적정주가 (수익 7 : 자산 3)")
        st.latex(r"\text{적정가} = \left[ \left( \frac{\text{EPS}}{\text{금리}} \times 0.7 \right) + \left( \text{BPS} \times 0.3 \right) \right] \times \text{심리보정}")
        st.caption("* 금리: 한국은행 기준금리 (약 3.25%)")
    with c2:
        st.markdown("##### 👻 공포탐욕지수 (주봉 기준)")
        st.latex(r"\text{Index} = (\text{RSI}_{14주} \times 0.5) + (\text{이격도}_{20주} \text{ 점수} \times 0.5)")
        st.caption("* **주봉(Weekly)** 데이터를 사용하여 중기 추세를 반영합니다.")

st.divider()

# --- 1. 설정 ---
st.header("1. 분석 설정")

mode = st.radio("분석 모드", ["🏆 시가총액 상위", "🔍 종목 검색"], horizontal=True)
target_stocks = pd.DataFrame()

if mode == "🏆 시가총액 상위":
    if 'stock_count' not in st.session_state:
        st.session_state.stock_count = 50

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
        with st.spinner("리스트 로딩..."):
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
    
    status_box.success(f"✅ 기준금리 **{applied_rate}%** 적용 | 데이터 정밀 분석 시작...")
    time.sleep(0.5)
    
    p_bar = st.progress(0)
    run_history_analysis(final_target, applied_rate, status_box, p_bar)
    
    status_box.success(f"✅ 분석 완료!")

# --- 3. 결과 ---
st.divider()
st.header("🏆 히스토리칼 분석 결과")

sort_opt = st.radio("정렬 기준", ["괴리율 높은 순", "ROE 높은 순", "공포지수 낮은 순"], horizontal=True)

if st.button("🔄 결과 새로고침"): st.rerun()

if os.path.exists(DB_FILE):
    try:
        df = pd.read_csv(DB_FILE)
        
        numeric_targets = ['현재가', '적정주가', '괴리율', 'EPS', 'BPS', 'ROE', '공포지수', '평균주가', '적정가']
        for col in df.columns:
            if any(t in col for t in numeric_targets):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        if '기본정보_종목코드' in df.columns:
            df = df.drop_duplicates(['기본정보_종목코드'], keep='last')
        elif '종목코드' in df.columns:
             df = df.drop_duplicates(['종목코드'], keep='last')
        
        # 정렬
        sort_col = '현재정보_괴리율'
        ascending = False
        if "ROE" in sort_opt: sort_col = '지표_ROE(%)'
        elif "공포" in sort_opt: 
            sort_col = '지표_공포지수'
            ascending = True
        
        if sort_col in df.columns:
            df = df.sort_values(by=sort_col, ascending=ascending)
        
        df = df.reset_index(drop=True)
        df.index += 1
        df.index.name = "순위"

        # MultiIndex 변환
        if '기본정보_종목명' in df.columns:
            df_display = df.set_index('기본정보_종목명', append=True)
        else:
            df_display = df

        new_cols = []
        for col in df_display.columns:
            if "_" in col:
                parts = col.split("_", 1)
                new_cols.append((parts[0], parts[1]))
            else:
                new_cols.append(("기타", col))
        
        df_display.columns = pd.MultiIndex.from_tuples(new_cols)
        
        # 컬럼 순서
        display_cols = [
            ('현재정보', '현재가'), ('현재정보', '적정주가'), ('현재정보', '괴리율'),
            ('지표', '공포지수'), ('지표', 'ROE(%)'), ('지표', 'EPS'), ('지표', 'BPS')
        ]
        
        levels = df_display.columns.levels[0]
        hist_groups = [l for l in levels if '년' in l and 'Q' in l]
        hist_groups.sort(reverse=True)
        
        for q in hist_groups:
            display_cols.append((q, '평균주가'))
            display_cols.append((q, '적정주가'))
            
        final_cols = [c for c in display_cols if c in df_display.columns]
        
        if not df_display.empty:
            try:
                top_row = df.iloc[0]
                t_name = top_row.name[1] if isinstance(top_row.name, tuple) else top_row.name
                t_gap = top_row.get(('현재정보', '괴리율'), 0)
                st.info(f"🥇 **1위: {t_name}** | 현재 괴리율: {t_gap}%")
            except: pass

        st.dataframe(
            df_display[final_cols].style.applymap(
                lambda x: 'color: red; font-weight: bold;' if x > 20 else ('color: blue;' if x < 0 else 'color: black;'), 
                subset=[('현재정보', '괴리율')] if ('현재정보', '괴리율') in df_display.columns else []
            ).format("{:,.0f}", na_rep="-"),
            height=800,
            use_container_width=True
        )
        
    except Exception as e: st.error(f"표시 오류: {e}")
else: st.info("👈 위에서 [분석 시작] 버튼을 눌러주세요.")

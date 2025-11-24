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
DB_FILE = "stock_analysis_v54.csv"

st.set_page_config(page_title="V54 가치투자 분석기 (오류해결)", page_icon="💎", layout="wide")

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
    return 3.50 # 최근 2년 평균

# --- 펀더멘털 정밀 크롤링 (현재용) ---
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

# --- [수정됨] 분석 실행 함수 ---
# target_stocks는 이미 DataFrame이므로, 내부에서 다시 head()나 Listing()을 호출하지 않습니다.
def run_history_analysis(target_stocks, applied_rate, status_text, progress_bar):
    
    today = datetime.now()
    
    # 1. 과거 8개 분기 날짜 계산
    quarters = []
    temp_date = today
    for _ in range(8):
        temp_date = temp_date - timedelta(days=95)
        q_date_str = temp_date.strftime('%Y-%m-%d')
        quarters.append(q_date_str)
    
    status_text.info(f"📅 과거 2년(8개 분기)의 EPS/BPS 데이터를 복원 중입니다... (잠시만 기다려주세요)")

    # 2. 과거 데이터 스냅샷 로딩 (속도 최적화)
    # 이 부분은 '종목 리스트'와 무관하게 과거 '시장 데이터'를 가져오는 것이라 유지
    snapshot_dfs = {}
    try:
        for i, q_date in enumerate(quarters):
            status_text.text(f"📥 [{i+1}/8] 과거({q_date}) 재무 데이터 로딩 중...")
            try:
                df = fdr.StockListing('KRX', q_date)
                if not df.empty:
                    snapshot_dfs[q_date] = df.set_index('Code')
            except: pass
            
    except Exception as e:
        st.error(f"데이터 준비 실패: {e}")
        return

    if os.path.exists(DB_FILE): os.remove(DB_FILE)

    total = len(target_stocks)
    new_data = []
    
    # 차트 데이터 시작일
    chart_start = (today - timedelta(days=365*2.5)).strftime('%Y-%m-%d')
    today_str = today.strftime('%Y-%m-%d')

    # --- 종목별 루프 ---
    # target_stocks는 이미 선택된 50개(또는 검색된 종목)의 DataFrame입니다.
    for step, (idx, row) in enumerate(target_stocks.iterrows()):
        code = str(row['Code'])
        name = row['Name']
        
        if name in ["맥쿼리인프라", "SK리츠"]: continue
        
        progress_bar.progress(min((step + 1) / total, 1.0))
        status_text.text(f"⏳ [{step+1}/{total}] {name} : 2년치 히스토리 분석 중...")
        
        try:
            # 1. 현재 시점 분석 (Live)
            current_price = to_float(row.get('Close', 0))
            
            # 정밀 크롤링
            eps_now, bps_now = get_fundamentals(code)
            if eps_now == 0: eps_now = to_float(row.get('EPS', 0))
            if bps_now == 0: bps_now = to_float(row.get('BPS', 0))
            
            time.sleep(0.02)
            df_chart_full = fdr.DataReader(code, chart_start, today_str)
            
            fg_score_now = 50
            if not df_chart_full.empty:
                fg_score_now = calculate_fear_greed_from_slice(df_chart_full.tail(60))
            
            # 현재 적정가 (하이브리드 모델)
            base_rate = applied_rate
            earnings_val = eps_now / (base_rate/100) if base_rate > 0 else 0
            base_fair = (earnings_val * 0.7) + (bps_now * 0.3)
            sentiment = 1 + ((50 - fg_score_now)/50 * 0.1)
            fair_now = base_fair * sentiment
            
            gap_now = 0
            if current_price > 0:
                gap_now = (fair_now - current_price) / current_price * 100
                
            data_row = {
                '종목코드': code,
                '종목명': name,
                '현재가': round(current_price, 0),
                '적정가': round(fair_now, 0),
                '괴리율': round(gap_now, 2),
                '공포지수': round(fg_score_now, 1),
                'ROE(%)': round((eps_now/bps_now)*100, 2) if bps_now > 0 else 0,
                'EPS': round(eps_now, 0),
                'BPS': round(bps_now, 0)
            }
            
            # 2. 과거 8개 분기 분석 (History)
            for i, q_date in enumerate(quarters):
                q_end_dt = datetime.strptime(q_date, '%Y-%m-%d')
                q_start_dt = q_end_dt - timedelta(days=90)
                q_start_str = q_start_dt.strftime('%Y-%m-%d')
                
                q_avg_price = 0
                q_fair = 0
                
                if not df_chart_full.empty:
                    q_chart = df_chart_full.loc[q_start_str:q_date]
                    if not q_chart.empty:
                        q_avg_price = q_chart['Close'].mean()
                        
                        # 당시 데이터
                        if q_date in snapshot_dfs and code in snapshot_dfs[q_date].index:
                            snap_row = snapshot_dfs[q_date].loc[code]
                            q_eps = to_float(snap_row.get('EPS', 0))
                            q_bps = to_float(snap_row.get('BPS', 0))
                            
                            # 역산 보정
                            q_price_close = to_float(snap_row.get('Close', 0))
                            if q_eps == 0 and q_price_close > 0:
                                q_per = to_float(snap_row.get('PER', 0))
                                if q_per > 0: q_eps = q_price_close / q_per
                            if q_bps == 0 and q_price_close > 0:
                                q_pbr = to_float(snap_row.get('PBR', 0))
                                if q_pbr > 0: q_bps = q_price_close / q_pbr
                            
                            q_fg = calculate_fear_greed_from_slice(q_chart)
                            q_rate = get_historical_base_rate(q_date)
                            
                            q_earn_val = q_eps / (q_rate/100)
                            q_base_fair = (q_earn_val * 0.7) + (q_bps * 0.3)
                            q_sent = 1 + ((50 - q_fg)/50 * 0.1)
                            q_fair = q_base_fair * q_sent
                
                yyyy = q_end_dt.year
                mm = q_end_dt.month
                q_num = (mm - 1) // 3 + 1
                if q_num == 0: q_num = 4; yyyy -= 1
                col_prefix = f"{str(yyyy)[2:]}년{q_num}Q"
                
                data_row[f"{col_prefix}_평균주가"] = round(q_avg_price, 0)
                data_row[f"{col_prefix}_적정주가"] = round(q_fair, 0)

            new_data.append(data_row)
            
            if len(new_data) >= 10:
                save_to_csv(new_data)
                new_data = []
        except: continue

    if new_data: save_to_csv(new_data)
    progress_bar.empty()
    return True

# --- 메인 UI ---

st.title("📚 V54 히스토리칼 밸류에이션 (완결)")

with st.expander("📘 **[분석 원리] 과거 8분기 추적**", expanded=True):
    st.info("💡 과거 2년 동안의 적정주가 변화와 평균주가 흐름을 한눈에 비교합니다.")

st.divider()

# --- 1. 설정 ---
st.header("1. 분석 설정")

mode = st.radio("분석 모드", ["🏆 시가총액 상위", "🔍 종목 검색"], horizontal=True)
target_stocks = pd.DataFrame()

if mode == "🏆 시가총액 상위":
    if 'stock_count' not in st.session_state: st.session_state.stock_count = 50
    
    def update_slider(): st.session_state.stock_count = st.session_state.slider_widget
    def apply_manual(): st.session_state.stock_count = st.session_state.num_input

    c1, c2 = st.columns([3, 1])
    with c1:
        st.slider("종목 수", 10, 200, key='slider_widget', value=st.session_state.stock_count, on_change=update_slider)
    with c2:
        st.number_input("직접 입력", 10, 500, key='num_input', value=st.session_state.stock_count, on_change=apply_manual)
        
    if st.button("✅ 수치 적용"):
        apply_manual()
        st.session_state.slider_widget = st.session_state.stock_count
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
    
    status_box.success(f"✅ 기준금리 **{applied_rate}%** 적용 | 데이터 분석을 시작합니다...")
    time.sleep(0.5)
    
    p_bar = st.progress(0)
    # [수정 완료] DataFrame을 직접 넘깁니다!
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
        numeric_cols = ['현재가', '적정가', '괴리율', 'EPS', 'BPS', 'ROE(%)', '공포지수']
        for c in df.columns:
            if '평균주가' in c or '적정주가' in c or c in numeric_cols:
                df[c] = df[c].apply(to_float)
            
        df = df.drop_duplicates(['종목코드'], keep='last')
        df = df[df['적정가'] > 0]
        
        if not df.empty:
            if "괴리율" in sort_opt: df = df.sort_values(by='괴리율', ascending=False)
            elif "ROE" in sort_opt: df = df.sort_values(by='ROE(%)', ascending=False)
            else: df = df.sort_values(by='공포지수', ascending=True)
            
            df = df.reset_index(drop=True)
            df.index += 1
            
            # UI 고정
            df.index.name = "순위"
            df_display = df.set_index('종목명', append=True)
            
            base_cols = ['현재가', '적정가', '괴리율', '공포지수', 'ROE(%)', 'EPS', 'BPS']
            hist_cols = [c for c in df.columns if '년' in c and 'Q' in c]
            hist_cols.sort(reverse=True) 
            
            final_cols = base_cols + hist_cols
            
            top = df.iloc[0]
            st.info(f"🥇 **1위: {top['종목명']}** | 현재 괴리율: {top['괴리율']}%")

            st.dataframe(
                df_display[final_cols].style.applymap(
                    lambda x: 'color: red; font-weight: bold;' if x > 20 else ('color: blue;' if x < 0 else 'color: black;'), 
                    subset=['괴리율']
                ).format("{:,.0f}", subset=['현재가', '적정가', 'EPS', 'BPS'] + [c for c in hist_cols]),
                height=800,
                use_container_width=True
            )
        else: st.warning("결과 데이터가 없습니다.")
    except Exception as e: st.error(f"오류: {e}")
else: st.info("👈 위에서 [분석 시작] 버튼을 눌러주세요.")

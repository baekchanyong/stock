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
DB_FILE = "stock_analysis_v67.csv"

st.set_page_config(page_title="V67 가치투자 분석기 (적정주가 복구)", page_icon="💎", layout="wide")

# --- [핵심] 강력한 숫자 변환 함수 ---
def to_float(val):
    """
    어떤 이상한 값이 들어와도 강제로 실수형(float)으로 변환합니다.
    """
    if pd.isna(val) or val == '' or val == 'N/A': return 0.0
    try:
        # 문자열인 경우 쉼표, 퍼센트 제거
        if isinstance(val, str):
            clean_val = val.replace(',', '').replace('%', '').strip()
            if clean_val == '-' or clean_val == '': return 0.0
            return float(clean_val)
        # 이미 숫자인 경우
        return float(val)
    except:
        return 0.0

# --- 한국은행 기준금리 ---
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

# --- 과거 금리 ---
def get_historical_base_rate(date_str):
    return 3.50

# --- 데이터 수집 (휴일 보정) ---
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

# --- [핵심 수정] 데이터 확보 및 역산 로직 강화 ---
def get_robust_metrics(code, row):
    """
    EPS, BPS가 없으면 PER, PBR과 주가를 이용해 강제로 역산합니다.
    """
    current_price = to_float(row.get('Close', 0))
    
    # 1. 1차 시도: 리스트에 있는 값 가져오기
    eps = to_float(row.get('EPS', 0))
    bps = to_float(row.get('BPS', 0))
    per = to_float(row.get('PER', 0))
    pbr = to_float(row.get('PBR', 0))
    
    # 2. 2차 시도: 야후 파이낸스 (0일 경우만)
    if eps == 0 or bps == 0:
        try:
            ticker = yf.Ticker(f"{code}.KS")
            info = ticker.info
            if eps == 0 and info.get('trailingEps'): eps = float(info['trailingEps'])
            if bps == 0 and info.get('bookValue'): bps = float(info['bookValue'])
        except: pass
        
    # 3. 3차 시도: 역산 (가장 확실한 방법)
    # EPS = 주가 / PER
    if eps == 0 and current_price > 0 and per > 0:
        eps = current_price / per
        
    # BPS = 주가 / PBR
    if bps == 0 and current_price > 0 and pbr > 0:
        bps = current_price / pbr
        
    # 4. 최후의 방어: PBR은 있는데 BPS가 없으면 역산
    # 그래도 없으면 0 리턴 (적자 기업 등)
        
    return eps, bps

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

# --- CSV 저장 (평탄화) ---
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

    # 과거 데이터 스냅샷 로딩
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
    
    chart_start = (today - timedelta(days=365*2.5)).strftime('%Y-%m-%d')
    today_str = today.strftime('%Y-%m-%d')

    for step, (idx, row) in enumerate(target_stocks.iterrows()):
        code = str(row['Code'])
        name = row['Name']
        
        if name in ["맥쿼리인프라", "SK리츠"]: continue
        
        progress_bar.progress(min((step + 1) / total, 1.0))
        status_text.text(f"⏳ [{step+1}/{total}] {name} 분석 중...")
        
        try:
            current_price = to_float(row.get('Close', 0))
            
            # [수정] 강력해진 데이터 확보 함수 호출
            eps_now, bps_now = get_robust_metrics(code, row)
            
            time.sleep(0.02)
            df_chart_full = fdr.DataReader(code, chart_start, today_str)
            
            fg_score_now = 50
            if not df_chart_full.empty:
                fg_score_now = calculate_fear_greed_from_slice(df_chart_full.tail(60))
            
            # 적정주가 계산 (안전장치: 금리 0 방지)
            base_rate = applied_rate if applied_rate > 0 else 3.5
            
            # 수익가치 (EPS 기반)
            earnings_val = eps_now / (base_rate/100)
            
            # 자산가치 (BPS 기반)
            asset_val = bps_now
            
            # 7:3 가중치
            base_fair = (earnings_val * 0.7) + (asset_val * 0.3)
            
            # 심리 보정
            sentiment = 1 + ((50 - fg_score_now)/50 * 0.1)
            fair_now = base_fair * sentiment
            
            gap_now = 0
            if current_price > 0:
                gap_now = (fair_now - current_price) / current_price * 100
            
            # ROE 계산 (보여주기용)
            roe_now = 0
            if bps_now > 0: roe_now = (eps_now / bps_now) * 100
            
            data_row = {
                '기본정보_종목코드': code,
                '기본정보_종목명': name,
                '현재정보_현재가': round(current_price, 0),
                '현재정보_적정주가': round(fair_now, 0),
                '현재정보_괴리율': round(gap_now, 2),
                '지표_공포지수': round(fg_score_now, 1),
                '지표_EPS': round(eps_now, 0),
                '지표_BPS': round(bps_now, 0),
                '지표_ROE(%)': round(roe_now, 2)
            }
            
            # 과거 데이터 (히스토리) 처리
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
                    q_chart = df_chart_full.loc[q_start_str:q_date]
                    if not q_chart.empty:
                        q_avg_price = q_chart['Close'].mean()
                        
                        # 스냅샷 찾기
                        found_snap = None
                        for snap_date in snapshot_dfs.keys():
                            diff = abs((datetime.strptime(snap_date, '%Y-%m-%d') - q_end_dt).days)
                            if diff < 10:
                                found_snap = snapshot_dfs[snap_date]
                                break
                        
                        if found_snap is not None and code in found_snap.index:
                            snap_row = found_snap.loc[code]
                            
                            # [중요] 과거 데이터도 강력한 역산 로직 적용
                            q_eps, q_bps = get_robust_metrics(code, snap_row)
                            
                            q_fg = calculate_fear_greed_from_slice(q_chart)
                            q_rate = get_historical_base_rate(q_date)
                            
                            # 적정주가 계산
                            q_rate = q_rate if q_rate > 0 else 3.5
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

st.title("💎 V67 가치투자 분석기 (적정주가 복구)")

with st.expander("📘 **[필독] 적정주가 산출 공식**", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🧮 적정주가 (수익7 : 자산3)")
        st.latex(r"\text{적정주가} = \left[ \left( \frac{\text{EPS}}{\text{금리}} \times 0.7 \right) + \left( \text{BPS} \times 0.3 \right) \right] \times \text{심리보정}")
        st.caption("* 금리: 한국은행 기준금리 (약 3.25%)")
    with c2:
        st.markdown("##### 👻 공포탐욕지수")
        st.latex(r"\text{Index} = (\text{RSI}_{14} \times 0.5) + (\text{이격도}_{20} \text{ 점수} \times 0.5)")

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
        st.slider("종목 수", 10, 500, key='slider_key', value=st.session_state.stock_count, on_change=update_from_slider)
    with c2:
        st.number_input("직접 입력", 10, 500, key='num_key', value=st.session_state.stock_count)
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
    
    status_box.success(f"✅ 기준금리 **{applied_rate}%** 적용 | 분석을 시작합니다...")
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
        
        # [수정] 적정가 0원인 종목도 표시 (데이터 상태 확인용)
        # df = df[df['현재정보_적정주가'] > 0] <--- 제거함
        
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

        # UI 복원
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
        
    except Exception as e: st.error(f"표시 오류 상세: {e}")
else: st.info("👈 위에서 [분석 시작] 버튼을 눌러주세요.")

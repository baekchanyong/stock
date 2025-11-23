import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
import time
import requests
from datetime import datetime, timedelta

# --- 설정 ---
DB_FILE = "stock_analysis_v25.csv"

st.set_page_config(page_title="V25 가치투자 분석기", page_icon="🧬", layout="wide")

# --- 숫자 변환 함수 ---
def to_float(val):
    try:
        if pd.isna(val) or val == '' or str(val).strip() == '-': return 0.0
        return float(str(val).replace(',', '').replace('%', ''))
    except:
        return 0.0

# --- 네이버 금융 크롤링 (기존 유지) ---
def get_naver_real_fundamentals(code):
    try:
        target_code = code
        if len(code) == 6 and code.isdigit():
            if not code.endswith('0'):
                target_code = code[:-1] + '0'
        
        url = f"https://finance.naver.com/item/main.naver?code={target_code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, headers=headers)
        response.encoding = 'cp949'
        
        dfs = pd.read_html(response.text)
        
        eps = 0.0
        bps = 0.0
        
        for df in dfs:
            df_str = df.to_string()
            if 'EPS' in df_str or 'BPS' in df_str or '주당순이익' in df_str:
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
                if eps > 0 and bps > 0:
                    break
        return eps, bps
    except Exception:
        return 0, 0

# --- 공포탐욕지수 ---
def calculate_fear_greed(df):
    if len(df) < 60: return 50
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    ma20 = df['Close'].rolling(window=20).mean()
    disparity = (df['Close'] / ma20) * 100
    
    # 이격도 점수화: 90이하=0점, 110이상=100점, 그 사이는 비율
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
def run_update_process(target_date, target_num, status_text, progress_bar):
    target_str = target_date.strftime('%Y-%m-%d')
    today_str = datetime.now().strftime('%Y-%m-%d')
    is_backtest = (target_str != today_str)

    start_date = (target_date - timedelta(days=365 * 5)).strftime('%Y-%m-%d')

    status_text.info(f"📅 {target_str} 기준 데이터를 복원 중입니다...")

    try:
        df_krx = fdr.StockListing('KRX', target_str)
        df_krx = df_krx[df_krx['Market'].isin(['KOSPI'])]
        df_krx = df_krx.sort_values(by='Marcap', ascending=False)
        target_stocks = df_krx.head(target_num)
    except Exception as e:
        st.error(f"목록 다운로드 실패: {e}")
        return
    
    current_prices_map = {}
    if is_backtest:
        try:
            df_now = fdr.StockListing('KRX') 
            current_prices_map = df_now.set_index('Code')['Close'].to_dict()
        except: pass

    if os.path.exists(DB_FILE): os.remove(DB_FILE)

    new_data = []
    total = len(target_stocks)
    
    for step, (idx, row) in enumerate(target_stocks.iterrows()):
        code = str(row['Code'])
        name = row['Name']
        
        if name in ["맥쿼리인프라", "SK리츠"]: continue
        
        progress_val = min((step + 1) / total, 1.0)
        progress_bar.progress(progress_val)
        status_text.text(f"⏳ [{step+1}/{total}] {name} 분석 중...")
        
        try:
            price_at_target = to_float(row.get('Close', 0))
            price_now = price_at_target 
            if is_backtest and code in current_prices_map:
                price_now = to_float(current_prices_map[code])
            
            # 1. 재무 데이터
            eps, bps = get_naver_real_fundamentals(code)
            if eps == 0 and 'EPS' in row: eps = to_float(row['EPS'])
            if bps == 0 and 'BPS' in row: bps = to_float(row['BPS'])
            
            # 2. 공포지수
            time.sleep(0.05)
            fg_score = 50
            try:
                df_chart = fdr.DataReader(code, start_date, target_str)
                if not df_chart.empty:
                    fg_score = calculate_fear_greed(df_chart)
                    if not pd.isna(df_chart['Close'].iloc[-1]):
                        price_at_target = df_chart['Close'].iloc[-1]
                        if not is_backtest: price_now = price_at_target
            except: pass

            # 3. [개선] 적정주가 계산 (ROE 프리미엄 적용)
            base_per = 15.0
            base_pbr = 1.2
            
            roe = 0
            if bps > 0:
                roe = (eps / bps) * 100
            
            roe_premium_per = max(0, roe - 10) * 1.0 
            roe_premium_pbr = max(0, roe - 10) * 0.1
            
            final_target_per = base_per + roe_premium_per
            final_target_pbr = base_pbr + roe_premium_pbr
            
            k_factor = 1 + ((50 - fg_score) / 50 * 0.1)
            
            final_target_per *= k_factor
            final_target_pbr *= k_factor
            
            fair_price = (eps * final_target_per * 0.7) + (bps * final_target_pbr * 0.3)
            
            gap = 0
            if price_at_target > 0 and fair_price > 0:
                gap = (fair_price - price_at_target) / price_at_target * 100
            
            data_row = {
                '종목코드': code,
                '종목명': name,
                '기준일': target_str,
                '기준일가격': round(price_at_target, 0),
                '현재가격': round(price_now, 0),
                '적정주가': round(fair_price, 0),
                '괴리율': round(gap, 2),
                '공포지수': round(fg_score, 1),
                'EPS': round(eps, 0),
                'BPS': round(bps, 0),
                'ROE(%)': round(roe, 2)
            }
            new_data.append(data_row)
            
            if len(new_data) >= 10:
                save_to_csv(new_data)
                new_data = [] 
        except: continue

    if new_data: save_to_csv(new_data)
    progress_bar.empty()
    return True

# --- 메인 화면 ---

st.title("🧬 가치투자 분석기 V25 (정렬 기능 추가)")

# 설명 섹션
with st.expander("📘 **[필독] 적정주가 & 공포지수 산출 공식 (Click)**", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🧮 1. 적정주가 (ROE 보정)")
        st.latex(r'''적정주가 = (EPS \times M_{per} \times 0.7) + (BPS \times M_{pbr} \times 0.3)''')
        st.markdown("""
        * **기본 멀티플:** PER 15배, PBR 1.2배
        * **ROE 프리미엄:** ROE가 10%를 초과하면, 초과분만큼 목표 배수를 상향합니다. (고성장주 저평가 문제 해결)
        * **공포 보정:** 공포지수가 낮을수록 적정주가 추가 할증
        """)
        
    with c2:
        st.markdown("##### 👻 2. 공포/탐욕지수 (0~100)")
        st.latex(r'''Index = (RSI_{14} \times 0.5) + (Disparity_{score} \times 0.5)''')
        st.markdown("""
        * **RSI:** 14일 기준 과매수/과매도 (0~100)
        * **이격도 점수:** 90% 이하면 0점(공포), 110% 이상이면 100점(탐욕)
        """)

st.divider()

tab1, tab2 = st.tabs(["⚙️ 데이터 분석 설정", "📊 분석 결과 리포트"])

with tab1:
    st.header("1. 분석 조건 설정")
    col1, col2 = st.columns(2)
    with col1:
        target_date = st.date_input("📅 분석 기준일", value=datetime.now(), min_value=datetime(2015, 1, 1), max_value=datetime.now())
    with col2:
        target_count = st.slider("분석 종목 수", 10, 200, 50)
    
    if st.button("▶️ 분석 시작 (Start)", type="primary"):
        status_box = st.empty()
        p_bar = st.progress(0)
        is_done = run_update_process(target_date, target_count, status_box, p_bar)
        if is_done:
            status_box.success(f"✅ {target_date.strftime('%Y-%m-%d')} 기준 분석 완료! 옆 탭을 확인하세요.")

with tab2:
    st.header("🏆 투자 추천 순위")
    
    # [새로운 기능] 정렬 옵션 추가
    sort_option = st.radio(
        "🔀 정렬 기준 선택", 
        ["괴리율 높은 순 (저평가 추천)", "📈 가격 상승액 순 (현재가 > 기준가)", "📉 가격 하락액 순 (현재가 < 기준가)"],
        horizontal=True
    )

    if st.button("🔄 결과 표 새로고침"): st.rerun()

    if os.path.exists(DB_FILE):
        try:
            df_res = pd.read_csv(DB_FILE)
            for col in ['기준일가격', '현재가격', '적정주가', '괴리율', 'EPS', 'BPS', '공포지수', 'ROE(%)']:
                if col in df_res.columns: df_res[col] = df_res[col].apply(to_float)

            # [새로운 기능] 차이금액 계산
            df_res['차이금액'] = df_res['현재가격'] - df_res['기준일가격']

            df_res = df_res.drop_duplicates(['종목코드'], keep='last')
            df_res = df_res[df_res['적정주가'] > 0]
            
            if not df_res.empty:
                # 정렬 로직
                if "괴리율" in sort_option:
                    df_res = df_res.sort_values(by='괴리율', ascending=False)
                elif "상승액" in sort_option:
                    df_res = df_res.sort_values(by='차이금액', ascending=False)
                elif "하락액" in sort_option:
                    df_res = df_res.sort_values(by='차이금액', ascending=True)

                df_res = df_res.reset_index(drop=True)
                df_res.index += 1
                
                top = df_res.iloc[0]
                st.info(f"🥇 **1위: {top['종목명']}** | 차이금액: {top['차이금액']:+,.0f}원 | 괴리율: {top['괴리율']}%")
                
                st.dataframe(
                    df_res[['기준일', '종목명', '기준일가격', '현재가격', '차이금액', '적정주가', '괴리율', '공포지수', 'EPS', 'BPS', 'ROE(%)']].style.applymap(
                        lambda x: 'color: red; font-weight: bold;' if x > 20 else ('color: blue;' if x < 0 else 'color: black;'), 
                        subset=['괴리율']
                    ).applymap(
                        lambda x: 'color: red; font-weight: bold;' if x > 0 else 'color: blue; font-weight: bold;',
                        subset=['차이금액']
                    ).format("{:,.0f}", subset=['기준일가격', '현재가격', '차이금액', '적정주가', 'EPS', 'BPS']),
                    height=800,
                    use_container_width=True
                )
            else: st.warning("데이터가 없습니다.")
        except Exception as e: st.error(f"오류: {e}")
    else: st.info("👈 [⚙️ 데이터 분석 설정] 탭에서 시작해주세요.")
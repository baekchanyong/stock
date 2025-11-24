import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta

# --- 설정 ---
DB_FILE = "stock_analysis_v27.csv"

st.set_page_config(page_title="V27 4분기 평균 가치투자", page_icon="⚖️", layout="wide")

# --- 헬퍼 함수 ---
def to_float(val):
    try:
        if pd.isna(val) or val == '' or str(val).strip() == '-': return 0.0
        return float(str(val).replace(',', '').replace('%', ''))
    except: return 0.0

# --- 공포탐욕지수 (차트 데이터 슬라이싱 활용) ---
def calculate_fear_greed_from_slice(df_slice):
    """
    잘라낸 차트 데이터로 공포지수 계산
    """
    if len(df_slice) < 20: return 50 # 데이터 너무 적으면 중립
    
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
        # 마지막 값 사용
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

# --- 핵심 분석 엔진 (쿼터백 시스템) ---
def run_quarterly_analysis(target_date, target_num, status_text, progress_bar):
    
    # 1. 4개의 시점 날짜 계산 (0, -3, -6, -9개월)
    dates = []
    for i in range(4):
        d = target_date - timedelta(days=91 * i) # 약 3개월 간격
        dates.append(d.strftime('%Y-%m-%d'))
    
    # 백테스팅 여부 확인 (가장 최근 날짜 기준)
    today_str = datetime.now().strftime('%Y-%m-%d')
    is_backtest = (dates[0] != today_str)

    status_text.info(f"📅 4개 분기 데이터({', '.join(dates)})를 모두 복원 중입니다... (속도 최적화 적용)")

    # 2. [속도 최적화] 4개 시점의 KRX 리스트를 미리 한 번에 다 가져옴 (캐싱)
    # 루프 안에서 매번 부르면 200종목 * 4회 = 800번 요청해야 해서 엄청 느림 -> 미리 4번만 요청
    df_krx_snapshots = {}
    
    try:
        # 메인 리스트 (종목 선정용 - 가장 최근 기준일)
        df_main = fdr.StockListing('KRX', dates[0])
        df_main = df_main[df_main['Market'].isin(['KOSPI'])]
        df_main = df_main.sort_values(by='Marcap', ascending=False)
        target_stocks = df_main.head(target_num)
        
        # 4개 시점 데이터 미리 로드
        for d in dates:
            status_text.text(f"📥 과거 데이터셋 복원 중... ({d})")
            snapshot = fdr.StockListing('KRX', d)
            # 빠른 검색을 위해 종목코드를 인덱스로 설정
            df_krx_snapshots[d] = snapshot.set_index('Code')
            
    except Exception as e:
        st.error(f"데이터셋 로드 실패: {e}")
        return

    # 현재가 로딩 (수익률 검증용)
    current_prices_map = {}
    if is_backtest:
        try:
            df_now = fdr.StockListing('KRX')
            current_prices_map = df_now.set_index('Code')['Close'].to_dict()
        except: pass

    if os.path.exists(DB_FILE): os.remove(DB_FILE)

    total = len(target_stocks)
    new_data = []
    
    # 차트 데이터용 시작일 (가장 옛날 기준일로부터 1년 전)
    chart_start_date = (datetime.strptime(dates[-1], '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')

    # --- 종목별 반복 분석 시작 ---
    for step, (idx, row) in enumerate(target_stocks.iterrows()):
        code = str(row['Code'])
        name = row['Name']
        
        if name in ["맥쿼리인프라", "SK리츠"]: continue
        
        progress_bar.progress(min((step + 1) / total, 1.0))
        status_text.text(f"⏳ [{step+1}/{total}] {name} : 1년치 흐름 정밀 분석 중...")
        
        try:
            # [속도 최적화] 차트 데이터를 1번만 가져와서 메모리에서 자름
            time.sleep(0.05)
            df_chart_full = fdr.DataReader(code, chart_start_date, dates[0])
            
            quarterly_fair_prices = [] # 4번의 적정주가를 담을 리스트
            
            # --- 4분기 반복 계산 ---
            for d in dates:
                # 해당 시점의 재무 데이터 꺼내기
                if code not in df_krx_snapshots[d].index:
                    continue # 그 당시에 상장 안 되어 있었으면 스킵
                
                snap_row = df_krx_snapshots[d].loc[code]
                
                price_then = to_float(snap_row.get('Close', 0))
                eps = to_float(snap_row.get('EPS', 0))
                bps = to_float(snap_row.get('BPS', 0))
                
                # 역산 로직 (데이터 누락 방지)
                if eps == 0 and price_then > 0:
                    per = to_float(snap_row.get('PER', 0))
                    if per > 0: eps = price_then / per
                
                if bps == 0 and price_then > 0:
                    pbr = to_float(snap_row.get('PBR', 0))
                    if pbr > 0: bps = price_then / pbr
                
                # 공포지수 (차트 슬라이싱)
                # 전체 차트에서 해당 날짜(d) 이전 데이터만 잘라냄
                fg_score = 50
                if not df_chart_full.empty:
                    chart_slice = df_chart_full.loc[:d].tail(60) # 과거 60일치
                    fg_score = calculate_fear_greed_from_slice(chart_slice)

                # ROE 프리미엄 및 적정주가 (그 시점 기준)
                base_per = 15.0
                base_pbr = 1.2
                
                roe = 0
                if bps > 0: roe = (eps / bps) * 100
                
                roe_premium_per = max(0, roe - 10) * 1.0 
                roe_premium_pbr = max(0, roe - 10) * 0.1
                
                final_target_per = (base_per + roe_premium_per) * (1 + ((50 - fg_score) / 50 * 0.1))
                final_target_pbr = (base_pbr + roe_premium_pbr) * (1 + ((50 - fg_score) / 50 * 0.1))
                
                q_fair_price = (eps * final_target_per * 0.7) + (bps * final_target_pbr * 0.3)
                
                if q_fair_price > 0:
                    quarterly_fair_prices.append(q_fair_price)

            # --- 최종 평균 산출 ---
            if not quarterly_fair_prices: continue
            
            avg_fair_price = sum(quarterly_fair_prices) / len(quarterly_fair_prices)
            
            # 기준일(가장 최근) 가격
            price_base = to_float(row.get('Close', 0))
            
            # 현재 가격 (수익률용)
            price_now = price_base
            if is_backtest and code in current_prices_map:
                price_now = to_float(current_prices_map[code])
            
            gap = 0
            if price_base > 0:
                gap = (avg_fair_price - price_base) / price_base * 100
            
            # 데이터 저장 (가장 최근 시점의 재무정보 표시)
            # EPS, BPS 등은 참고용으로 가장 최근 분기 것만 보여줌
            current_eps = to_float(row.get('EPS', 0))
            current_bps = to_float(row.get('BPS', 0))
            
            data_row = {
                '종목코드': code,
                '종목명': name,
                '기준일': dates[0],
                '기준일가격': round(price_base, 0),
                '현재가격': round(price_now, 0),
                '평균적정주가': round(avg_fair_price, 0), # 1년치 평균값
                '괴리율': round(gap, 2),
                '최근공포지수': round(fg_score, 1), # 참고용
                'EPS': round(current_eps, 0),
                'BPS': round(current_bps, 0),
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

st.title("⚖️ 가치투자 분석기 V27 (1년 평균 보정)")

with st.expander("📘 **[NEW] 4분기 평균 적정주가 산출 방식 (Click)**", expanded=True):
    st.markdown("""
    이 버전은 단순히 현재 시점만 보는 것이 아니라, **과거 1년(4개 분기)의 적정주가를 모두 계산하여 평균**을 냅니다.
    
    1. **분석 시점:** 기준일로부터 0개월, 3개월, 6개월, 9개월 전 데이터를 모두 복원합니다.
    2. **개별 계산:** 각 시점마다 [실적 $\times$ 공포지수 $\times$ ROE 프리미엄]을 적용해 적정가를 구합니다.
    3. **최종 산출:** $$ \text{최종 적정주가} = \frac{\text{1분기적정가} + \text{2분기적정가} + \text{3분기적정가} + \text{4분기적정가}}{4} $$
    
    👉 **장점:** 일시적인 실적 쇼크나 주가 급등락에 따른 왜곡을 방지하고, 기업의 **기초 체력 추세**를 반영합니다.
    """)

st.divider()

tab1, tab2 = st.tabs(["⚙️ 데이터 분석 설정", "📊 분석 결과 리포트"])

with tab1:
    st.header("1. 분석 조건 설정")
    col1, col2 = st.columns(2)
    with col1:
        target_date = st.date_input("📅 분석 기준일", value=datetime.now(), min_value=datetime(2016, 1, 1), max_value=datetime.now())
        st.caption("선택한 날짜를 포함해 과거 1년치(4분기) 데이터를 정밀 분석합니다.")
    with col2:
        target_count = st.slider("분석 종목 수", 10, 200, 50)
    
    if st.button("▶️ 정밀 분석 시작 (Deep Scan)", type="primary"):
        status_box = st.empty()
        p_bar = st.progress(0)
        is_done = run_quarterly_analysis(target_date, target_count, status_box, p_bar)
        if is_done:
            status_box.success(f"✅ 정밀 분석 완료! 옆 탭을 확인하세요.")

with tab2:
    st.header("🏆 1년 평균 가치투자 순위")
    
    sort_option = st.radio(
        "🔀 정렬 기준", 
        ["괴리율 높은 순", "📈 가격 상승액 순", "📉 가격 하락액 순"],
        horizontal=True
    )

    if st.button("🔄 결과 표 새로고침"): st.rerun()

    if os.path.exists(DB_FILE):
        try:
            df_res = pd.read_csv(DB_FILE)
            for col in ['기준일가격', '현재가격', '평균적정주가', '괴리율', 'EPS', 'BPS', '최근공포지수']:
                if col in df_res.columns: df_res[col] = df_res[col].apply(to_float)

            df_res['차이금액'] = df_res['현재가격'] - df_res['기준일가격']
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
                
                # 모바일 고정 뷰 설정
                df_res.index.name = "순번"
                df_display = df_res.set_index('종목명', append=True)
                
                top = df_res.iloc[0]
                st.info(f"🥇 **1위: {top['종목명']}** | 1년평균 적정가: {top['평균적정주가']:,.0f}원 | 괴리율: {top['괴리율']}%")
                
                st.dataframe(
                    df_display[['기준일', '기준일가격', '현재가격', '차이금액', '평균적정주가', '괴리율', '최근공포지수', 'EPS', 'BPS']].style.applymap(
                        lambda x: 'color: red; font-weight: bold;' if x > 20 else ('color: blue;' if x < 0 else 'color: black;'), 
                        subset=['괴리율']
                    ).applymap(
                        lambda x: 'color: red; font-weight: bold;' if x > 0 else 'color: blue; font-weight: bold;',
                        subset=['차이금액']
                    ).format("{:,.0f}", subset=['기준일가격', '현재가격', '차이금액', '평균적정주가', 'EPS', 'BPS']),
                    height=800,
                    use_container_width=True
                )
            else: st.warning("데이터가 없습니다.")
        except Exception as e: st.error(f"오류: {e}")
    else: st.info("👈 [⚙️ 데이터 분석 설정] 탭에서 시작해주세요.")

import streamlit as st
import pandas as pd
import requests
import io
import time
from bs4 import BeautifulSoup

# 페이지 기본 설정
st.set_page_config(page_title="주식탐색기 Ver 1.0", page_icon="📈", layout="wide")

# --- 헤더 ---
st.title("주식탐색기 Ver 1.0")
with st.expander("📝 패치노트 (클릭하여 열기)"):
    st.write("(26.04.15) 1.0Ver 최초 배포")

# --- 계산식 안내 ---
st.markdown("### 🧮 산출 방식 안내")
st.markdown("""
- **적정주가**: `(연간 EPS * 10 or 15) + BPS - (부채 패널티)`
- **목표주가**: `(연간 예상 EPS * 10 or 15) + BPS - (유동부채 / 주식수)`
- **데이터 출처**: 연간 EPS(최신 실적/예상), 분기 EPS(최신 예상치)
- **정렬**: 괴리율(10) 낮은 순 (저평가 매력 순)
""")

st.divider()

# --- 상태 관리 초기화 ---
if 'running' not in st.session_state: st.session_state.running = False
if 'results' not in st.session_state: st.session_state.results = []
if 'skipped_results' not in st.session_state: st.session_state.skipped_results = []
if 'market_df' not in st.session_state: st.session_state.market_df = pd.DataFrame()
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0
if 'target_stocks' not in st.session_state: st.session_state.target_stocks = []

# --- 시장(KOSPI/KOSDAQ) 정보 로드 함수 ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_page_data(sosok, page):
    url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.find('table', {'class': 'type_2'})
        if not table: return [], False
        
        data = []
        has_data = False
        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 5:
                a_tag = cols[1].find('a')
                if a_tag:
                    name = a_tag.text.strip()
                    code = a_tag['href'].split('code=')[-1]
                    close_txt = cols[2].text.strip().replace(',', '')
                    stocks_txt = cols[7].text.strip().replace(',', '')
                    if close_txt and stocks_txt:
                        data.append({
                            'Code': code, 'Name': name,
                            'Close': float(close_txt),
                            'Stocks': float(stocks_txt) * 1000
                        })
                        has_data = True
        return data, has_data
    except: return [], False

if st.session_state.market_df.empty:
    loading_placeholder = st.empty()
    progress_bar = st.empty()
    
    data = []
    start_time = time.time()
    
    total_est = 55
    current_idx = 0
    for sosok, market_name in [(0, 'KOSPI'), (1, 'KOSDAQ')]:
        marcap_rank = 1
        for page in range(1, 45):
            current_idx += 1
            percent = min(current_idx / total_est, 1.0)
            elapsed = time.time() - start_time
            if current_idx > 1:
                eta = elapsed / current_idx * (total_est - current_idx)
            else:
                eta = 0
                
            loading_placeholder.markdown(f"### ⏳ Data Loading 중... (예상 남은 시간: {max(0, int(eta))}초)")
            progress_bar.progress(percent)
            
            page_data, has_data = fetch_page_data(sosok, page)
            if not has_data:
                break
            
            for item in page_data:
                item['Market'] = market_name
                item['Marcap_Rank'] = marcap_rank
                data.append(item)
                marcap_rank += 1
                
    st.session_state.market_df = pd.DataFrame(data)
    loading_placeholder.empty()
    progress_bar.empty()
    st.rerun()

# --- 스크래핑 및 계산 함수 ---
def get_numeric_value(row):
    for val in row.values[::-1]:
        if pd.notna(val) and str(val).replace('.', '', 1).replace('-', '', 1).isdigit(): return float(val)
    return 0.0

def analyze_stock(ticker, name, market, current_price, shares, marcap_rank):
    is_pref = not str(ticker).endswith('0') or name.endswith('우') or '우(' in name or '우B' in name
    if is_pref:
        return None, {"시장": market, "시총순위": marcap_rank, "종목명": name, "현재주가": current_price, "제외사유": "우선주 제외"}
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    url_fin = f"https://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp?pGB=1&gicode=A{ticker}"
    url_main = f"https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A{ticker}"
    
    t_equity = 0.0; t_debt = 0.0; c_liab = 0.0; a_eps = 0.0; q_eps = 0.0; bps = 0.0
    try:
        res_fin = requests.get(url_fin, headers=headers, timeout=5)
        res_fin.encoding = 'utf-8'
        tables_fin = pd.read_html(io.StringIO(res_fin.text))
        df_bs = tables_fin[2]
        for _, row in df_bs.iterrows():
            nm = str(row.iloc[0]).replace('\xa0', ' ').replace(' ', '').strip()
            if nm in ['자본', '자본총계'] and t_equity == 0: t_equity = get_numeric_value(row) * 100000000
            if nm in ['부채', '부채총계'] and t_debt == 0: t_debt = get_numeric_value(row) * 100000000
            if '유동부채' in nm and '비유동' not in nm and c_liab == 0: c_liab = get_numeric_value(row) * 100000000
    except: return None, {"시장": market, "시총순위": marcap_rank, "종목명": name, "현재주가": current_price, "제외사유": "재무 데이터 로드 오류"}
        
    try:
        res_main = requests.get(url_main, headers=headers, timeout=5)
        res_main.encoding = 'utf-8'
        tables_main = pd.read_html(io.StringIO(res_main.text))
        for df in tables_main:
            if df.columns.nlevels > 1:
                col_types = [str(c[0]).strip() for c in df.columns]
                for _, row in df.iterrows():
                    nm = str(row.iloc[0]).replace('\xa0', ' ').replace(' ', '').strip()
                    if 'EPS(원)' in nm:
                        for j in range(len(row)-1, 0, -1):
                            val = row.iloc[j]
                            if pd.notna(val) and str(val).replace('.', '', 1).replace('-', '', 1).isdigit():
                                if 'Annual' in col_types[j] and a_eps == 0: a_eps = float(val)
                                elif 'NetQuarter' in col_types[j].replace(' ', '') and q_eps == 0: q_eps = float(val)
                    elif 'BPS(원)' in nm and bps == 0: bps = get_numeric_value(row)
            if a_eps != 0 and q_eps != 0 and bps != 0: break
    except: return None, {"시장": market, "시총순위": marcap_rank, "종목명": name, "현재주가": current_price, "제외사유": "재무 데이터 파싱 오류"}

    if a_eps == 0 or bps == 0 or t_equity == 0: return None, {"시장": market, "시총순위": marcap_rank, "종목명": name, "현재주가": current_price, "제외사유": "필요 데이터 누락"}
    if q_eps == 0: q_eps = a_eps / 4

    d_ratio = (t_debt / t_equity) * 100
    pnlty = ((t_debt - t_equity) / shares) if d_ratio > 100 else 0
    p10 = (a_eps * 10) + bps - pnlty
    p15 = (a_eps * 15) + bps - pnlty
    t10 = (a_eps * 10) + bps - (c_liab / shares)
    t15 = (a_eps * 15) + bps - (c_liab / shares)
    
    reason = None
    if a_eps <= 0: reason = "적자 기업"
    elif p10 <= 0: reason = "적정주가 음수"
    elif t10 <= 0: reason = "목표주가 음수"
    
    data = {"시장": market, "시총순위": marcap_rank, "종목명": name, "현재주가": current_price, "적정주가(10)": p10, "목표주가(10)": t10, "괴리율(10)": ((p10 - current_price) / current_price) * 100 if p10 > 0 else 0, "적정주가(15)": p15, "목표주가(15)": t15, "EPS": a_eps, "BPS": bps, "부채비율(%)": d_ratio, "총부채_원": t_debt, "유동부채_원": c_liab, "총자본_원": t_equity, "상장주식수_원": shares}
    if reason:
        data["제외사유"] = reason
        return (None, data)
    return (data, None)

# --- UI 설정 ---
market_df = st.session_state.market_df
if not market_df.empty:
    st.markdown("### ⚙️ 탐색 모드 설정")
    top_n = 50
    selected_custom = []
    col1, col2 = st.columns([1, 2])
    with col1:
        search_mode = st.radio("모드를 선택하세요:", ("KOSPI 전체 탐색", "KOSPI 상위 N개 탐색", "KOSDAQ 전체 탐색", "KOSDAQ 상위 N개 탐색", "사용자 지정 탐색"))
    with col2:
        if "상위 N개" in search_mode:
            cur_market = "KOSPI" if "KOSPI" in search_mode else "KOSDAQ"
            max_len = len(market_df[market_df['Market'] == cur_market])
            top_n = st.number_input("탐색 개수:", min_value=1, max_value=max_len if max_len > 0 else 1, value=50 if max_len >= 50 else max_len)
        elif search_mode == "사용자 지정 탐색":
            selected_custom = st.multiselect("종목 선택:", market_df['Name'].tolist())

    st.divider()
    
    def get_targets():
        if search_mode == "KOSPI 전체 탐색": return market_df[market_df['Market'] == 'KOSPI'].to_dict('records')
        elif search_mode == "KOSDAQ 전체 탐색": return market_df[market_df['Market'] == 'KOSDAQ'].to_dict('records')
        elif search_mode == "KOSPI 상위 N개 탐색": return market_df[market_df['Market'] == 'KOSPI'].head(top_n).to_dict('records')
        elif search_mode == "KOSDAQ 상위 N개 탐색": return market_df[market_df['Market'] == 'KOSDAQ'].head(top_n).to_dict('records')
        else: return market_df[market_df['Name'].isin(selected_custom)].to_dict('records')

    btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1.2, 1, 1.8])
    with btn_col1:
        if st.button("🚀 새로 탐색", disabled=st.session_state.running):
            st.session_state.target_stocks = get_targets()
            st.session_state.results = []; st.session_state.skipped_results = []; st.session_state.current_idx = 0
            if len(st.session_state.target_stocks) > 0: st.session_state.running = True
            st.rerun()
    with btn_col2:
        if st.button("▶️ 이어서/추가 탐색", disabled=st.session_state.running):
            new_targets = get_targets()
            
            existing_codes = set([s['Code'] for s in st.session_state.target_stocks])
            for stock in new_targets:
                if stock['Code'] not in existing_codes:
                    st.session_state.target_stocks.append(stock)
            
            if st.session_state.current_idx < len(st.session_state.target_stocks):
                st.session_state.running = True
            st.rerun()
    with btn_col3:
        if st.button("⏹️ 일시정지", disabled=not st.session_state.running):
            st.session_state.running = False; st.rerun()

    progress_container = st.empty(); status_text = st.empty()

    def render_result_table():
        def fmt_curr(v): return f"{v/1e8:,.1f} 억원" if v >= 1e8 else "1억 미만"
        if len(st.session_state.results) > 0:
            st.markdown("### 🏆 탐색 결과")
            df = pd.DataFrame(st.session_state.results).sort_values("괴리율(10)", ascending=False).reset_index(drop=True)
            res = pd.DataFrame()
            res["순위"] = df.index+1; res["시장"] = df.get("시장", "KOSPI"); res["시총순위"] = df["시총순위"]; res["종목"] = df["종목명"]; res["현재주가"] = df["현재주가"].apply(lambda x: f"{x:,.0f} 원")
            res["적정주가(10)"] = df["적정주가(10)"].apply(lambda x: f"{x:,.0f} 원"); res["목표주가(10)"] = df["목표주가(10)"].apply(lambda x: f"{x:,.0f} 원")
            res["괴리율(10)"] = df["괴리율(10)"].apply(lambda x: f"{x:.2f} %"); res["적정주가(15)"] = df["적정주가(15)"].apply(lambda x: f"{x:,.0f} 원"); res["목표주가(15)"] = df["목표주가(15)"].apply(lambda x: f"{x:,.0f} 원")
            res["EPS"] = df["EPS"].apply(lambda x: f"{x:,.0f}"); res["BPS"] = df["BPS"].apply(lambda x: f"{x:,.0f}"); res["부채비율"] = df["부채비율(%)"].apply(lambda x: f"{x:.2f} %")
            res["총부채"] = df["총부채_원"].apply(fmt_curr); res["유동부채"] = df["유동부채_원"].apply(fmt_curr); res["총자본"] = df["총자본_원"].apply(fmt_curr); res["주식수"] = df["상장주식수_원"].apply(lambda x: f"{x/1e4:,.0f} 만개")
            st.dataframe(res, use_container_width=True, hide_index=True)
        if len(st.session_state.skipped_results) > 0:
            with st.expander("🚫 분석 제외 종목", expanded=True):
                dfS = pd.DataFrame(st.session_state.skipped_results).sort_values("시총순위")
                skip = pd.DataFrame()
                skip["시장"] = dfS.get("시장", "KOSPI"); skip["시총순위"] = dfS["시총순위"]; skip["종목"] = dfS["종목명"]; skip["사유"] = dfS.get("제외사유", "데이터 오류"); skip["현재주가"] = dfS["현재주가"].apply(lambda x: f"{float(x):,.0f} 원")
                st.dataframe(skip, use_container_width=True, hide_index=True)

    if st.session_state.running:
        total = len(st.session_state.target_stocks)
        prog = progress_container.progress(0.0)
        for i, stock in enumerate(st.session_state.target_stocks[st.session_state.current_idx:], start=st.session_state.current_idx):
            prog.progress((i)/total); status_text.markdown(f"**진행중:** {i}/{total} ({stock['Name']})")
            res, extra = analyze_stock(stock['Code'], stock['Name'], stock.get('Market', 'KOSPI'), float(stock['Close']), float(stock['Stocks']), stock['Marcap_Rank'])
            if res: st.session_state.results.append(res)
            else: st.session_state.skipped_results.append(extra)
            render_result_table(); st.session_state.current_idx = i + 1; time.sleep(0.3)
        st.session_state.running = False; progress_container.empty(); status_text.success("완료!"); st.rerun()
    else: render_result_table()

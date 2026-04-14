import streamlit as st
import pandas as pd
import requests
import io
import time
import FinanceDataReader as fdr

# 페이지 기본 설정
st.set_page_config(page_title="주식탐색기 Ver 1.0", page_icon="📈", layout="wide")

# --- CSS 적용 ---
# 모바일 환경에서 테이블이 잘 보이고 진행상태 바를 고정하도록 추가
st.markdown("""
<style>
    .reportview-container .main .block-container{
        max-width: 1200px;
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 헤더 및 패치노트 ---
st.title("📈 주식탐색기 Ver 1.0")
with st.expander("📝 패치노트 (클릭하여 열기)"):
    st.write("(26.04.14) 1.0Ver 최초배포")

# --- 계산식 안내 ---
st.markdown("### 🧮 산출 방식 안내")
st.markdown("""
- **목표주가** = `EPS * 10 + BPS - (유동부채 / 주식수)`
- **적정주가**:
    - **부채비율 100% 이하**: `EPS * 10 + BPS`
    - **부채비율 100% 초과**: `(EPS * 10 + BPS) - (총부채 - 총자본) / 주식수`
- **괴리율** = `(적정주가 - 현재주가) / 현재주가 * 100` (%)
- **정렬 기준**: 괴리율이 **낮은** 순서 (적정주가 대비 현재가가 높게 설정되어 있는 종목부터 또는 적정주가 비례 저평가 확인)
- **필터링**: EPS 등 계산 결과상 **적자기업(음수 데이터)** 존재 시 목록에서 제외됩니다.
""")

st.divider()

# --- 상태 관리 초기화 ---
if 'running' not in st.session_state:
    st.session_state.running = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'kospi_df' not in st.session_state:
    st.session_state.kospi_df = pd.DataFrame()
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0
if 'target_stocks' not in st.session_state:
    st.session_state.target_stocks = []

# --- KOSPI 정보 로드 함수 ---
from bs4 import BeautifulSoup

@st.cache_data(ttl=3600)
def load_kospi_data():
    try:
        data = []
        marcap_rank = 1
        headers = {'User-Agent': 'Mozilla/5.0'}
        # KOSPI 종목 약 900+개, 페이지 1~40 (페이지당 50개)
        for page in range(1, 20):  # 상위 1000개 정도만 가져와도 충분 (시총 1000위 밖은 관리종목 등이 많음)
            url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page={page}"
            res = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            table = soup.find('table', {'class': 'type_2'})
            if not table:
                break
            rows = table.find_all('tr')
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) > 5:
                    a_tag = cols[1].find('a')
                    if a_tag:
                        name = a_tag.text.strip()
                        code = a_tag['href'].split('code=')[-1]
                        close_txt = cols[2].text.strip().replace(',', '')
                        marcap_txt = cols[6].text.strip().replace(',', '')
                        stocks_txt = cols[7].text.strip().replace(',', '')
                        
                        if close_txt and marcap_txt and stocks_txt:
                            data.append({
                                'Code': code,
                                'Name': name,
                                'Close': float(close_txt),
                                'Marcap': float(marcap_txt),
                                'Stocks': float(stocks_txt) * 1000, # 네이버 상장주식수 단위는 '천주'
                                'Marcap_Rank': marcap_rank
                            })
                            marcap_rank += 1
                            
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"KOSPI 데이터를 불러오는 데 실패했습니다: {e}")
        return pd.DataFrame()

# 데이터 최초 1회 로드
if st.session_state.kospi_df.empty:
    with st.spinner("KOSPI 종목 정보를 불러오는 중입니다..."):
        st.session_state.kospi_df = load_kospi_data()

# --- 스크래핑 및 계산 함수 ---
def get_numeric_value(row):
    """행(Row)에서 가장 최근 연도의 숫자(float) 데이터를 추출"""
    for val in row.values[::-1]:
        if pd.notna(val) and str(val).replace('.', '', 1).replace('-', '', 1).isdigit():
            return float(val)
    return 0.0

def analyze_stock(ticker, name, current_price, shares, marcap_rank):
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # 1. 재무상태표 (총자본, 총부채, 유동부채) - 단위: 억원
    url_fin = f"https://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp?pGB=1&gicode=A{ticker}"
    
    # 2. Financial Highlight (EPS, BPS) - 단위: 원
    url_main = f"https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A{ticker}"
    
    total_equity = 0.0
    total_debt = 0.0
    current_liability = 0.0
    eps = 0.0
    bps = 0.0
    
    try:
        res_fin = requests.get(url_fin, headers=headers, timeout=5)
        tables_fin = pd.read_html(io.StringIO(res_fin.text))
        
        # 재무상태표(연간)은 통상 3번째 테이블 (인덱스 2)
        df_bs = tables_fin[2]
        
        # 각 항목 찾기 (문자열 포함 여부로)
        for _, row in df_bs.iterrows():
            col_name = str(row.iloc[0]).strip()
            if '자본' in col_name and '총계' not in col_name and total_equity == 0:  # 보통 '자본'
                total_equity = get_numeric_value(row) * 100000000
            if '자본총계' in col_name:
                total_equity = get_numeric_value(row) * 100000000
                
            if '부채' in col_name and '총계' not in col_name and total_debt == 0:
                total_debt = get_numeric_value(row) * 100000000
            if '부채총계' in col_name:
                total_debt = get_numeric_value(row) * 100000000
                
            if '유동부채' in col_name:
                current_liability = get_numeric_value(row) * 100000000
                
    except Exception:
        # 정보 누락/에러 시 None 반환
        return None
        
    try:
        res_main = requests.get(url_main, headers=headers, timeout=5)
        tables_main = pd.read_html(io.StringIO(res_main.text))
        
        # 메인 테이블들 중 EPS, BPS가 있는 테이블 찾기
        for df in tables_main:
            if df.columns.nlevels > 0:
                for _, row in df.iterrows():
                    col_name = str(row.iloc[0]).strip()
                    if 'EPS(원)' in col_name:
                        eps = get_numeric_value(row)
                    elif 'BPS(원)' in col_name:
                        bps = get_numeric_value(row)
    except Exception:
        return None

    # 모든 필수 데이터가 확보되었는지 확인, 적자기업(EPS 음수 또는 자본 잠식) 필터링
    if eps <= 0 or bps <= 0 or current_price <= 0 or shares <= 0:
        return None
        
    if total_equity <= 0:
        return None

    # 계산 로직
    debt_ratio = (total_debt / total_equity) * 100

    if debt_ratio <= 100:
        intrinsic_value = (eps * 10) + bps
    else:
        intrinsic_value = (eps * 10) + bps - ((total_debt - total_equity) / shares)
        
    target_price = (eps * 10) + bps - (current_liability / shares)
    
    deviation_rate = ((intrinsic_value - current_price) / current_price) * 100
    
    return {
        "시총순위": marcap_rank,
        "종목명": name,
        "현재주가": current_price,
        "적정주가": intrinsic_value,
        "목표주가": target_price,
        "괴리율(%)": deviation_rate,
        "EPS": eps,
        "BPS": bps,
        "부채비율(%)": debt_ratio,
        "총부채": total_debt,
        "유동부채": current_liability,
        "총자본": total_equity,
        "상장주식수": shares
    }

# --- 사용자 설정 UI ---
kospi_df = st.session_state.kospi_df

if not kospi_df.empty:
    st.markdown("### ⚙️ 탐색 모드 설정")
    col1, col2 = st.columns([1, 2])
    with col1:
        search_mode = st.radio(
            "탐색 모드를 선택하세요:",
            ("KOSPI 전체 탐색", "KOSPI 상위 N개 탐색", "사용자 지정 탐색 (장바구니)")
        )
    
    with col2:
        top_n = 50
        selected_custom = []
        if search_mode == "KOSPI 상위 N개 탐색":
            top_n = st.number_input("탐색할 상위 종목 개수를 입력하세요 (예: 50)", min_value=1, max_value=len(kospi_df), value=50)
        elif search_mode == "사용자 지정 탐색 (장바구니)":
            stock_list = kospi_df['Name'].tolist()
            selected_custom = st.multiselect("탐색할 종목들을 선택하세요:", stock_list)

    st.divider()

    # --- 버튼 액션 로직 ---
    # 버튼을 누르면 상태만 변경하고 즉시 st.rerun() 호출로 렌더링을 제어합니다.
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("🚀 탐색 시작", use_container_width=True, disabled=st.session_state.running):
            # 타겟 리스트 생성
            if search_mode == "KOSPI 전체 탐색":
                st.session_state.target_stocks = kospi_df.to_dict('records')
            elif search_mode == "KOSPI 상위 N개 탐색":
                st.session_state.target_stocks = kospi_df.head(top_n).to_dict('records')
            else:
                st.session_state.target_stocks = kospi_df[kospi_df['Name'].isin(selected_custom)].to_dict('records')
            
            st.session_state.results = []
            st.session_state.current_idx = 0
            if len(st.session_state.target_stocks) > 0:
                st.session_state.running = True
            else:
                st.warning("탐색할 종목이 없습니다.")
            st.rerun()

    with c2:
        if st.button("⏹️ 탐색 정지", use_container_width=True, disabled=not st.session_state.running):
            st.session_state.running = False
            st.rerun()

    # --- 진행 상태 및 표 출력 영역 ---
    progress_container = st.empty()
    status_text = st.empty()
    table_container = st.empty()

    # 결과 테이블 렌더링 함수
    def render_result_table():
        if len(st.session_state.results) > 0:
            df_res = pd.DataFrame(st.session_state.results)
            # 괴리율 낮은 순 정렬
            df_res = df_res.sort_values(by="괴리율(%)", ascending=True).reset_index(drop=True)
            # 포맷팅 적용 전 순위 부여
            df_res.insert(0, "순위", df_res.index + 1)
            
            # 숫자 포맷팅 처리
            format_dict = {
                "현재주가": "{:,.0f} 원",
                "적정주가": "{:,.0f} 원",
                "목표주가": "{:,.0f} 원",
                "괴리율(%)": "{:.2f} %",
                "EPS": "{:,.0f}",
                "BPS": "{:,.0f}",
                "부채비율(%)": "{:.2f} %",
                "총부채": "{:,.0f}",
                "유동부채": "{:,.0f}",
                "총자본": "{:,.0f}",
                "상장주식수": "{:,.0f}"
            }
            styled_df = df_res.style.format(format_dict)
            table_container.dataframe(styled_df, use_container_width=True, height=600)
        elif not st.session_state.running:
            table_container.info("검색된 결과가 없습니다. '탐색 시작'을 눌러주세요.")

    # --- 탐색 실행 루프 ---
    if st.session_state.running:
        total_len = len(st.session_state.target_stocks)
        idx = st.session_state.current_idx
        
        # 프로그레스 바는 실행 중일때만 표시
        prog_bar = progress_container.progress(0.0)
        
        # 여기서는 UI 렌더링 블로킹을 최소화하기 위해 한 번에 1개 종목씩 분석하고 rerun 합니다.
        # Streamlit 1.30+ 부터는 time.sleep 등으로 루프를 돌리는 중에 버튼 클릭 시 즉시 StopException이 발생하여 
        # 안전하게 session_state에 보관된 결과를 남길 수 있습니다.
        
        for i, stock in enumerate(st.session_state.target_stocks[idx:], start=idx):
            # 상태 업데이트
            progress_pct = (i) / total_len
            prog_bar.progress(progress_pct)
            status_text.markdown(f"**진행중:** {i}/{total_len} ({stock['Name']} 분석중...)")
            
            # 주가 정보 추출
            ticker = stock['Code']
            name = stock['Name']
            current_price = float(stock['Close']) if pd.notna(stock['Close']) else 0.0
            shares = float(stock['Stocks']) if 'Stocks' in stock and pd.notna(stock['Stocks']) else 0.0
            marcap_rank = stock['Marcap_Rank']
            
            # 스크래핑 및 계산 수행
            result = analyze_stock(ticker, name, current_price, shares, marcap_rank)
            if result is not None:
                st.session_state.results.append(result)
            
            # 테이블 실시간 업데이트 (옵션: 속도 저하가 심하면 10개마다 업데이트 등 조율 가능)
            render_result_table()
            
            # 다음 렌더링에 이어서 할 수 있도록 인덱스 저장 (정지 방어용)
            st.session_state.current_idx = i + 1
            
            # 서버 과부하 방지 및 방어적 크롤링 딜레이
            time.sleep(0.3)
            
        # 모든 탐색 완료
        st.session_state.running = False
        progress_container.empty()
        status_text.success("탐색이 완료되었습니다!")
        st.rerun()
    else:
        # 실행 중이 아닐 때만 최종/중간 저장 표 표시
        if st.session_state.current_idx > 0 and st.session_state.current_idx < len(st.session_state.target_stocks):
            status_text.warning("탐색이 중지되었습니다. 현재까지의 결과를 표시합니다.")
        render_result_table()

else:
    st.warning("데이터 초기화 중 문제를 겪었습니다. 새로고침을 해주세요.")

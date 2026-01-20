"""
펩톤 효소 추천 시스템 - Streamlit 웹 앱 v2.0

원료 성분 분석 데이터를 업로드하면 최적의 효소를 추천합니다.

v2.0 업데이트:
- 다양한 Excel 형식 지원
- 향상된 오류 처리
- 빈 데이터 자동 필터링

실행: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
import traceback

# src 폴더 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from recommender import EnzymeRecommender, load_composition_data, SubstrateAnalysis, EnzymeRecommendation
except ImportError as e:
    st.error(f"모듈 임포트 오류: {e}")
    st.stop()


# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="🧬 펩톤 효소 추천 시스템",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .enzyme-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .score-badge {
        background: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .warning-badge {
        background: #ffc107;
        color: #333;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 초기화
# ============================================================
@st.cache_resource
def load_recommender():
    """추천 엔진 로드 (캐싱)"""
    # 여러 가능한 경로 시도
    possible_paths = [
        Path(__file__).parent.parent / 'data' / 'enzyme_database_extended.json',
        Path(__file__).parent.parent / 'data' / 'enzyme_database.json',
        Path('data') / 'enzyme_database_extended.json',
        Path('data') / 'enzyme_database.json',
    ]
    
    for db_path in possible_paths:
        if db_path.exists():
            return EnzymeRecommender(str(db_path))
    
    raise FileNotFoundError(f"효소 데이터베이스 파일을 찾을 수 없습니다. 시도한 경로: {possible_paths}")


def create_amino_acid_chart(analysis: SubstrateAnalysis) -> go.Figure:
    """아미노산 조성 차트 생성"""
    aa_profile = analysis.amino_acid_profile
    
    # 주요 아미노산만 필터링
    main_aas = ['Asp', 'Glu', 'Ser', 'Gly', 'Ala', 'Val', 'Leu', 'Ile', 
                'Thr', 'Pro', 'Phe', 'Tyr', 'Trp', 'Lys', 'Arg', 'His', 'Met', 'Cys']
    
    filtered = {k: v for k, v in aa_profile.items() if k in main_aas and v > 0}
    
    if not filtered:
        return None
    
    # 정렬
    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]
    
    # 색상 지정 (그룹별)
    colors = []
    for aa in names:
        if aa in ['Leu', 'Ile', 'Val', 'Phe', 'Trp', 'Met', 'Ala']:
            colors.append('#667eea')  # 소수성 - 보라
        elif aa in ['Lys', 'Arg', 'His']:
            colors.append('#28a745')  # 염기성 - 초록
        elif aa in ['Asp', 'Glu']:
            colors.append('#dc3545')  # 산성 - 빨강
        elif aa in ['Pro', 'Gly']:
            colors.append('#fd7e14')  # 특수 - 주황
        else:
            colors.append('#6c757d')  # 기타 - 회색
    
    fig = go.Figure(data=[
        go.Bar(x=names, y=values, marker_color=colors)
    ])
    
    fig.update_layout(
        title="아미노산 조성 프로파일",
        xaxis_title="아미노산",
        yaxis_title="함량 (g/100g)",
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def create_group_ratio_chart(analysis: SubstrateAnalysis) -> go.Figure:
    """아미노산 그룹 비율 파이 차트"""
    labels = ['소수성', '방향족', '염기성', '산성', '기타']
    
    other_ratio = 1 - (analysis.hydrophobic_ratio + analysis.aromatic_ratio + 
                       analysis.basic_ratio + analysis.acidic_ratio)
    other_ratio = max(0, other_ratio)
    
    values = [
        analysis.hydrophobic_ratio,
        analysis.aromatic_ratio,
        analysis.basic_ratio,
        analysis.acidic_ratio,
        other_ratio
    ]
    
    # 모든 값이 0인 경우 처리
    if sum(values) == 0:
        return None
    
    colors = ['#667eea', '#764ba2', '#28a745', '#dc3545', '#6c757d']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title="아미노산 그룹 비율",
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def display_recommendation(rec: EnzymeRecommendation, expanded: bool = True):
    """효소 추천 결과 표시"""
    
    score_color = "#28a745" if rec.score >= 70 else "#ffc107" if rec.score >= 50 else "#dc3545"
    
    with st.expander(f"#{rec.rank} {rec.enzyme_name} (점수: {rec.score}점)", expanded=expanded):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🌡️ 최적 온도", rec.optimal_temp)
            st.metric("📊 E/S 비율", rec.es_ratio)
        
        with col2:
            st.metric("🧪 최적 pH", rec.optimal_pH)
            st.metric("⏱️ 반응 시간", rec.reaction_time)
        
        with col3:
            st.metric("📈 예상 DH", rec.dh_range)
            st.metric("🎯 FAN 수율", rec.fan_yield)
        
        st.markdown("---")
        
        # 추천 근거
        st.markdown("**📌 추천 근거**")
        for reason in rec.rationale:
            st.markdown(f"- {reason}")
        
        # 주의사항
        if rec.warnings:
            st.markdown("**⚠️ 주의사항**")
            for warn in rec.warnings:
                st.warning(warn)
        
        # 추가 정보
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**제조사:** {rec.manufacturer}")
        with col2:
            st.markdown(f"**쓴맛 수준:** {rec.bitterness}")


def get_sample_options(df: pd.DataFrame) -> list:
    """DataFrame에서 샘플 옵션 목록 생성"""
    options = []
    
    # sample_id 컬럼 찾기
    id_col = None
    for col in ['sample_id', 'Sample_id', 'SAMPLE_ID', 'ID', 'id']:
        if col in df.columns:
            id_col = col
            break
    
    # Sample_name 컬럼 찾기
    name_col = None
    for col in ['Sample_name', 'sample_name', 'SAMPLE_NAME', 'Name', 'name']:
        if col in df.columns:
            name_col = col
            break
    
    for idx, row in df.iterrows():
        # ID 추출
        if id_col and pd.notna(row[id_col]):
            sample_id = row[id_col]
            if isinstance(sample_id, float):
                sample_id = int(sample_id)
            sample_id = str(sample_id)
        else:
            sample_id = str(idx)
        
        # 이름 추출
        if name_col and pd.notna(row[name_col]):
            sample_name = str(row[name_col])
        else:
            sample_name = f"Sample {idx+1}"
        
        options.append({
            'id': sample_id,
            'name': sample_name,
            'display': f"{sample_id} - {sample_name}",
            'index': idx
        })
    
    return options


# ============================================================
# 메인 앱
# ============================================================
def main():
    # 헤더
    st.markdown('<p class="main-header">🧬 펩톤 효소 추천 시스템</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">원료의 성분 분석 데이터를 기반으로 최적의 효소를 추천합니다.</p>', 
                unsafe_allow_html=True)
    
    # 추천 엔진 로드
    try:
        recommender = load_recommender()
    except Exception as e:
        st.error(f"효소 데이터베이스 로드 실패: {e}")
        st.info("data 폴더에 enzyme_database.json 또는 enzyme_database_extended.json 파일이 있는지 확인하세요.")
        return
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        input_method = st.radio(
            "입력 방식 선택",
            ["📁 Excel 파일 업로드", "✏️ 직접 입력"]
        )
        
        st.markdown("---")
        
        st.markdown("### 📚 효소 DB 정보")
        st.markdown(f"- 등록 효소: **{len(recommender.enzymes)}종**")
        st.markdown(f"- 원료 유형: **{len(recommender.substrate_rules)}종**")
        
        st.markdown("---")
        
        st.markdown("### ℹ️ 사용 방법")
        st.markdown("""
        1. Excel 파일 업로드 또는 직접 입력
        2. 분석할 샘플 선택
        3. 추천 결과 확인
        4. 최적 반응 조건 적용
        """)
        
        st.markdown("---")
        st.markdown("### 📋 지원 컬럼명")
        st.markdown("""
        - `sample_id`: 샘플 ID
        - `Sample_name`: 샘플명
        - `raw_material`: 원료 유형
        - `general_TN`: 총질소
        - `general_AN`: 아미노태질소
        - `taa_Glutamic acid`: 글루탐산 등
        """)
    
    # 메인 컨텐츠
    if input_method == "📁 Excel 파일 업로드":
        uploaded_file = st.file_uploader(
            "성분 분석 Excel 파일 업로드 (.xlsx)",
            type=['xlsx', 'xls'],
            help="아미노산 데이터가 포함된 Excel 파일을 업로드하세요."
        )
        
        if uploaded_file:
            try:
                # Excel 파일 로드
                xlsx = pd.ExcelFile(uploaded_file)
                
                # 시트 선택
                if len(xlsx.sheet_names) > 1:
                    sheet_name = st.selectbox("시트 선택", xlsx.sheet_names, 
                                             index=xlsx.sheet_names.index('data') if 'data' in xlsx.sheet_names else 0)
                else:
                    sheet_name = xlsx.sheet_names[0]
                
                df = pd.read_excel(xlsx, sheet_name=sheet_name)
                
                # 전처리
                df_processed = recommender.preprocess_dataframe(df)
                
                if len(df_processed) == 0:
                    st.error("❌ 유효한 데이터가 없습니다. 아미노산 컬럼(taa_로 시작)이 있는지 확인하세요.")
                    
                    # 원본 데이터 표시
                    with st.expander("📊 원본 데이터 확인"):
                        st.dataframe(df)
                        st.write(f"컬럼: {list(df.columns)}")
                    return
                
                st.success(f"✅ 파일 로드 완료: {len(df_processed)}개 샘플")
                
                # 샘플 옵션 생성
                sample_options = get_sample_options(df_processed)
                
                if not sample_options:
                    st.error("샘플을 찾을 수 없습니다.")
                    return
                
                # 샘플 선택
                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_display = st.selectbox(
                        "분석할 샘플 선택",
                        options=[opt['display'] for opt in sample_options]
                    )
                    # 선택된 샘플의 인덱스 찾기
                    selected_opt = next(opt for opt in sample_options if opt['display'] == selected_display)
                    selected_index = selected_opt['index']
                
                with col2:
                    top_n = st.number_input("추천 효소 개수", min_value=1, max_value=5, value=2)
                
                if st.button("🔍 효소 추천 받기", type="primary", use_container_width=True):
                    with st.spinner("분석 중..."):
                        try:
                            # 선택된 행만 추출
                            selected_row = df_processed.iloc[[selected_index]]
                            
                            # 추천 실행
                            results = recommender.recommend(selected_row, top_n=top_n)
                            
                            if not results:
                                st.error("추천 결과를 생성할 수 없습니다.")
                                return
                            
                            # 첫 번째 결과 사용
                            result_key = list(results.keys())[0]
                            result = results[result_key]
                            analysis = result['analysis']
                            recommendations = result['recommendations']
                            
                            # 결과 표시
                            st.markdown("---")
                            
                            # 원료 분석 결과
                            st.header("📋 원료 분석 결과")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("원료명", analysis.raw_material)
                            with col2:
                                st.metric("감지된 유형", analysis.detected_type)
                            with col3:
                                st.metric("총질소 (TN)", f"{analysis.total_nitrogen:.2f}%")
                            with col4:
                                st.metric("아미노태질소 (AN)", f"{analysis.amino_nitrogen:.2f}%")
                            
                            # 특성 플래그
                            if analysis.is_collagen_like or analysis.has_cell_wall:
                                st.markdown("**특이사항:**")
                                if analysis.is_collagen_like:
                                    st.info("🔹 콜라겐/젤라틴 계열 원료로 판단됨")
                                if analysis.has_cell_wall:
                                    st.info("🔹 세포벽 함유 원료 - 전처리 권장")
                            
                            # 차트
                            col1, col2 = st.columns(2)
                            with col1:
                                fig1 = create_amino_acid_chart(analysis)
                                if fig1:
                                    st.plotly_chart(fig1, use_container_width=True)
                                else:
                                    st.info("아미노산 데이터가 부족합니다.")
                            with col2:
                                fig2 = create_group_ratio_chart(analysis)
                                if fig2:
                                    st.plotly_chart(fig2, use_container_width=True)
                                else:
                                    st.info("그룹 비율을 계산할 수 없습니다.")
                            
                            # 효소 추천 결과
                            st.markdown("---")
                            st.header("🧪 효소 추천 결과")
                            
                            for i, rec in enumerate(recommendations):
                                display_recommendation(rec, expanded=(i == 0))
                            
                            # 결과 요약 테이블
                            st.markdown("---")
                            st.subheader("📊 추천 요약")
                            
                            summary_data = []
                            for rec in recommendations:
                                summary_data.append({
                                    '순위': rec.rank,
                                    '효소명': rec.enzyme_name,
                                    '점수': rec.score,
                                    '최적온도': rec.optimal_temp,
                                    '최적pH': rec.optimal_pH,
                                    'E/S비율': rec.es_ratio,
                                    '반응시간': rec.reaction_time
                                })
                            
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True, hide_index=True)
                            
                        except Exception as e:
                            st.error(f"분석 중 오류 발생: {str(e)}")
                            with st.expander("🔍 상세 오류 정보"):
                                st.code(traceback.format_exc())
                
                # 데이터 미리보기
                with st.expander("📊 데이터 미리보기"):
                    st.dataframe(df_processed, use_container_width=True)
                    st.caption(f"전처리 후 {len(df_processed)}개 샘플 (원본: {len(df)}개)")
                    
            except Exception as e:
                st.error(f"파일 처리 오류: {str(e)}")
                with st.expander("🔍 상세 오류 정보"):
                    st.code(traceback.format_exc())
    
    else:  # 직접 입력
        st.subheader("✏️ 아미노산 데이터 직접 입력")
        
        col1, col2 = st.columns(2)
        
        with col1:
            raw_material = st.selectbox(
                "원료 유형",
                ['soy', 'wheat', 'pea', 'rice', 'fish', 'pork', 'collagen', 
                 'casein', 'yeast', 'microalgae', 'insect', 'cotton', 'malt']
            )
            total_nitrogen = st.number_input("총질소 (%)", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
        
        with col2:
            sample_name = st.text_input("샘플명", value="테스트 샘플")
            top_n = st.number_input("추천 효소 개수", min_value=1, max_value=5, value=2)
        
        st.markdown("### 아미노산 함량 입력 (g/100g)")
        
        aa_cols = st.columns(6)
        aa_list = ['Asp', 'Glu', 'Ser', 'Gly', 'Ala', 'Val', 'Leu', 'Ile', 
                   'Thr', 'Pro', 'Phe', 'Tyr', 'Trp', 'Lys', 'Arg', 'His', 'Met', 'Cys']
        
        aa_profile = {}
        for i, aa in enumerate(aa_list):
            with aa_cols[i % 6]:
                aa_profile[aa] = st.number_input(aa, min_value=0.0, max_value=30.0, value=2.0, step=0.1, key=f"aa_{aa}")
        
        if st.button("🔍 효소 추천 받기", type="primary", use_container_width=True):
            with st.spinner("분석 중..."):
                try:
                    analysis, recommendations = recommender.recommend_single(
                        aa_profile,
                        raw_material=raw_material,
                        total_nitrogen=total_nitrogen,
                        top_n=top_n
                    )
                    
                    # 결과 표시
                    st.markdown("---")
                    st.header("📋 원료 분석 결과")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("감지된 유형", analysis.detected_type)
                    with col2:
                        st.metric("소수성 AA 비율", f"{analysis.hydrophobic_ratio:.1%}")
                    with col3:
                        st.metric("염기성 AA 비율", f"{analysis.basic_ratio:.1%}")
                    
                    # 차트
                    col1, col2 = st.columns(2)
                    with col1:
                        fig1 = create_amino_acid_chart(analysis)
                        if fig1:
                            st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        fig2 = create_group_ratio_chart(analysis)
                        if fig2:
                            st.plotly_chart(fig2, use_container_width=True)
                    
                    st.markdown("---")
                    st.header("🧪 효소 추천 결과")
                    
                    for i, rec in enumerate(recommendations):
                        display_recommendation(rec, expanded=(i == 0))
                        
                except Exception as e:
                    st.error(f"분석 중 오류 발생: {str(e)}")
                    with st.expander("🔍 상세 오류 정보"):
                        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

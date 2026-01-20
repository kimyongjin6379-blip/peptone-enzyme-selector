"""
펩톤 생산용 효소 추천 엔진 (Peptone Enzyme Recommender) v2.0

원료의 성분 분석 데이터를 기반으로 최적의 효소 2종을 추천합니다.
아미노산 프로파일, 원료 유형 등을 분석하여 효소-기질 매칭 점수를 계산합니다.

v2.0 업데이트:
- 다양한 Excel 형식 지원 (유연한 컬럼 매핑)
- 빈 행/NaN 데이터 자동 필터링
- <LOQ, N.D 등 다양한 결측치 처리
- sample_id 숫자/문자열 모두 지원

Author: R&D Team
Version: 2.0
"""

import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import warnings
import re

warnings.filterwarnings('ignore')


@dataclass
class EnzymeRecommendation:
    """효소 추천 결과를 담는 데이터 클래스"""
    rank: int
    enzyme_id: str
    enzyme_name: str
    manufacturer: str
    score: float
    optimal_temp: str
    optimal_pH: str
    es_ratio: str
    reaction_time: str
    dh_range: str
    fan_yield: str
    bitterness: str
    rationale: List[str]
    warnings: List[str] = field(default_factory=list)


@dataclass
class SubstrateAnalysis:
    """원료 분석 결과를 담는 데이터 클래스"""
    sample_id: str
    sample_name: str
    raw_material: str
    detected_type: str
    total_nitrogen: float
    amino_nitrogen: float
    
    # 아미노산 그룹 비율
    hydrophobic_ratio: float
    aromatic_ratio: float
    basic_ratio: float
    acidic_ratio: float
    proline_ratio: float
    glycine_ratio: float
    hydroxyproline_ratio: float
    
    # 특성 플래그
    is_collagen_like: bool
    is_high_glutamic: bool
    is_high_basic: bool
    has_cell_wall: bool
    
    # 원본 데이터
    amino_acid_profile: Dict[str, float] = field(default_factory=dict)


class EnzymeRecommender:
    """
    펩톤 생산용 효소 추천 시스템 v2.0
    
    원료의 성분 분석 데이터를 입력받아 최적의 효소 조합을 추천합니다.
    규칙 기반 스코어링과 아미노산 프로파일 매칭을 활용합니다.
    """
    
    # 아미노산 그룹 정의
    AMINO_ACID_GROUPS = {
        'hydrophobic': ['Leu', 'Ile', 'Val', 'Phe', 'Trp', 'Met', 'Ala'],
        'aromatic': ['Phe', 'Tyr', 'Trp'],
        'basic': ['Lys', 'Arg', 'His'],
        'acidic': ['Asp', 'Glu'],
        'hydroxyl': ['Ser', 'Thr'],
        'amide': ['Asn', 'Gln'],
        'imino': ['Pro'],
        'small': ['Gly', 'Ala'],
        'collagen_marker': ['Pro', 'Gly', 'Hyp']
    }
    
    # 컬럼명 → 아미노산 코드 매핑 (유연하게 처리)
    # 다양한 컬럼명 형식을 지원
    COLUMN_PATTERNS = {
        'Asp': [r'taa_Aspartic\s*acid', r'Aspartic\s*acid', r'Asp', r'ASP'],
        'Hyp': [r'taa_Hydroxyproline', r'Hydroxyproline', r'Hyp', r'HYP'],
        'Thr': [r'taa_Threonine', r'Threonine', r'Thr', r'THR'],
        'Ser': [r'taa_Serine', r'Serine', r'Ser', r'SER'],
        'Asn': [r'taa_Asparagine', r'Asparagine', r'Asn', r'ASN'],
        'Glu': [r'taa_Glutamic\s*acid', r'Glutamic\s*acid', r'Glu', r'GLU'],
        'Gln': [r'taa_Glutamine', r'Glutamine', r'Gln', r'GLN'],
        'Cys': [r'taa_Cysteine', r'Cysteine', r'Cys', r'CYS'],
        'Pro': [r'taa_Proline', r'Proline', r'Pro', r'PRO'],
        'Gly': [r'taa_Glycine', r'Glycine', r'Gly', r'GLY'],
        'Ala': [r'taa_Alanine', r'Alanine', r'Ala', r'ALA'],
        'Val': [r'taa_Valine', r'Valine', r'Val', r'VAL'],
        'Met': [r'taa_Methionine', r'Methionine', r'Met', r'MET'],
        'Ile': [r'taa_Isoleucine', r'Isoleucine', r'Ile', r'ILE'],
        'Leu': [r'taa_Leucine', r'Leucine', r'Leu', r'LEU'],
        'Tyr': [r'taa_Tyrosine', r'Tyrosine', r'Tyr', r'TYR'],
        'Phe': [r'taa_Phenylalanine', r'Phenylalanine', r'Phe', r'PHE'],
        'His': [r'taa_Histidine', r'Histidine', r'His', r'HIS'],
        'Trp': [r'taa_Tryptophan', r'Tryptophan', r'Trp', r'TRP'],
        'Lys': [r'taa_Lysine', r'Lysine', r'Lys', r'LYS'],
        'Arg': [r'taa_Arginine', r'Arginine', r'Arg', r'ARG'],
        'Cit': [r'taa_Citruline', r'Citruline', r'Citrulline', r'Cit'],
        'Cys2': [r'taa_Cystine', r'Cystine'],
        'GABA': [r'taa_GABA', r'GABA'],
        'Orn': [r'taa_Ornithine', r'Ornithine', r'Orn']
    }
    
    def __init__(self, enzyme_db_path: str = None):
        """
        Args:
            enzyme_db_path: 효소 데이터베이스 JSON 파일 경로
        """
        if enzyme_db_path is None:
            # 기본 경로 설정
            enzyme_db_path = Path(__file__).parent.parent / 'data' / 'enzyme_database.json'
        
        with open(enzyme_db_path, 'r', encoding='utf-8') as f:
            self.db = json.load(f)
        
        self.enzymes = {e['id']: e for e in self.db['enzymes']}
        self.substrate_rules = self.db.get('substrate_type_rules', {})
        self.scoring_weights = self.db.get('scoring_weights', {
            'hydrophobic_weight': 30,
            'aromatic_weight': 25,
            'basic_weight': 20,
            'acidic_weight': 15,
            'proline_penalty_weight': 10,
            'substrate_match_bonus': 1.20,
            'collagen_specialist_bonus': 1.25,
            'cell_wall_bonus': 1.30
        })
        
        # 컬럼 매핑 캐시
        self._column_mapping_cache = {}
    
    def _clean_numeric(self, value: Any) -> float:
        """숫자가 아닌 값 (N.D, <LOQ 등)을 0으로 변환"""
        if pd.isna(value):
            return 0.0
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)
        if isinstance(value, str):
            value = value.strip()
            # 결측치 패턴
            if value.upper() in ['N.D', 'N.D.', 'ND', '<LOQ', '< LOQ', '<LOD', '< LOD', 
                                  '미량', '-', '', 'TRACE', 'TR', 'NAN', 'NULL', '`']:
                return 0.0
            # "<300" 같은 값 처리
            if value.startswith('<') or value.startswith('< '):
                try:
                    num_part = re.sub(r'[<\s]', '', value)
                    return float(num_part) * 0.5  # 검출한계의 절반으로 추정
                except:
                    return 0.0
            # ">1000" 같은 값 처리
            if value.startswith('>') or value.startswith('> '):
                try:
                    num_part = re.sub(r'[>\s]', '', value)
                    return float(num_part)
                except:
                    return 0.0
            try:
                # 쉼표 제거 후 변환 (예: "1,234.5")
                return float(value.replace(',', ''))
            except:
                return 0.0
        return 0.0
    
    def _find_column_for_aa(self, columns: List[str], aa_code: str) -> Optional[str]:
        """아미노산 코드에 해당하는 컬럼명 찾기"""
        patterns = self.COLUMN_PATTERNS.get(aa_code, [])
        
        for col in columns:
            for pattern in patterns:
                if re.search(pattern, col, re.IGNORECASE):
                    return col
        return None
    
    def _build_column_mapping(self, columns: List[str]) -> Dict[str, str]:
        """DataFrame 컬럼에서 아미노산 컬럼 매핑 생성"""
        # 캐시 키 생성
        cache_key = tuple(sorted(columns))
        if cache_key in self._column_mapping_cache:
            return self._column_mapping_cache[cache_key]
        
        mapping = {}
        for aa_code in self.COLUMN_PATTERNS.keys():
            col = self._find_column_for_aa(columns, aa_code)
            if col:
                mapping[col] = aa_code
        
        self._column_mapping_cache[cache_key] = mapping
        return mapping
    
    def _extract_amino_acid_profile(self, row: pd.Series) -> Dict[str, float]:
        """데이터 행에서 아미노산 프로파일 추출"""
        columns = list(row.index)
        col_mapping = self._build_column_mapping(columns)
        
        profile = {}
        for col, aa_code in col_mapping.items():
            if col in row.index:
                profile[aa_code] = self._clean_numeric(row[col])
        
        return profile
    
    def _calculate_group_ratio(self, profile: Dict[str, float], group: List[str]) -> float:
        """특정 아미노산 그룹의 비율 계산"""
        total = sum(v for v in profile.values() if v > 0)
        if total == 0:
            return 0.0
        group_sum = sum(profile.get(aa, 0) for aa in group)
        return group_sum / total
    
    def _detect_substrate_type(self, row: pd.Series, analysis: SubstrateAnalysis) -> str:
        """
        원료 유형 자동 감지
        
        1. raw_material 컬럼 값 확인
        2. 아미노산 패턴 기반 추정
        """
        # 1. raw_material 컬럼에서 직접 확인
        raw_mat = ''
        for col in ['raw_material', 'Raw_material', 'RAW_MATERIAL', 'material', 'Material']:
            if col in row.index and pd.notna(row[col]):
                raw_mat = str(row[col]).lower().strip()
                break
        
        if raw_mat:
            # 직접 매칭
            type_mapping = {
                'soy': 'soy', 'soya': 'soy', '대두': 'soy',
                'wheat': 'wheat', '밀': 'wheat',
                'pea': 'pea', '완두': 'pea',
                'rice': 'rice', '쌀': 'rice',
                'fish': 'fish', '어류': 'fish',
                'pork': 'pork', '돼지': 'pork',
                'casein': 'casein', '카제인': 'casein',
                'yeast': 'yeast', '효모': 'yeast',
                'collagen': 'collagen', '콜라겐': 'collagen',
                'gelatin': 'collagen', '젤라틴': 'collagen',
                'algae': 'microalgae', 'microalgae': 'microalgae', '미세조류': 'microalgae',
                'chlorella': 'microalgae', '클로렐라': 'microalgae',
                'spirulina': 'microalgae', '스피루리나': 'microalgae',
                'plant': 'plant', '식물': 'plant',
                'insect': 'insect', '곤충': 'insect', 'mealworm': 'insect', '밀웜': 'insect',
                'cotton': 'cotton', '면실': 'cotton',
                'malt': 'malt', '맥아': 'malt', '몰트': 'malt',
                'corn': 'corn', '옥수수': 'corn',
                'potato': 'potato', '감자': 'potato',
                'blood': 'blood', '혈액': 'blood'
            }
            
            for key, value in type_mapping.items():
                if key in raw_mat:
                    return value
        
        # 2. 아미노산 패턴 기반 추정
        # 콜라겐 계열: Gly + Pro + Hyp > 25%
        if analysis.is_collagen_like:
            return 'collagen'
        
        # 효모 계열: 높은 Glu, 중간 정도의 다양성
        if analysis.is_high_glutamic and analysis.acidic_ratio > 0.15:
            return 'yeast'
        
        # 동물성: 높은 Lys/Arg
        if analysis.is_high_basic and analysis.basic_ratio > 0.15:
            return 'animal'
        
        # 기본값: 식물성
        return 'plant'
    
    def _get_sample_id(self, row: pd.Series, idx: int) -> str:
        """샘플 ID 추출 (다양한 형식 지원)"""
        for col in ['sample_id', 'Sample_id', 'SAMPLE_ID', 'SampleID', 'ID', 'id']:
            if col in row.index and pd.notna(row[col]):
                val = row[col]
                # 숫자인 경우 문자열로 변환
                if isinstance(val, (int, float)):
                    if pd.isna(val):
                        continue
                    return f"Sample_{int(val)}"
                return str(val)
        return f"Sample_{idx+1}"
    
    def _get_sample_name(self, row: pd.Series) -> str:
        """샘플명 추출"""
        for col in ['Sample_name', 'sample_name', 'SAMPLE_NAME', 'SampleName', 'Name', 'name']:
            if col in row.index and pd.notna(row[col]):
                return str(row[col])
        return "Unknown"
    
    def _get_raw_material(self, row: pd.Series) -> str:
        """원료명 추출"""
        for col in ['raw_material', 'Raw_material', 'RAW_MATERIAL', 'RawMaterial', 'material', 'Material']:
            if col in row.index and pd.notna(row[col]):
                return str(row[col])
        return "Unknown"
    
    def _get_total_nitrogen(self, row: pd.Series) -> float:
        """총질소 함량 추출"""
        for col in ['general_TN', 'TN', 'Total_Nitrogen', 'total_nitrogen']:
            if col in row.index:
                return self._clean_numeric(row[col])
        return 0.0
    
    def _get_amino_nitrogen(self, row: pd.Series) -> float:
        """아미노태질소 함량 추출"""
        for col in ['general_AN', 'AN', 'Amino_Nitrogen', 'amino_nitrogen']:
            if col in row.index:
                return self._clean_numeric(row[col])
        return 0.0
    
    def analyze_substrate(self, row: pd.Series, idx: int = 0) -> SubstrateAnalysis:
        """
        원료 데이터 분석
        
        Args:
            row: 단일 샘플 데이터 (DataFrame의 한 행)
            idx: 행 인덱스 (샘플 ID 생성용)
        
        Returns:
            SubstrateAnalysis: 분석 결과 객체
        """
        # 기본 정보 추출
        sample_id = self._get_sample_id(row, idx)
        sample_name = self._get_sample_name(row)
        raw_material = self._get_raw_material(row)
        
        # 질소 함량
        total_nitrogen = self._get_total_nitrogen(row)
        amino_nitrogen = self._get_amino_nitrogen(row)
        
        # 아미노산 프로파일 추출
        aa_profile = self._extract_amino_acid_profile(row)
        
        # 총 아미노산 합계
        total_aa = sum(v for v in aa_profile.values() if v > 0)
        
        # 그룹별 비율 계산 (총합이 0이면 0으로 처리)
        if total_aa > 0:
            hydrophobic_ratio = self._calculate_group_ratio(aa_profile, self.AMINO_ACID_GROUPS['hydrophobic'])
            aromatic_ratio = self._calculate_group_ratio(aa_profile, self.AMINO_ACID_GROUPS['aromatic'])
            basic_ratio = self._calculate_group_ratio(aa_profile, self.AMINO_ACID_GROUPS['basic'])
            acidic_ratio = self._calculate_group_ratio(aa_profile, self.AMINO_ACID_GROUPS['acidic'])
            proline_ratio = aa_profile.get('Pro', 0) / total_aa
            glycine_ratio = aa_profile.get('Gly', 0) / total_aa
            hydroxyproline_ratio = aa_profile.get('Hyp', 0) / total_aa
        else:
            hydrophobic_ratio = aromatic_ratio = basic_ratio = acidic_ratio = 0.0
            proline_ratio = glycine_ratio = hydroxyproline_ratio = 0.0
        
        # 특성 플래그
        collagen_marker_ratio = proline_ratio + glycine_ratio + hydroxyproline_ratio
        is_collagen_like = collagen_marker_ratio > 0.25 or hydroxyproline_ratio > 0.05
        is_high_glutamic = (aa_profile.get('Glu', 0) / total_aa > 0.12) if total_aa > 0 else False
        is_high_basic = basic_ratio > 0.12
        
        # material_type으로 세포벽 유무 판단
        material_type = ''
        for col in ['material_type', 'Material_type', 'type']:
            if col in row.index and pd.notna(row[col]):
                material_type = str(row[col]).lower()
                break
        
        has_cell_wall = ('yeast' in material_type or 'yeast' in raw_material.lower() or
                        'algae' in raw_material.lower() or 'microalgae' in raw_material.lower())
        
        analysis = SubstrateAnalysis(
            sample_id=sample_id,
            sample_name=sample_name,
            raw_material=raw_material,
            detected_type='',  # 나중에 설정
            total_nitrogen=total_nitrogen,
            amino_nitrogen=amino_nitrogen,
            hydrophobic_ratio=hydrophobic_ratio,
            aromatic_ratio=aromatic_ratio,
            basic_ratio=basic_ratio,
            acidic_ratio=acidic_ratio,
            proline_ratio=proline_ratio,
            glycine_ratio=glycine_ratio,
            hydroxyproline_ratio=hydroxyproline_ratio,
            is_collagen_like=is_collagen_like,
            is_high_glutamic=is_high_glutamic,
            is_high_basic=is_high_basic,
            has_cell_wall=has_cell_wall,
            amino_acid_profile=aa_profile
        )
        
        # 원료 유형 감지
        analysis.detected_type = self._detect_substrate_type(row, analysis)
        
        return analysis
    
    def _calculate_enzyme_score(self, enzyme: Dict, analysis: SubstrateAnalysis) -> Tuple[float, List[str], List[str]]:
        """
        효소-기질 매칭 점수 계산
        
        Returns:
            Tuple[score, rationale_list, warning_list]
        """
        affinity = enzyme.get('specificity', {}).get('affinity_scores', {
            'hydrophobic': 0.5, 'aromatic': 0.5, 'basic': 0.5, 
            'acidic': 0.5, 'proline_penalty': 0.5
        })
        weights = self.scoring_weights
        
        rationale = []
        warnings = []
        
        # 기본 점수 계산 (가중 합산)
        score = 0
        
        # 1. 소수성 아미노산 매칭
        hydrophobic_score = analysis.hydrophobic_ratio * affinity.get('hydrophobic', 0.5) * weights.get('hydrophobic_weight', 30)
        score += hydrophobic_score
        if analysis.hydrophobic_ratio > 0.25 and affinity.get('hydrophobic', 0.5) > 0.8:
            rationale.append(f"소수성 아미노산 비율({analysis.hydrophobic_ratio:.1%})이 높아 효과적 절단 예상")
        
        # 2. 방향족 아미노산 매칭
        aromatic_score = analysis.aromatic_ratio * affinity.get('aromatic', 0.5) * weights.get('aromatic_weight', 25)
        score += aromatic_score
        if analysis.aromatic_ratio > 0.06 and affinity.get('aromatic', 0.5) > 0.75:
            rationale.append(f"방향족 아미노산({analysis.aromatic_ratio:.1%})에 대한 친화도 우수")
        
        # 3. 염기성 아미노산 매칭
        basic_score = analysis.basic_ratio * affinity.get('basic', 0.5) * weights.get('basic_weight', 20)
        score += basic_score
        if analysis.basic_ratio > 0.10 and affinity.get('basic', 0.5) > 0.7:
            rationale.append(f"염기성 아미노산(Lys, Arg, His) 비율({analysis.basic_ratio:.1%})에 적합")
        
        # 4. 산성 아미노산 매칭
        acidic_score = analysis.acidic_ratio * affinity.get('acidic', 0.5) * weights.get('acidic_weight', 15)
        score += acidic_score
        
        # 5. 프롤린 페널티 (프롤린이 많으면 일부 효소에 불리)
        proline_penalty = analysis.proline_ratio * affinity.get('proline_penalty', 0.5) * weights.get('proline_penalty_weight', 10)
        score -= proline_penalty
        if analysis.proline_ratio > 0.06 and affinity.get('proline_penalty', 0.5) > 0.6:
            warnings.append(f"프롤린 함량({analysis.proline_ratio:.1%})이 높아 가수분해 효율 저하 가능")
        
        # 보너스/페널티 적용
        substrate_type = analysis.detected_type
        
        # 6. 원료 유형 적합성 보너스
        suitable_substrates = enzyme.get('suitable_substrates', [])
        if substrate_type in suitable_substrates or 'plant' in suitable_substrates and substrate_type in ['soy', 'wheat', 'pea', 'rice', 'cotton', 'malt', 'corn']:
            score *= weights.get('substrate_match_bonus', 1.20)
            rationale.append(f"'{substrate_type}' 원료에 적합한 효소")
        
        # 7. 콜라겐 특화 효소 보너스
        if analysis.is_collagen_like:
            if enzyme['id'] in ['neutrase', 'neutrase_0.8L', 'papain', 'bromelain', 'pepsin', 'corolase_7089', 'corolase_8000']:
                score *= weights.get('collagen_specialist_bonus', 1.25)
                rationale.append("콜라겐/젤라틴 분해에 특화된 효소")
            else:
                score *= 0.75
                warnings.append("콜라겐 계열 원료에는 최적이 아닐 수 있음")
        
        # 8. 세포벽 원료 (효모, 미세조류) 처리
        if analysis.has_cell_wall:
            if enzyme['id'] in ['pronase', 'pronase_e', 'viscozyme_cellulase', 'cellulase_protease']:
                score *= weights.get('cell_wall_bonus', 1.30)
                rationale.append("세포벽 분해 또는 세포벽 단백질 처리에 효과적")
            else:
                warnings.append("세포벽 파쇄 전처리 권장")
        
        # 9. 곤충(insect) 원료 특별 처리
        if substrate_type == 'insect':
            if enzyme['id'] in ['alcalase', 'alcalase_2.4L', 'flavourzyme', 'flavourzyme_1000L', 'pronase', 'pronase_e']:
                score *= 1.15
                rationale.append("곤충 단백질 가수분해에 효과적")
        
        # 점수 정규화 (0-100)
        score = min(100, max(0, score))
        
        # 근거가 없으면 일반적 설명 추가
        if not rationale:
            rationale.append("일반적인 아미노산 조성에 대한 적합성 기반 추천")
        
        return score, rationale, warnings
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame 전처리
        - NaN 행 제거
        - 빈 sample_id 행 제거
        """
        # 원본 복사
        df = df.copy()
        
        # sample_id 또는 Sample_name이 모두 NaN인 행 제거
        id_cols = [col for col in df.columns if col.lower() in ['sample_id', 'sample_name', 'id', 'name']]
        if id_cols:
            # 모든 ID 컬럼이 NaN인 행 제거
            mask = df[id_cols].notna().any(axis=1)
            df = df[mask]
        
        # 아미노산 컬럼이 모두 NaN/0인 행 제거
        aa_cols = [col for col in df.columns if col.startswith('taa_') or col in self.COLUMN_PATTERNS.keys()]
        if aa_cols:
            # 적어도 하나의 아미노산 값이 있는 행만 유지
            def has_valid_aa(row):
                for col in aa_cols:
                    if col in row.index:
                        val = self._clean_numeric(row[col])
                        if val > 0:
                            return True
                return False
            
            mask = df.apply(has_valid_aa, axis=1)
            df = df[mask]
        
        # 인덱스 리셋
        df = df.reset_index(drop=True)
        
        return df
    
    def recommend(
        self, 
        data: pd.DataFrame, 
        sample_id: str = None,
        top_n: int = 2
    ) -> Dict[str, Dict]:
        """
        효소 추천 실행
        
        Args:
            data: 성분 분석 데이터 DataFrame
            sample_id: 특정 샘플만 분석 (None이면 전체)
            top_n: 추천할 효소 개수
        
        Returns:
            Dict[sample_id, {'analysis': SubstrateAnalysis, 'recommendations': List[EnzymeRecommendation]}]
        """
        # 전처리
        data = self.preprocess_dataframe(data)
        
        if len(data) == 0:
            raise ValueError("유효한 데이터가 없습니다. 아미노산 데이터를 확인해주세요.")
        
        results = {}
        
        for idx, row in data.iterrows():
            # 현재 행의 샘플 ID
            current_sid = self._get_sample_id(row, idx)
            
            # 특정 샘플만 분석하는 경우
            if sample_id is not None:
                # sample_id 비교 (숫자/문자열 모두 처리)
                target_ids = [sample_id, f"Sample_{sample_id}", str(sample_id)]
                if current_sid not in target_ids and str(idx) != str(sample_id):
                    continue
            
            # 1. 원료 분석
            analysis = self.analyze_substrate(row, idx)
            
            # 2. 모든 효소에 대해 스코어 계산
            enzyme_scores = []
            for enzyme_id, enzyme in self.enzymes.items():
                score, rationale, warnings = self._calculate_enzyme_score(enzyme, analysis)
                enzyme_scores.append({
                    'enzyme_id': enzyme_id,
                    'enzyme': enzyme,
                    'score': score,
                    'rationale': rationale,
                    'warnings': warnings
                })
            
            # 3. 점수순 정렬
            enzyme_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # 4. 상위 N개 추천 생성
            recommendations = []
            for rank, item in enumerate(enzyme_scores[:top_n], 1):
                enzyme = item['enzyme']
                opt = enzyme.get('optimal_conditions', {})
                char = enzyme.get('characteristics', {})
                
                # 안전하게 값 추출
                temp = opt.get('temperature', {'min': 50, 'max': 60, 'unit': '°C'})
                ph = opt.get('pH', {'min': 6.0, 'max': 8.0})
                es = opt.get('ES_ratio', {'min': 0.5, 'max': 2.0, 'unit': '%'})
                time = opt.get('reaction_time', {'min': 2, 'max': 6, 'unit': 'hours'})
                
                rec = EnzymeRecommendation(
                    rank=rank,
                    enzyme_id=item['enzyme_id'],
                    enzyme_name=enzyme.get('name', item['enzyme_id']),
                    manufacturer=enzyme.get('manufacturer', 'Unknown'),
                    score=round(item['score'], 1),
                    optimal_temp=f"{temp.get('min', 50)}-{temp.get('max', 60)}{temp.get('unit', '°C')}",
                    optimal_pH=f"{ph.get('min', 6.0)}-{ph.get('max', 8.0)}",
                    es_ratio=f"{es.get('min', 0.5)}-{es.get('max', 2.0)}{es.get('unit', '%')}",
                    reaction_time=f"{time.get('min', 2)}-{time.get('max', 6)} {time.get('unit', 'hours')}",
                    dh_range=char.get('DH_range', 'N/A'),
                    fan_yield=char.get('FAN_yield', 'N/A'),
                    bitterness=char.get('bitterness', 'N/A'),
                    rationale=item['rationale'],
                    warnings=item['warnings']
                )
                recommendations.append(rec)
            
            results[current_sid] = {
                'analysis': analysis,
                'recommendations': recommendations
            }
        
        return results
    
    def recommend_single(
        self, 
        amino_acid_profile: Dict[str, float],
        raw_material: str = 'unknown',
        total_nitrogen: float = 10.0,
        top_n: int = 2
    ) -> Tuple[SubstrateAnalysis, List[EnzymeRecommendation]]:
        """
        단일 샘플에 대한 간편 추천
        
        Args:
            amino_acid_profile: {아미노산코드: 함량} 딕셔너리
            raw_material: 원료명
            total_nitrogen: 총질소 함량 (%)
            top_n: 추천 개수
        
        Returns:
            Tuple[분석결과, 추천목록]
        """
        # DataFrame 형식으로 변환
        row_data = {
            'sample_id': 'single', 
            'Sample_name': raw_material, 
            'raw_material': raw_material, 
            'general_TN': total_nitrogen
        }
        
        # 아미노산 데이터 추가
        for aa, value in amino_acid_profile.items():
            col_name = f'taa_{aa}' if not aa.startswith('taa_') else aa
            row_data[col_name] = value
        
        df = pd.DataFrame([row_data])
        results = self.recommend(df, top_n=top_n)
        
        if 'single' in results:
            result = results['single']
            return result['analysis'], result['recommendations']
        elif results:
            # 첫 번째 결과 반환
            first_key = list(results.keys())[0]
            result = results[first_key]
            return result['analysis'], result['recommendations']
        else:
            raise ValueError("추천 결과를 생성할 수 없습니다.")


def load_composition_data(file_path: str, sheet_name: str = None) -> pd.DataFrame:
    """
    성분 분석 Excel 파일 로드
    
    Args:
        file_path: Excel 파일 경로
        sheet_name: 시트명 (None이면 자동 감지)
    
    Returns:
        DataFrame
    """
    xlsx = pd.ExcelFile(file_path)
    
    if sheet_name is None:
        # 'data' 시트 우선, 없으면 첫 번째 시트
        if 'data' in xlsx.sheet_names:
            sheet_name = 'data'
        else:
            sheet_name = xlsx.sheet_names[0]
    
    df = pd.read_excel(xlsx, sheet_name=sheet_name)
    return df


def print_recommendation_report(
    analysis: SubstrateAnalysis, 
    recommendations: List[EnzymeRecommendation]
) -> None:
    """추천 결과를 보기 좋게 출력"""
    
    print("=" * 70)
    print(f"📋 원료 분석 결과: {analysis.sample_name}")
    print("=" * 70)
    print(f"  • Sample ID: {analysis.sample_id}")
    print(f"  • 원료: {analysis.raw_material}")
    print(f"  • 감지된 유형: {analysis.detected_type}")
    print(f"  • 총질소(TN): {analysis.total_nitrogen:.2f}%")
    print(f"  • 아미노태질소(AN): {analysis.amino_nitrogen:.2f}%")
    print()
    print("  [아미노산 그룹 비율]")
    print(f"    - 소수성: {analysis.hydrophobic_ratio:.1%}")
    print(f"    - 방향족: {analysis.aromatic_ratio:.1%}")
    print(f"    - 염기성: {analysis.basic_ratio:.1%}")
    print(f"    - 산성: {analysis.acidic_ratio:.1%}")
    print(f"    - 프롤린: {analysis.proline_ratio:.1%}")
    print(f"    - 글리신: {analysis.glycine_ratio:.1%}")
    print()
    
    if analysis.is_collagen_like:
        print("  ⚠️ 콜라겐/젤라틴 계열로 판단됨")
    if analysis.has_cell_wall:
        print("  ⚠️ 세포벽 함유 원료 (전처리 권장)")
    
    print()
    print("=" * 70)
    print("🧪 효소 추천 결과")
    print("=" * 70)
    
    for rec in recommendations:
        print()
        print(f"  #{rec.rank} {rec.enzyme_name} (점수: {rec.score}점)")
        print(f"  " + "-" * 50)
        print(f"  제조사: {rec.manufacturer}")
        print(f"  최적 온도: {rec.optimal_temp}")
        print(f"  최적 pH: {rec.optimal_pH}")
        print(f"  E/S 비율: {rec.es_ratio}")
        print(f"  반응 시간: {rec.reaction_time}")
        print(f"  예상 DH: {rec.dh_range}")
        print(f"  FAN 수율: {rec.fan_yield}")
        print(f"  쓴맛 수준: {rec.bitterness}")
        print()
        print("  📌 추천 근거:")
        for reason in rec.rationale:
            print(f"    • {reason}")
        
        if rec.warnings:
            print()
            print("  ⚠️ 주의사항:")
            for warn in rec.warnings:
                print(f"    • {warn}")
    
    print()
    print("=" * 70)


# CLI 실행
if __name__ == "__main__":
    import sys
    
    # 기본 테스트
    print("펩톤 효소 추천 시스템 v2.0")
    print()
    
    # 테스트용 아미노산 프로파일 (대두 펩톤 유사)
    test_profile = {
        'Asp': 7.2, 'Thr': 2.7, 'Ser': 3.3, 'Glu': 12.1, 'Pro': 3.2,
        'Gly': 2.5, 'Ala': 2.5, 'Val': 2.6, 'Met': 0.2, 'Ile': 2.6,
        'Leu': 4.2, 'Tyr': 1.9, 'Phe': 2.8, 'His': 2.1, 'Lys': 4.6,
        'Arg': 5.7
    }
    
    # 추천 실행
    db_path = Path(__file__).parent.parent / 'data' / 'enzyme_database.json'
    
    if db_path.exists():
        recommender = EnzymeRecommender(str(db_path))
        analysis, recommendations = recommender.recommend_single(
            test_profile, 
            raw_material='soy',
            total_nitrogen=9.9
        )
        print_recommendation_report(analysis, recommendations)
    else:
        print(f"효소 DB 파일을 찾을 수 없습니다: {db_path}")

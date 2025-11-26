"""
LLM explanation module.

This module generates natural language explanations for detected anomalies
using an LLM service (Gemini API).
"""

from typing import Optional, Dict
import pandas as pd
import logging
import json

from .config import LLMConfig, DetectionConfig


logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """Generates explanations for anomalies using LLM."""
    
    def __init__(self, config: LLMConfig, detection_config: Optional[DetectionConfig] = None):
        """
        Initialize ExplanationGenerator.
        
        Args:
            config: LLM configuration object.
            detection_config: Optional detection configuration for threshold information.
        """
        self.config = config
        self.detection_config = detection_config or DetectionConfig()
        
    def _create_context_data(
        self,
        anomaly_date: pd.Timestamp,
        anomaly_data: pd.DataFrame
    ) -> Dict:
        """
        Create context data for LLM explanation.
        
        Args:
            anomaly_date: Date of the anomaly.
            anomaly_data: DataFrame row(s) containing anomaly information.
            
        Returns:
            Dict: Context data for LLM.
        """
        date_str = pd.to_datetime(anomaly_date).strftime('%Y-%m-%d')
        
        # Extract first row if multiple rows
        data_row = anomaly_data.iloc[0] if len(anomaly_data) > 1 else anomaly_data.iloc[0]
        
        context = {
            'Anomaly_Date': date_str,
            'Anomaly_Score': float(data_row['anomaly_score']),
            'TotalCost': float(data_row['TotalCost']),
            'Decrease_Rate': float(data_row['pct_change']),
            'LSTM_MSE': float(data_row['mse']),
            'Decrease_Threshold': self.detection_config.decrease_threshold * 100
        }
        
        return context
    
    def _create_prompt(self, context_data: Dict) -> str:
        """
        Create prompt for LLM.
        
        Args:
            context_data: Context data dictionary.
            
        Returns:
            str: Formatted prompt for LLM.
        """
        system_prompt = (
            "당신은 클라우드 비용 이상 탐지 시스템의 전문 분석가입니다. "
            "아래 제공된 탐지 결과의 통계적 수치를 기반으로, "
            "이 이상 현상이 '실무적으로 얼마나 심각한 감소'인지와 "
            "'패턴 파괴의 정도'를 종합하여 70자 이내의 한국어 단일 문장으로 요약 설명해주세요."
        )
        
        user_query = f"""
클라우드 비용 이상 탐지 결과:
- 탐지 날짜: {context_data['Anomaly_Date']}
- 최종 이상 점수 (1.0 = Strong Anomaly): {context_data['Anomaly_Score']}
- 전일 대비 비용 변화율: {context_data['Decrease_Rate']:.2f}%
- 실무 기준 감소 임계값: {context_data.get('Decrease_Threshold', -30):.2f}% (이 값보다 감소했음)
- LSTM 재구성 오차 (MSE): {context_data['LSTM_MSE']:.6f} (패턴 파괴 정도)

이 이상치에 대한 해석을 요청합니다.
"""
        
        full_prompt = f"{system_prompt}\n\n{user_query}"
        return full_prompt
    
    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """
        Call LLM API to generate explanation.
        
        Args:
            prompt: Prompt text for LLM.
            
        Returns:
            Optional[str]: Generated explanation, or None if API call fails.
        """
        if not self.config.api_key or self.config.api_key == '':
            logger.warning("API key not provided. Skipping LLM API call.")
            return None
        
        try:
            import requests
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }
            
            response = requests.post(
                self.config.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            response.raise_for_status()
            
            result = response.json()
            explanation = result['candidates'][0]['content']['parts'][0]['text']
            
            logger.info("Successfully generated LLM explanation")
            return explanation.strip()
            
        except Exception as e:
            logger.error(f"Failed to call LLM API: {e}")
            return None
    
    def _generate_fallback_explanation(
        self,
        context_data: Dict
    ) -> str:
        """
        Generate a fallback explanation when LLM API is unavailable.
        
        Args:
            context_data: Context data dictionary.
            
        Returns:
            str: Fallback explanation.
        """
        if context_data['Anomaly_Score'] == 1.0:
            return (
                f"패턴 이탈 및 비용 {context_data['Decrease_Rate']:.0f}%의 "
                f"심각한 급감 감지. 즉각적인 원인 분석 및 조치 필요."
            )
        elif context_data['Anomaly_Score'] >= 0.5:
            return (
                f"비용 {context_data['Decrease_Rate']:.0f}% 감소 또는 "
                f"패턴 이상 감지. 모니터링 필요."
            )
        else:
            return "이상치가 감지되지 않았습니다."
    
    def generate_explanation(
        self,
        anomaly_date: pd.Timestamp,
        anomaly_data: pd.DataFrame,
        use_api: bool = False
    ) -> str:
        """
        Generate explanation for a detected anomaly.
        
        Args:
            anomaly_date: Date of the anomaly.
            anomaly_data: DataFrame containing anomaly information.
            use_api: Whether to use LLM API (True) or fallback explanation (False).
            
        Returns:
            str: Generated explanation text.
        """
        context_data = self._create_context_data(anomaly_date, anomaly_data)
        
        if use_api and self.config.api_key:
            prompt = self._create_prompt(context_data)
            explanation = self._call_llm_api(prompt)
            
            if explanation:
                # Truncate to max length if needed
                if len(explanation) > self.config.max_explanation_length:
                    explanation = explanation[:self.config.max_explanation_length - 3] + "..."
                return explanation
        
        # Use fallback explanation
        explanation = self._generate_fallback_explanation(context_data)
        
        return explanation
    
    def generate_batch_explanations(
        self,
        anomaly_df: pd.DataFrame,
        use_api: bool = False
    ) -> pd.DataFrame:
        """
        Generate explanations for all anomalies in a DataFrame.
        
        Args:
            anomaly_df: DataFrame containing anomalies.
            use_api: Whether to use LLM API for explanations.
            
        Returns:
            pd.DataFrame: Input DataFrame with added 'explanation' column.
        """
        result_df = anomaly_df.copy()
        explanations = []
        
        for anomaly_date in anomaly_df.index:
            anomaly_data = anomaly_df.loc[[anomaly_date]]
            explanation = self.generate_explanation(
                anomaly_date,
                anomaly_data,
                use_api=use_api
            )
            explanations.append(explanation)
        
        result_df['explanation'] = explanations
        
        logger.info(f"Generated {len(explanations)} explanations")
        
        return result_df

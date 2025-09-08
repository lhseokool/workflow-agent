"""
Workflow Agent Performance Test

test_data.json을 사용하여 다양한 모델의 성능을 측정하고 비교
- Baseline Model
- LangGraph Model (conditional edge)
- 3-Stage Model
"""

import json
import time
import os
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

from models.baseline_model import BaselineWorkflowAgent


def load_test_data() -> List[Dict[str, Any]]:
    """테스트 데이터 로드"""
    try:
        with open("data/test_data.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ data/test_data.json not found")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in test_data.json: {e}")
        raise


def remove_flow_name(data: Any) -> Any:
    """flow_name 필드 제거 (비교용)"""
    if isinstance(data, dict):
        return {k: remove_flow_name(v) for k, v in data.items() if k != 'flow_name'}
    elif isinstance(data, list):
        return [remove_flow_name(item) for item in data]
    return data


def exact_match_eval(generated: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """Exact match 평가 (flow_name 제외)"""
    return remove_flow_name(generated) == remove_flow_name(expected)


def save_to_excel(results_data: List[Dict], summary_stats: Dict, model_type: str = "baseline", output_dir: str = "results") -> str:
    """결과를 엑셀 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{model_type}_test_results_{timestamp}.xlsx")

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 상세 결과 시트
        detailed_df = pd.DataFrame(results_data)
        detailed_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        
        # 요약 통계 시트
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 타입별 통계 시트
        type_stats_data = []
        for case_type in ['LLM', 'Sequential', 'Loop', 'Parallel']:
            type_cases = [r for r in results_data if r['Type'] == case_type]
            if type_cases:
                exact_matches = sum(1 for r in type_cases if r['Exact_Match'] == 'O')
                type_stats_data.append({
                    'Workflow_Type': case_type,
                    'Total_Cases': len(type_cases),
                    'Exact_Match_Count': exact_matches,
                    'Exact_Match_Rate': round(exact_matches / len(type_cases) * 100, 1),
                    'Avg_Time': round(sum(r['Execution_Time'] for r in type_cases) / len(type_cases), 2)
                })
        
        if type_stats_data:
            type_df = pd.DataFrame(type_stats_data)
            type_df.to_excel(writer, sheet_name='Type Analysis', index=False)
    
    return filename


def main() -> None:
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Workflow Agent Performance Tester")
    parser.add_argument("--model", choices=["baseline", "langgraph", "3stage"], 
                       default="baseline", help="Model to test")
    parser.add_argument("--outdir", default="results", 
                       help="Directory to save Excel results")
    args = parser.parse_args()

    model_type = args.model
    print(f"🚀 {model_type.capitalize()} Model Test")
    print("=" * 50)

    # 모델 초기화
    try:
        if model_type == "baseline":
            agent = BaselineWorkflowAgent()
        elif model_type == "langgraph":
            from models.langgraph_model_1 import LangGraphRetryAgent
            agent = LangGraphRetryAgent()
        elif model_type == "3stage":
            from models.langgraph_model_2 import ThreeStageWorkflowAgent
            agent = ThreeStageWorkflowAgent()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except ImportError as e:
        print(f"❌ Failed to import model: {e}")
        return

    # 테스트 데이터 로드
    test_data = load_test_data()
    
    # 결과 추적 변수
    exact_matches = 0
    total_time = 0
    results_data = []
    
    print(f"📋 Testing {len(test_data)} cases...")
    
    # 각 테스트 케이스 실행
    for case in test_data:
        print(f"\n📝 Case {case['id']}: {case['type']}")
        
        start_time = time.time()
        
        # 워크플로우 생성
        result = agent.generate_workflow(case['input'])
        generated = result['label_json']
        expected = case['expected_output']
        
        exec_time = time.time() - start_time
        total_time += exec_time
        
        # 정확도 평가
        exact_match = exact_match_eval(generated, expected)
        if exact_match:
            exact_matches += 1
        
        # 결과 데이터 저장
        results_data.append({
            'Case_ID': case['id'],
            'Type': case['type'],
            'Instruction': case['input'][:100] + '...' if len(case['input']) > 100 else case['input'],
            'Exact_Match': 'O' if exact_match else 'X',
            'Execution_Time': round(exec_time, 2),
            'Generated_JSON': json.dumps(generated, ensure_ascii=False),
            'Expected_JSON': json.dumps(expected, ensure_ascii=False),
            'Generated_Clean': json.dumps(remove_flow_name(generated), ensure_ascii=False),
            'Expected_Clean': json.dumps(remove_flow_name(expected), ensure_ascii=False)
        })
        
        # 진행상황 출력
        print(f"  Exact Match: {'✅' if exact_match else '❌'}")
        print(f"  Time: {exec_time:.2f}s")
    
    # 최종 결과 계산
    total_cases = len(test_data)
    exact_match_rate = exact_matches / total_cases * 100
    avg_time = total_time / total_cases
    
    print(f"\n📊 Final Results:")
    print(f"  Exact Match: {exact_matches}/{total_cases} ({exact_match_rate:.1f}%)")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Average Time: {avg_time:.2f}s")
    
    # 요약 통계 생성
    summary_stats = {
        'Test_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Model_Type': model_type,
        'Total_Cases': total_cases,
        'Exact_Match_Count': exact_matches,
        'Exact_Match_Rate': round(exact_match_rate, 1),
        'Total_Time': round(total_time, 2),
        'Average_Time': round(avg_time, 2)
    }
    
    # 결과 저장
    try:
        filename = save_to_excel(results_data, summary_stats, 
                               model_type=model_type, output_dir=args.outdir)
        print(f"\n💾 Results saved to Excel: {filename}")
        
    except Exception as e:
        print(f"⚠️ Failed to save Excel file: {e}")
        print("📄 Saving as JSON instead...")
        
        # JSON 백업 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(args.outdir, exist_ok=True)
        json_filename = os.path.join(args.outdir, f"{model_type}_test_results_{timestamp}.json")
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary_stats,
                'detailed_results': results_data
            }, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Backup saved as: {json_filename}")


if __name__ == "__main__":
    main()

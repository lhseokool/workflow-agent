"""
Baseline Model Performance Test
test_data.json을 사용하여 baseline model 성능 측정 (단순 버전)
"""

import json
import time
import copy
import os
import argparse
import pandas as pd
from datetime import datetime
from models.baseline_model import BaselineWorkflowAgent


def load_test_data():
    """테스트 데이터 로드"""
    with open("data/test_data.json", 'r', encoding='utf-8') as f:
        return json.load(f)


def remove_flow_name(data):
    """flow_name 제거"""
    if isinstance(data, dict):
        return {k: remove_flow_name(v) for k, v in data.items() if k != 'flow_name'}
    elif isinstance(data, list):
        return [remove_flow_name(item) for item in data]
    return data


def exact_match_eval(generated, expected):
    """Exact match 평가 (flow_name 제외)"""
    return remove_flow_name(generated) == remove_flow_name(expected)


def save_to_excel(results_data, summary_stats, model_type: str = "baseline", output_dir: str = "results"):
    """결과를 엑셀 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{model_type}_test_results_{timestamp}.xlsx")

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 1) 상세 결과 시트
        detailed_df = pd.DataFrame(results_data)
        detailed_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        
        # 2) 요약 통계 시트
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 3) 타입별 통계 시트
        type_stats_data = []
        for case_type in ['LLM', 'Sequential', 'Loop', 'Parallel']:
            type_cases = [r for r in results_data if r['Type'] == case_type]
            if type_cases:
                type_stats_data.append({
                    'Workflow_Type': case_type,
                    'Total_Cases': len(type_cases),
                    'Exact_Match_Count': sum(1 for r in type_cases if r['Exact_Match'] == 'O'),
                    'Exact_Match_Rate': sum(1 for r in type_cases if r['Exact_Match'] == 'O') / len(type_cases) * 100,
                    'Avg_Time': sum(r['Execution_Time'] for r in type_cases) / len(type_cases)
                })
        
        if type_stats_data:
            type_df = pd.DataFrame(type_stats_data)
            type_df.to_excel(writer, sheet_name='Type Analysis', index=False)
    
    return filename


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Workflow Agent Tester")
    parser.add_argument("--model", choices=["baseline", "langgraph"], default="baseline", help="Model to test")
    parser.add_argument("--outdir", default="results", help="Directory to save Excel results")
    args = parser.parse_args()

    model_type = args.model
    print(f"🚀 {model_type.capitalize()} Model Test")
    print("="*50)

    # 모델 초기화
    if model_type == "baseline":
        agent = BaselineWorkflowAgent()
    else:
        # Lazy import to avoid unnecessary deps if not used
        from models.langgraph_model import LangGraphRetryAgent
        agent = LangGraphRetryAgent()

    test_data = load_test_data()
    
    exact_matches = 0
    total_time = 0
    results_data = []  # 엑셀 저장용 데이터
    
    for i, case in enumerate(test_data):
        print(f"\n📝 Case {case['id']}: {case['type']}")
        
        start_time = time.time()
        
        # JSON 생성
        result = agent.generate_workflow(case['input'])
        generated = result['label_json']
        expected = case['expected_output']
        
        exec_time = time.time() - start_time
        total_time += exec_time
        
        # 평가 (Exact Match만)
        exact_match = exact_match_eval(generated, expected)
        
        if exact_match:
            exact_matches += 1
        
        # 엑셀용 데이터 저장 (정답값도 포함)
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
        
        print(f"  Exact Match: {'✅' if exact_match else '❌'}")
        print(f"  Time: {exec_time:.2f}s")
    
    # 결과 출력
    total_cases = len(test_data)
    exact_match_rate = exact_matches/total_cases*100
    avg_time = total_time/total_cases
    
    print(f"\n📊 Results:")
    print(f"  Exact Match: {exact_matches}/{total_cases} ({exact_match_rate:.1f}%)")
    print(f"  Avg Time: {avg_time:.2f}s")
    
    # 요약 통계
    summary_stats = {
        'Test_Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Model_Type': model_type,
        'Total_Cases': total_cases,
        'Exact_Match_Count': exact_matches,
        'Exact_Match_Rate': round(exact_match_rate, 1),
        'Total_Time': round(total_time, 2),
        'Average_Time': round(avg_time, 2)
    }
    
    # 엑셀 저장
    try:
        filename = save_to_excel(results_data, summary_stats, model_type=model_type, output_dir=args.outdir)
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

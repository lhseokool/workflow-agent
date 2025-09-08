"""
JSON Parsing Performance Test

다양한 JSON 파서의 성능을 비교하는 테스트:
- StrOutputParser (기본)
- JsonOutputParser  
- PydanticOutputParser
- with_structured_output
"""

import json
import time
import os
import sys
import argparse
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

# 환경 변수 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 필수 의존성 확인 및 import
def check_dependencies() -> None:
    """필수 의존성 확인"""
    try:
        import pandas as pd
        from pydantic import BaseModel, Field
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser
        from langchain_openai import ChatOpenAI
        from prompts import JSON_PROMPT
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install required packages:")
        print("   pip install pandas pydantic langchain langchain-openai langchain-core")
        sys.exit(1)

# 의존성 확인 실행
check_dependencies()

# 이제 안전하게 import
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from prompts import JSON_PROMPT


# ==== 모델 정의 ====
class Flow(BaseModel):
    """
    단일 공개 모델.
    - type == 'LLM'       → tools 필수, sub_agents 금지
    - type in others      → sub_agents 필수, tools 금지
    - sub_agents 원소는 {"agent_name": str} 또는 {"flow": Flow} (재귀)
    - 모든 dict는 예상 키만 허용
    """
    flow_name: str = Field(description="플로우 이름")
    type: Literal["LLM", "Sequential", "Loop", "Parallel"]
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="LLM 전용: [{'agent_name': str}, ...]"
    )
    sub_agents: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Sequential/Loop/Parallel 전용: [{'agent_name': str} | {'flow': Flow}, ...]"
    )


def load_test_data() -> List[Dict[str, Any]]:
    """테스트 데이터 로드"""
    try:
        with open("data/test_data.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ data/test_data.json not found. Make sure you're in the correct directory.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in test_data.json: {e}")
        sys.exit(1)


def clean_for_comparison(data: Any, remove_flow_name: bool = True, remove_none: bool = True) -> Any:
    """비교를 위해 데이터 정리"""
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            # flow_name 제거 옵션
            if remove_flow_name and k == 'flow_name':
                continue
            # None 값 제거 옵션
            if remove_none and v is None:
                continue
            result[k] = clean_for_comparison(v, remove_flow_name, remove_none)
        return result
    elif isinstance(data, list):
        return [clean_for_comparison(item, remove_flow_name, remove_none) 
                for item in data if not (remove_none and item is None)]
    return data


def exact_match_eval(generated: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """Exact match 평가 (flow_name과 None 값 제외)"""
    return clean_for_comparison(generated) == clean_for_comparison(expected)


def validate_api_key() -> None:
    """API 키 유효성 확인"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("💡 Set it with: export OPENAI_API_KEY='your-key-here'")
        print("💡 Or create .env file with: OPENAI_API_KEY=your-key-here")
        sys.exit(1)


def setup_parsers() -> Dict[str, Any]:
    """모든 파서와 체인 설정"""
    validate_api_key()
    
    # LLM 초기화
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        print("💡 Check your API key and internet connection")
        sys.exit(1)
    
    parsers = {}
    
    try:
        # 1. StrOutputParser (기본)
        parser_str = StrOutputParser()
        prompt_str = ChatPromptTemplate.from_template(JSON_PROMPT)
        parsers["StrOutputParser"] = prompt_str | llm | parser_str
        
        # 2. JsonOutputParser
        parser_json = JsonOutputParser(pydantic_object=Flow)
        prompt_json = ChatPromptTemplate.from_template(JSON_PROMPT).partial(
            format_instructions=parser_json.get_format_instructions()
        )
        parsers["JsonOutputParser"] = prompt_json | llm | parser_json
        
        # 3. PydanticOutputParser
        parser_pydantic = PydanticOutputParser(pydantic_object=Flow)
        prompt_pydantic = ChatPromptTemplate.from_template(JSON_PROMPT).partial(
            format_instructions=parser_pydantic.get_format_instructions()
        )
        parsers["PydanticOutputParser"] = prompt_pydantic | llm | parser_pydantic
        
        # 4. with_structured_output
        llm_structured = llm.with_structured_output(Flow)
        prompt_structured = ChatPromptTemplate.from_template(JSON_PROMPT)
        parsers["StructuredOutput"] = prompt_structured | llm_structured
        
    except Exception as e:
        print(f"❌ Failed to setup parsers: {e}")
        sys.exit(1)
    
    return parsers


def process_chain_result(result: Any, start_time: float) -> Tuple[Any, float, Optional[str]]:
    """체인 결과 처리"""
    if isinstance(result, BaseModel):
        return result.dict(), time.time() - start_time, None
    elif isinstance(result, str):
        # StrOutputParser의 경우 JSON 문자열을 파싱
        try:
            return json.loads(result), time.time() - start_time, None
        except json.JSONDecodeError as e:
            return result, time.time() - start_time, f"JSON parsing failed: {e}"
    else:
        return result, time.time() - start_time, None


def print_verbose_comparison(expected: Dict, generated: Dict, exact_match: bool) -> None:
    """상세 비교 결과 출력"""
    print(f"Expected: {expected}")
    print(f"Generated: {generated}")
    print(f"Exact match: {exact_match}")
    expected_clean = clean_for_comparison(expected)
    generated_clean = clean_for_comparison(generated)
    print(f"Expected (clean): {expected_clean}")
    print(f"Generated (clean): {generated_clean}")
    print(f"Clean match: {expected_clean == generated_clean}")
    print("----")


def test_single_case(chain: Any, instruction: str, expected_output: Dict[str, Any], 
                    parser_name: str, verbose: bool = False) -> Dict[str, Any]:
    """단일 케이스 테스트"""
    start_time = time.time()
    
    try:
        # 체인 실행
        raw_result = chain.invoke({"instruction": instruction})
        
        # 결과 처리
        result, execution_time, error = process_chain_result(raw_result, start_time)
        
        if error:
            return {
                "success": False,
                "error": error,
                "execution_time": execution_time,
                "generated": result,
                "exact_match": False
            }
        
        # 정확도 평가
        exact_match = exact_match_eval(result, expected_output)
        
        # 상세 출력 (옵션)
        if verbose:
            print_verbose_comparison(expected_output, result, exact_match)
        
        return {
            "success": True,
            "error": None,
            "execution_time": execution_time,
            "generated": result,
            "exact_match": exact_match
        }
        
    except Exception as e:
        if verbose:
            print(f"Exception occurred: {e}")
        return {
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time,
            "generated": None,
            "exact_match": False
        }


def initialize_parser_stats() -> Dict[str, Any]:
    """파서 통계 초기화"""
    return {
        "total_cases": 0,
        "successful_cases": 0,
        "exact_matches": 0,
        "total_time": 0,
        "avg_time": 0,
        "success_rate": 0,
        "exact_match_rate": 0,
        "errors": []
    }


def run_case_multiple_times(chain: Any, case: Dict[str, Any], parser_name: str, 
                          n_runs: int, verbose: bool) -> Tuple[List[Dict], int, int, float]:
    """케이스를 여러 번 실행하고 결과 반환"""
    case_results = []
    for run in range(n_runs):
        result = test_single_case(chain, case['input'], case['expected_output'], parser_name, verbose)
        case_results.append(result)
    
    successful_runs = sum(1 for r in case_results if r["success"])
    exact_match_runs = sum(1 for r in case_results if r["exact_match"])
    avg_time = sum(r["execution_time"] for r in case_results) / len(case_results)
    
    return case_results, successful_runs, exact_match_runs, avg_time


def create_detailed_result(parser_name: str, case: Dict[str, Any], successful_runs: int, 
                         exact_match_runs: int, avg_time: float, best_result: Dict, n_runs: int) -> Dict[str, Any]:
    """상세 결과 생성"""
    return {
        "Parser": parser_name,
        "Case_ID": case['id'],
        "Type": case['type'],
        "Instruction": case['input'][:100] + "..." if len(case['input']) > 100 else case['input'],
        "Success_Rate": f"{successful_runs}/{n_runs}",
        "Exact_Match_Rate": f"{exact_match_runs}/{n_runs}",
        "Avg_Time": round(avg_time, 3),
        "Best_Exact_Match": best_result["exact_match"],
        "Best_Success": best_result["success"],
        "Error": best_result["error"] if best_result["error"] else "",
        "Generated_JSON": json.dumps(best_result["generated"], ensure_ascii=False) if best_result["generated"] else "",
        "Expected_JSON": json.dumps(case['expected_output'], ensure_ascii=False)
    }


def finalize_parser_stats(parser_stats: Dict[str, Any], n_runs: int) -> Dict[str, Any]:
    """파서 통계 최종 계산"""
    total_possible = parser_stats["total_cases"] * n_runs
    parser_stats["avg_time"] = parser_stats["total_time"] / parser_stats["total_cases"]
    parser_stats["success_rate"] = (parser_stats["successful_cases"] / total_possible) * 100
    parser_stats["exact_match_rate"] = (parser_stats["exact_matches"] / total_possible) * 100
    return parser_stats


def run_comprehensive_test(n_cases: Optional[int] = None, n_runs: int = 3, verbose: bool = False) -> Tuple[List[Dict], Dict[str, Any]]:
    """포괄적인 테스트 실행"""
    print("🚀 JSON Parsing Performance Test")
    print("=" * 50)
    
    # 데이터 및 파서 로드
    test_data = load_test_data()
    parsers = setup_parsers()
    
    # 테스트할 케이스 수 결정
    if n_cases is None:
        n_cases = len(test_data)
    else:
        n_cases = min(n_cases, len(test_data))
    
    test_cases = test_data[:n_cases]
    
    print(f"Testing {len(test_cases)} cases with {len(parsers)} parsers ({n_runs} runs each)")
    print(f"Total tests: {len(test_cases) * len(parsers) * n_runs}")
    
    detailed_results = []
    parser_summary = {}
    
    # 각 파서별로 테스트
    for parser_name, chain in parsers.items():
        print(f"\n🔍 Testing {parser_name}...")
        
        parser_stats = initialize_parser_stats()
        
        # 각 테스트 케이스 처리
        for case in test_cases:
            if verbose:
                print(f"  Case {case['id']} ({case['type']}):", end=" ")
            
            # 케이스를 여러 번 실행
            case_results, successful_runs, exact_match_runs, avg_time = run_case_multiple_times(
                chain, case, parser_name, n_runs, verbose
            )
            
            # 가장 성공적인 결과 선택
            best_result = max(case_results, key=lambda x: (x["exact_match"], x["success"]))
            
            # 상세 결과 저장
            detailed_results.append(create_detailed_result(
                parser_name, case, successful_runs, exact_match_runs, avg_time, best_result, n_runs
            ))
            
            # 파서별 통계 업데이트
            parser_stats["total_cases"] += 1
            parser_stats["successful_cases"] += successful_runs
            parser_stats["exact_matches"] += exact_match_runs
            parser_stats["total_time"] += avg_time
            
            if not best_result["success"] and best_result["error"]:
                parser_stats["errors"].append(f"Case {case['id']}: {best_result['error']}")
            
            # 진행상황 출력
            if verbose:
                status = "✅" if best_result["exact_match"] else ("⚠️" if best_result["success"] else "❌")
                print(f"{status} ({successful_runs}/{n_runs} success, {exact_match_runs}/{n_runs} exact)")
        
        # 파서별 최종 통계 계산
        parser_stats = finalize_parser_stats(parser_stats, n_runs)
        parser_summary[parser_name] = parser_stats
        
        print(f"  ✅ {parser_name} completed: {parser_stats['exact_match_rate']:.1f}% exact match, {parser_stats['avg_time']:.3f}s avg")
    
    return detailed_results, parser_summary


def save_results(detailed_results, parser_summary, output_dir="results"):
    """결과를 엑셀 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"json_parsing_test_results_{timestamp}.xlsx")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 1. 상세 결과
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        
        # 2. 파서별 요약
        summary_data = []
        for parser_name, stats in parser_summary.items():
            summary_data.append({
                "Parser": parser_name,
                "Total_Cases": stats["total_cases"],
                "Success_Rate(%)": round(stats["success_rate"], 1),
                "Exact_Match_Rate(%)": round(stats["exact_match_rate"], 1),
                "Avg_Time(s)": round(stats["avg_time"], 3),
                "Error_Count": len(stats["errors"])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Parser Summary', index=False)
        
        # 3. 타입별 분석
        type_analysis = []
        for case_type in ['LLM', 'Sequential', 'Loop', 'Parallel']:
            type_cases = [r for r in detailed_results if r['Type'] == case_type]
            if type_cases:
                for parser in parser_summary.keys():
                    parser_type_cases = [r for r in type_cases if r['Parser'] == parser]
                    if parser_type_cases:
                        exact_matches = sum(1 for r in parser_type_cases if r['Best_Exact_Match'])
                        type_analysis.append({
                            "Type": case_type,
                            "Parser": parser,
                            "Cases": len(parser_type_cases),
                            "Exact_Matches": exact_matches,
                            "Exact_Match_Rate(%)": round((exact_matches / len(parser_type_cases)) * 100, 1),
                            "Avg_Time(s)": round(sum(r['Avg_Time'] for r in parser_type_cases) / len(parser_type_cases), 3)
                        })
        
        if type_analysis:
            type_df = pd.DataFrame(type_analysis)
            type_df.to_excel(writer, sheet_name='Type Analysis', index=False)
    
    return filename


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="JSON Parsing Performance Test")
    parser.add_argument("--cases", type=int, default=None, 
                       help="Number of test cases to run (default: all)")
    parser.add_argument("--runs", type=int, default=3, 
                       help="Number of runs per case (default: 3)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output")
    parser.add_argument("--output", default="results", 
                       help="Output directory for results (default: results)")
    
    args = parser.parse_args()
    
    print("Starting JSON parsing performance test...")
    
    try:
        # 포괄적인 테스트 실행
        detailed_results, parser_summary = run_comprehensive_test(
            n_cases=args.cases, 
            n_runs=args.runs, 
            verbose=args.verbose
        )
        
        # 결과 출력
        print("\n" + "="*60)
        print("📈 FINAL RESULTS")
        print("="*60)
        
        # 파서별 순위
        sorted_parsers = sorted(parser_summary.items(), 
                              key=lambda x: (x[1]['exact_match_rate'], -x[1]['avg_time']), 
                              reverse=True)
        
        print("\n Parser Ranking (by Exact Match Rate):")
        for i, (parser_name, stats) in enumerate(sorted_parsers, 1):
            print(f"  {i}. {parser_name}")
            print(f"     - Exact Match: {stats['exact_match_rate']:.1f}%")
            print(f"     - Success: {stats['success_rate']:.1f}%") 
            print(f"     - Avg Time: {stats['avg_time']:.3f}s")
            if stats['errors']:
                print(f"     - Errors: {len(stats['errors'])}")
            print()
        
        # 타입별 요약
        print("📊 Performance by Workflow Type:")
        type_stats = {}
        for result in detailed_results:
            case_type = result['Type']
            if case_type not in type_stats:
                type_stats[case_type] = {"total": 0, "exact": 0}
            type_stats[case_type]["total"] += 1
            if result['Best_Exact_Match']:
                type_stats[case_type]["exact"] += 1
        
        for case_type, stats in type_stats.items():
            rate = (stats["exact"] / stats["total"]) * 100
            print(f"  {case_type}: {rate:.1f}% exact match ({stats['exact']}/{stats['total']})")
        
        # 엑셀 저장
        try:
            filename = save_results(detailed_results, parser_summary, args.output)
            print(f"\n💾 Results saved to: {filename}")
        except Exception as e:
            print(f"\n⚠️ Failed to save Excel: {e}")
            
            # JSON 백업
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"{args.output}/json_parsing_test_backup_{timestamp}.json"
            os.makedirs(args.output, exist_ok=True)
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "detailed_results": detailed_results,
                    "parser_summary": parser_summary
                }, f, ensure_ascii=False, indent=2)
            print(f" Backup saved as: {json_filename}")
        
        print("\n✅ Test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

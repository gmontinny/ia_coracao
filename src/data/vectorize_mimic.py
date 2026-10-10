"""
Script Utilitário CLI para Vetorização em Larga Escala do MIMIC-IV-ECG no Qdrant.
Permite executar streaming, ingestão em lotes e consultas k-NN diretamente pelo terminal.

Uso:
    python src/data/vectorize_mimic.py --records 1000 --batch-size 200
    python src/data/vectorize_mimic.py --recreate --records 500
    python src/data/vectorize_mimic.py --search "STEMI anterior supradesnivelamento V1-V4"
"""

import sys
import os
import argparse
import time

# Adiciona raiz ao PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Garante suporte a UTF-8 no terminal Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data.mimic_vectorizer import MIMICIVVectorizer


def main():
    parser = argparse.ArgumentParser(description="Vetorizador em Larga Escala MIMIC-IV-ECG -> Qdrant Vector Database")
    parser.add_argument("--records", type=int, default=500, help="Quantidade total de exames para vetorizar e indexar")
    parser.add_argument("--batch-size", type=int, default=100, help="Tamanho do lote para streaming / upsert")
    parser.add_argument("--recreate", action="store_true", help="Recriar a coleção do zero no Qdrant")
    parser.add_argument("--search", type=str, default="", help="Executar busca vetorial semântica de teste")
    parser.add_argument("--top-k", type=int, default=3, help="Número de casos similares a retornar na busca")
    parser.add_argument("--host", type=str, default=os.getenv("VECTOR_DB_HOST", "localhost"), help="Host do Qdrant")
    parser.add_argument("--port", type=int, default=int(os.getenv("VECTOR_DB_PORT", "6333")), help="Porta do Qdrant")

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 VETORIZADOR DE BIOSSINAIS MIMIC-IV-ECG (800k) -> QDRANT VECTOR DATABASE")
    print("=" * 80)

    vectorizer = MIMICIVVectorizer(vector_db_host=args.host, vector_db_port=args.port)
    is_online, status_msg = vectorizer.check_qdrant_connection()

    print(f"\n📡 Conexão Qdrant: {'🟢 ONLINE' if is_online else '🟡 MODO EMBARCADO'} ({status_msg})")
    print(f"📦 Coleção Alvo: '{vectorizer.COLLECTION_NAME}' (Dimensão: {vectorizer.vector_size}d, Métrica: Cosine)")

    if args.search:
        print(f"\n🔍 Executando Busca Vetorial k-NN para: '{args.search}'...")
        results = vectorizer.search_similar_cases(args.search, top_k=args.top_k)
        print(f"\nEncontrados {len(results)} casos hospitalares compatíveis:")
        for idx, res in enumerate(results, 1):
            print(f"\n  #{idx} [Similaridade: {res.get('similarity_percentage', 'N/A')}] - {res.get('ecg_id', 'N/A')}")
            print(f"     Diagnóstico: {res.get('diagnostic_superclass')} | {', '.join(res.get('diagnostic_statements', []))}")
            print(f"     Paciente: {res.get('age')} anos ({res.get('gender')}) | FC: {res.get('heart_rate')} bpm | ST: {res.get('st_deviation_mm')} mm")
            print(f"     Oclusão Coronariana: {res.get('coronary_occlusion')}")
            print(f"     Desfecho Clínico: {res.get('clinical_outcome')}")
        print("\n" + "=" * 80)
        return

    print(f"\n⚙️ Iniciando Vetorização em Lotes: {args.records} exames (Lotes de {args.batch_size})...")
    
    def on_progress(processed, total, rate):
        pct = (processed / total) * 100
        print(f"   ⏳ Progresso: {processed}/{total} ({pct:.1f}%) | Velocidade: {rate:.1f} exames/s", end="\r")

    t0 = time.time()
    result = vectorizer.ingest_mimic_records(
        total_records=args.records,
        batch_size=args.batch_size,
        recreate_collection=args.recreate,
        progress_callback=on_progress
    )
    print()

    print("\n" + "-" * 80)
    print(f"✅ {result['message']}")
    print(f"   • Exames Indexados: {result['records_indexed']}")
    print(f"   • Tempo Decorrido: {result['elapsed_seconds']}s")
    print(f"   • Taxa de Processamento: {result['throughput_records_per_sec']} exames/segundo")
    if result.get("qdrant_dashboard_url"):
        print(f"   • Dashboard Qdrant: {result['qdrant_dashboard_url']}")
    print("-" * 80)


if __name__ == "__main__":
    main()

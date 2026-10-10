"""
Script Utilitário para Sincronização e Indexação Automática do Qdrant Vector Database.
Cria as 3 coleções cardiológicas essenciais:
1. ecg_clinical_guidelines (AHA/ACC, ESC, SBC, 4ª Def. Infarto)
2. ecg_datasets_catalog (Catálogo do Hugging Face Spaces: vlbthambawita/ECGDatasets)
3. ecg_diagnostic_cases (Casos Clínicos de Referência para Case-Based Reasoning)
"""

import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Garante suporte a UTF-8 no terminal Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data.ecg_knowledge_rag import ECGKnowledgeVectorDB


def main():
    print("=" * 80)
    print("🚀 SINCRONIZADOR VETORIAL ECG-AI -> QDRANT VECTOR DATABASE (PORTA 6333)")
    print("=" * 80)

    host = os.getenv("VECTOR_DB_HOST", "localhost")
    port = int(os.getenv("VECTOR_DB_PORT", "6333"))
    rag_engine = ECGKnowledgeVectorDB(vector_db_host=host, vector_db_port=port)

    is_online, status_msg = rag_engine.check_qdrant_connection()
    print(f"\n📡 Verificando conexão com Qdrant em {rag_engine.qdrant_base_url}...")
    print(f"   Status: {'🟢 CONECTADO' if is_online else '🔴 INACESSÍVEL'} ({status_msg})")

    if not is_online:
        print("\n⚠️ AVISO: O Qdrant não está respondendo na porta configurada.")
        print("   Certifique-se de que o container Docker está em execução:")
        print("   -> docker-compose up -d qdrant-vectordb")
        print("\n   Enquanto o container não estiver ativo, a aplicação usará o modo")
        print("   embarcado (in-memory TF-IDF Cosine Similarity) automaticamente.")
        return

    print("\n📦 Iniciando criação de coleções e indexação de pontos vetoriais...")
    result = rag_engine.sync_all_collections_to_qdrant()

    print(f"\nResultado: {'✅ SUCESSO' if result['success'] else '❌ FALHA'}")
    print(f"Mensagem: {result['message']}")

    if result.get("collections_created"):
        print("\nColeções indexadas no Qdrant:")
        for col in result["collections_created"]:
            print(f"  • Coleção: '{col['name']}' | {col['count']} pontos vetoriais indexados")
            print(f"    Descrição: {col['description']}")

    print("\n" + "=" * 80)
    print(f"🌐 Acesse a interface gráfica no navegador: {rag_engine.qdrant_base_url}/dashboard")
    print("=" * 80)


if __name__ == "__main__":
    main()

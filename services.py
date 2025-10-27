from decouple import config
from huggingface_hub import InferenceClient

print("Configurando cliente de inferência da Hugging Face...")
try:
    # Carrega o token a partir da variável de ambiente ou arquivo .env
    huggingface_token = config('HUGGINGFACE_TOKEN', default=None)
    if not huggingface_token:
        raise ValueError("A variável de ambiente HUGGINGFACE_TOKEN não foi configurada.")

    # Passa o token explicitamente para o cliente
    inference_client = InferenceClient(token=huggingface_token)

    # Define o modelo que será usado para a sumarização
    MODEL_ID = "csebuetnlp/mT5_multilingual_XLSum"
    print("Cliente de inferência configurado com sucesso.")

except Exception as e:
    print(f"Erro ao configurar o cliente de inferência: {e}")
    inference_client = None


def summarize_text(text_to_summarize: str) -> str | None:
    """
    Gera um resumo de um texto usando a API de Inferência da Hugging Face.

    Args:
        text_to_summarize: O texto a ser resumido.

    Returns:
        O texto resumido, ou None se o cliente não estiver disponível ou ocorrer um erro.
    """
    if not inference_client:
        print("Cliente de inferência não está disponível.")
        return None

    try:
        # Chama a API de sumarização, passando os parâmetros diretamente
        summary_list = inference_client.summarization(
            text_to_summarize,
            model=MODEL_ID,
        )
        # A API pode retornar uma lista, então pegamos o primeiro item.
        # O objeto retornado tem um atributo 'summary_text'.
        return summary_list[0]['summary_text'] if isinstance(summary_list, list) else summary_list.get('summary_text')

    except Exception as e:
        print(f"Erro ao gerar o resumo via API: {e}")
        return None


if __name__ == '__main__':
    # Este bloco só será executado se o arquivo services rodar o arquivo diretamente
    TEXTO_EXEMPLO = """
    No período de 01 a 27 de outubro de 2025, o Condomínio Solar das Flores registrou e acompanhou diversas atividades e ocorrências com o objetivo de manter a ordem e o bem-estar dos moradores. Foram realizadas manutenções preventivas na piscina e nos portões automáticos, garantindo que os espaços comuns permanecessem seguros e funcionais. Comunicados sobre horários de silêncio foram enviados em datas estratégicas, lembrando os moradores da importância do respeito mútuo. No período, foram registradas algumas ocorrências, como vazamentos em unidades, lâmpadas queimadas nos corredores e queixas de barulho excessivo, todas sendo devidamente resolvidas ou monitoradas pela administração. As áreas comuns, incluindo salão de festas, churrasqueira e quadra de esportes, foram reservadas de acordo com a programação pelos moradores, sem conflitos de horários. Financeiramente, o condomínio manteve um controle rigoroso, com quatro unidades em atraso de pagamento, despesas dentro do previsto com limpeza, manutenção e energia elétrica, e um saldo positivo em conta de R$ 15.450,00, permitindo a continuidade das melhorias e a boa gestão do patrimônio comum.
    """

    print("\n--- Resumindo texto de exemplo ---")
    resumo = summarize_text(TEXTO_EXEMPLO)

    if resumo:
        print("\nTexto Original:")
        print(TEXTO_EXEMPLO)
        print("\nResumo Gerado:")
        print(resumo)
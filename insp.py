import joblib

# Carregue seu modelo
try:
    modelo = joblib.load('modelo_v1.pkl')

    # Tenta acessar o atributo '.feature_names_in_'
    # Disponível em versões mais recentes do scikit-learn
    if hasattr(modelo, 'feature_names_in_'):
        features = modelo.feature_names_in_
        print("✅ Features encontradas diretamente no modelo:")
        print(list(features))
    else:
        print("❌ Modelo não tem o atributo '.feature_names_in_'. Provavelmente é um Pipeline. Tente o Método 2.")

except FileNotFoundError:
    print("Erro: Arquivo 'melhor_modelo_gpu.pkl' não encontrado. Verifique o nome e o caminho.")
except Exception as e:
    print(f"Ocorreu um erro ao carregar o modelo: {e}")
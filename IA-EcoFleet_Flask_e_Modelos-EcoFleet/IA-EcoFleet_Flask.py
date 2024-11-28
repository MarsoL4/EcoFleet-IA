from flask import Flask, request, jsonify
import pickle

# Inicializando o servidor Flask
app = Flask(__name__)

# Carregando os modelos treinados
with open('modelo_regressao.pkl', 'rb') as f:
    modelo_regressao = pickle.load(f)

with open('modelo_classificacao.pkl', 'rb') as f:
    modelo_classificacao = pickle.load(f)

with open('modelo_clusterizacao.pkl', 'rb') as f:
    modelo_clusterizacao = pickle.load(f)

# Carregando o scaler para o modelo de clusterização
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Endpoint para prever consumo de fontes não renováveis (Regressão)
@app.route('/api/prever_consumo', methods=['POST'])
def prever_consumo():
    try:
        dados = request.get_json()
        energia_renovavel = dados['energia_renovavel']

        # Realizando a previsão
        previsao = modelo_regressao.predict([[energia_renovavel]])
        return jsonify({'previsao_consumo': previsao[0]})
    except Exception as e:
        return jsonify({'erro': str(e)}), 400

# Endpoint para classificar eficiência energética (Classificação)
@app.route('/api/classificar_eficiencia', methods=['POST'])
def classificar_eficiencia():
    try:
        dados = request.get_json()
        energia_renovavel = dados['energia_renovavel']
        energia_por_pib = dados['energia_por_pib']

        # Realizando a classificação
        classe = modelo_classificacao.predict([[energia_por_pib, energia_renovavel]])
        return jsonify({'classe_eficiencia': classe[0]})
    except Exception as e:
        return jsonify({'erro': str(e)}), 400

# Endpoint para prever cluster de adoção de tecnologias limpas (Clusterização)
@app.route('/api/prever_cluster', methods=['POST'])
def prever_cluster():
    try:
        dados = request.get_json()
        energia_solar = dados['energia_solar']
        energia_eolica = dados['energia_eolica']
        biocombustivel = dados['biocombustivel']
        energia_per_capita = dados['energia_per_capita']
        fossil_per_capita = dados['fossil_per_capita']

        # Escalando os dados
        dados_escalados = scaler.transform([[energia_solar, energia_eolica, biocombustivel, energia_per_capita, fossil_per_capita]])

        # Prevendo o cluster
        cluster = modelo_clusterizacao.predict(dados_escalados)
        return jsonify({'cluster': int(cluster[0])})
    except Exception as e:
        return jsonify({'erro': str(e)}), 400

# Rota inicial para verificar o funcionamento do servidor
@app.route('/', methods=['GET'])
def home():
    return jsonify({'mensagem': 'A API Flask para os modelos de IA criados pela EcoFleet está funcionando corretamente!'})

# Rodando o servidor
app.run(debug=True)
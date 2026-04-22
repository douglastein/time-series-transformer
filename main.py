import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.data_prep import gerar_serie_temporal, preparar_dataloaders
from src.model import TimeSeriesTransformer, previsao_futuro

def main():
    # 1. Configurações Iniciais com PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 2. Geração e Preparação de Dados
    input_window = 50
    serie = gerar_serie_temporal(n_pontos=1000)
    train_loader, test_loader, scaler, serie_train, serie_test, X_test, y_test = preparar_dataloaders(serie, input_window)

    # 3. Inicialização do Modelo
    modelo = TimeSeriesTransformer().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(modelo.parameters(), lr=0.0005)

    # 4. Treinamento
    # Define quantas vezes o modelo será treinado "do início ao final", considerando os aprendizados anteriores
    epochs = 60 
    modelo.train()
    print("Iniciando Treinamento...")
    for epoch in range(epochs):
        total_loss = 0

        # O train_loader entrega os dados já fatiados em batches de 32 amostras, em que cada amostra
        # é uma janela de dados da série temporal
        for seq, label in train_loader:

            # Enviamos os dados e a resposta certa para o device
            seq, label = seq.to(device), label.to(device)
            
            # Limpa a memória de erros da iteração passada. O PyTorch acumula gradientes por padrão
            # Se não zerados, ele soma o erro de agora com o de antes
            optimizer.zero_grad()

            # O modelo olha para os 50 dias (seq) e faz sua previsão
            y_pred = modelo(seq)

            # label.squeeze(-1) remove uma dimensão extra do tensor (ex: transforma [32, 1]
            # em [32]) para ter as mesmas dimensões do y_pred
            loss = criterion(y_pred, label.squeeze(-1))

            # Calcula a responsabilidade de cada peso da rede neural sobre aquele erro e depois
            # altera os pesos com o otimizador
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} concluída, Loss média: {avg_loss:.6f}")

    # 5. Avaliação
    modelo.eval()
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for seq, label in test_loader:
            seq, label = seq.to(device), label.to(device)
            y_pred = modelo(seq)

            # Tira os dados do device e converte o Tensor do PyTorch para um Array do NumPy
            all_predictions.extend(y_pred.cpu().numpy())
            all_actuals.extend(label.cpu().numpy())

    # Trazendo as predições e valores reais para a escala original
    actual_predictions = scaler.inverse_transform(np.array(all_predictions))
    y_test_original = scaler.inverse_transform(np.array(all_actuals).reshape(-1, 1))

    print(f"MSE: {mean_squared_error(y_test_original, actual_predictions):.6f}")
    print(f"RMSE: {mean_squared_error(y_test_original, actual_predictions) ** 0.5:.6f}")
    print(f"MAE: {mean_absolute_error(y_test_original, actual_predictions):.6f}")

    # 6. Forecast Futuro
    print("Gerando Forecast...")
    steps_to_forecast = 200

    # Considera a última janela do conjunto de teste para ser a semente do forecast
    last_sequence_real = X_test[-1].to(device)
    future_forecast_normalized = previsao_futuro(modelo, last_sequence_real, steps_to_forecast, device)
    future_forecast = scaler.inverse_transform(future_forecast_normalized.reshape(-1, 1))

    # 7. Plotagem
    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(len(serie)), serie, label='Dados Originais', color='gray', alpha=0.6)
    
    test_start = len(serie_train)
    test_time_axis = np.arange(test_start + input_window, len(serie))
    
    plt.plot(test_time_axis, y_test_original, label='Valores Reais (Teste)', color='blue')
    plt.plot(test_time_axis, actual_predictions, label='Previsões no Teste', color='red', linestyle='--')
    
    future_time_axis = np.arange(len(serie), len(serie) + steps_to_forecast)
    plt.plot(future_time_axis, future_forecast, label='Forecast Futuro', color='green', linewidth=2)
    
    plt.axvline(x=len(serie)-1, color='black', linestyle='--', label='Início do Forecast')
    plt.title("Forecast de Série Temporal com Transformer")
    plt.legend()
    plt.grid(True)
    plt.xlim(test_start - 50, len(serie) + steps_to_forecast)
    plt.savefig('forecast_plot.png')
    print("Gráfico salvo como 'forecast_plot.png'. Concluído!")

if __name__ == "__main__":
    main()
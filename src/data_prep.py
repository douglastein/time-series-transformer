import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

def gerar_serie_temporal(n_pontos=1000):
    """
    Gera uma série temporal fictícia composta por duas ondas senoidais e ruído.

    Args:
        n_pontos (int, opcional): O número total de pontos de tempo a serem gerados. 
            O padrão é 1000.

    Returns:
        numpy.ndarray: Um array contendo os valores da série temporal 
        gerada, no formato float32.
    """
    t = np.arange(n_pontos)
    seno1 = np.sin(2 * np.pi * t / 20)
    seno2 = 0.5 * np.sin(2 * np.pi * t / 50)
    ruido = 0.2 * np.random.randn(n_pontos)
    serie = seno1 + seno2 + ruido
    return serie.astype(np.float32)

def cria_sequencias(dados, input_window):
    """
    Cria pares de entrada (features) e saída (labels) a partir de uma série contínua,
    deslizando uma janela de observação ao longo dos dados.

    Args:
        dados (numpy.ndarray ou list): Os dados contínuos da série temporal.
        input_window (int): O tamanho da janela de entrada (quantos passos no tempo 
            o modelo vai olhar para o passado para fazer a previsão).

    Returns:
        list: Uma lista de tuplas, onde cada tupla contém um par `(seq, label)`.
            `seq` é a janela de observação e `label` é o valor imediatamente seguinte.
    """
    sequencias = []
    L = len(dados)

    # O loop percorre a série até onde é possível extrair uma janela completa e um alvo
    for i in range(L - input_window): 
        seq = dados[i:i + input_window]

        # O uso de [idx : idx+1] garante que o label mantenha uma estrutura de array
        label = dados[i + input_window:i + input_window + 1]

        sequencias.append((seq, label))
    return sequencias

def converte_para_tensor(data):
    """
    Converte dados sequenciais em Tensores do PyTorch.

    Args:
        data (list): Uma lista de tuplas (X, y).

    Returns:
        tuple: Uma tupla (X, y) contendo:
            - X (torch.Tensor): Tensor com as features de entrada.
            - y (torch.Tensor): Tensor com os labels (alvos).
            Ambos convertidos para o tipo `torch.float32`.
    """
    # [item[0] for item in data] extrai todas as sequências (X) da lista de tuplas
    X = torch.tensor(np.array([item[0] for item in data]), dtype=torch.float32)

    # [item[1] for item in data] extrai todos os labels  (y) da lista de tuplas
    y = torch.tensor(np.array([item[1] for item in data]), dtype=torch.float32)
    return X, y

def preparar_dataloaders(serie, input_window=50, batch_size=32):
    """
    Pipeline completo de pré-processamento de dados para o modelo. 
    Realiza a divisão de treino/teste, normalização e empacotamento em DataLoaders do PyTorch.

    Args:
        serie (numpy.ndarray): A série temporal completa original.
        input_window (int, opcional): O tamanho da janela de entrada para o modelo. 
            O padrão é 50.
        batch_size (int, opcional): O número de amostras em cada lote de treinamento/teste. 
            O padrão é 32.
 
    Returns:
        tuple: Retorna múltiplos objetos necessários para o treino e avaliação:
            - train_loader (DataLoader): O iterador de lotes para os dados de treino.
            - test_loader (DataLoader): O iterador de lotes para os dados de teste.
            - scaler (MinMaxScaler): O objeto do scikit-learn ajustado para desnormalizar 
              os dados futuramente.
            - serie_train (numpy.ndarray): A porção bruta separada para treino.
            - serie_test (numpy.ndarray): A porção bruta separada para teste.
            - X_test (torch.Tensor): Tensores de entrada do conjunto de teste.
            - y_test (torch.Tensor): Tensores alvo (labels) do conjunto de teste.
    """
    cutoff = int(len(serie) * 0.8)
    serie_train = serie[:cutoff]
    serie_test = serie[cutoff:]

    # Inicializa o escalonador para colocar os dados entre -1 e 1, facilitando na convergência
    # do Transformer
    scaler = MinMaxScaler(feature_range=(-1, 1))
    serie_train_scaled = scaler.fit_transform(serie_train.reshape(-1, 1)).astype(np.float32)
    serie_test_scaled = scaler.transform(serie_test.reshape(-1, 1)).astype(np.float32)

    # Aplica a lógica de janela deslizante da série temporal para criar os pares (X, y)
    train_sequences = cria_sequencias(serie_train_scaled, input_window)
    test_sequences = cria_sequencias(serie_test_scaled, input_window)

    X_train, y_train = converte_para_tensor(train_sequences)
    X_test, y_test = converte_para_tensor(test_sequences)

    # O TensorDataset agrupa os tensores X e y para que possam ser acessados pelo mesmo índice
    # O DataLoader gerencia a criação de 'batches' (lotes), permitindo o treino em partes
    # shuffle=True no treino ajuda o modelo a não decorar a ordem dos dados (pacotes de passado e futuro)
    # entre os batches
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    # shuffle=False no teste pois queremos avaliar a performance na ordem cronológica real
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler, serie_train, serie_test, X_test, y_test
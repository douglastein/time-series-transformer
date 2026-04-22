import math
import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Injeta informações sobre a posição relativa ou absoluta dos elementos na sequência.
    
    Como os modelos Transformer não possuem recorrência ou convolução (como RNNs e CNNs), 
    esta classe adiciona uma representação posicional (usando funções seno e cosseno de diferentes 
    frequências) aos embeddings de entrada para que o modelo tenha noção da ordem temporal dos dados.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Inicializa a camada de Positional Encoding.

        Args:
            d_model (int): A dimensão do embedding (tamanho do vetor que representa cada ponto da série).
            dropout (float, opcional): A probabilidade de zerar alguns elementos do tensor. 
                Ajuda a prevenir overfitting. O padrão é 0.1.
            max_len (int, opcional): O comprimento máximo esperado para as sequências de entrada. 
                O padrão é 5000.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        # Cria um vetor de posições [0, 1, 2, ..., max_len-1] e transforma em coluna (unsqueeze)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Esta fórmula matemática complexa garante que cada posição tenha uma "assinatura" única
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Preenche as colunas pares (0, 2, 4...) com o seno das posições
        pe[:, 0::2] = torch.sin(position * div_term)

        # Preenche as colunas ímpares (1, 3, 5...) com o cosseno das posições
        pe[:, 1::2] = torch.cos(position * div_term)

        # Adiciona uma dimensão extra no início para bater com o formato de (Batch, Seq, Dim)
        pe = pe.unsqueeze(0)

        # register_buffer salva a matriz 'pe' no modelo, mas ainda sem treinar
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Aplica a codificação posicional à sequência de entrada durante a passagem para frente.

        Args:
            x (torch.Tensor): Tensor de entrada com formato (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: O tensor de entrada somado à codificação posicional, 
            após a aplicação do dropout.
        """
        # x.size(1) considera apenas o trecho da matriz pe correspondente ao tamanho da janela temporal atual
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """
    Arquitetura de Rede Neural baseada no Transformer, adaptada especificamente 
    para forecasting de séries temporais.
    """
    def __init__(self, input_dim=1, d_model=32, nhead=8, num_layers=2, dropout=0.1):
        """
        Inicializa as camadas do modelo Transformer.

        Args:
            input_dim (int, opcional): O número de variáveis na série temporal de entrada 
                (1 para série univariada). O padrão é 1.
            d_model (int, opcional): A dimensão do espaço de representação (embeddings). 
                O padrão é 32.
            nhead (int, opcional): O número de cabeças no mecanismo de Multi-Head Attention. 
                O padrão é 2.
            num_layers (int, opcional): O número de subcamadas empilhadas. 
                O padrão é 2.
            dropout (float, opcional): A taxa de dropout aplicada nas camadas internas. 
                O padrão é 0.1.
        """
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        
        # Projeta o dado bruto para o espaço do modelo (ex: 1 valor de 1 dimensão passa
        # a ser representado por por exemplo por 32 dimensões)
        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Define a arquitetura padrão de uma camada do encoder (Attention + FeedForward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )

        # Empilha as camadas (num_layers) para criar a rede profunda
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Camada final que volta das d_model dimensões para o valor único de previsão
        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, src):
        """
        Define o fluxo de passagem dos dados pela rede.

        Args:
            src (torch.Tensor): Tensor contendo a sequência de entrada no formato 
                (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Tensor contendo as previsões geradas pelo modelo. 
            O formato de saída é (batch_size, input_dim).
        """
        # Projeta os dados e escala pela raíz quadrada de d_model a fim de estabilizar os gradientes
        src = self.encoder(src) * math.sqrt(self.d_model)

        # Injeta a noção de tempo com o Positional Encoder
        src = self.pos_encoder(src)

        # Passa pelas camadas de Atenção da arquitetura Transformer
        output = self.transformer_encoder(src)

        # output.mean(dim=1) tira a média de toda a janela de tempo (dim=1)
        # transformando a sequência em um único vetor que resume o passado. Fixa-se os batches e as
        # "características" (d_model)
        output = output.mean(dim=1)

        # Gera o valor final previsto, o trazendo para a dimensão original
        output = self.decoder(output)
        return output

def previsao_futuro(model, start_sequence, future_steps, device):
    """
    Gera previsões para passos futuros além dos dados de teste observados.
    
    A função pega a última janela conhecida, prevê o próximo ponto, anexa 
    esse ponto à janela (removendo o ponto mais antigo) e repete o processo.

    Args:
        model (nn.Module): O modelo Transformer treinado.
        start_sequence (torch.Tensor): A última sequência observada nos dados, 
            usada como semente (seed) inicial para começar as previsões.
        future_steps (int): A quantidade de passos no tempo que se deseja 
            prever para o futuro.
        device (torch.device): O dispositivo de processamento ('cpu', 'cuda', ou 'mps') 
            onde os tensores e o modelo estão alocados.

    Returns:
        numpy.ndarray: Um array contendo as previsões geradas para os próximos 
        `future_steps` temporais.
    """
    # Coloca o modelo em modo de avaliação (desativa dropout)
    model.eval()
    future_predictions = []

    # Faz uma cópia da última janela conhecida para não alterar o dado original
    current_sequence = start_sequence.clone().detach()

    # Desativa cálculo de gradientes, passo desnecessário na avaliação
    with torch.no_grad(): 
        for _ in range(future_steps):
            # Adiciona dimensão de batch (1) para o modelo aceitar o tensor
            next_pred = model(current_sequence.unsqueeze(0))

            # .item() extrai o valor numérico do tensor do PyTorch
            future_predictions.append(next_pred.item())
            
            # Transforma a previsão em tensor
            next_pred_tensor = torch.tensor([[next_pred.item()]], device=device)

            # Avança a janela em um passo, considerando a previsão anterior como último ponto observado
            current_sequence = torch.cat((current_sequence[1:], next_pred_tensor), dim=0)
            
    return np.array(future_predictions)
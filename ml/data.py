import numpy as np # usado para criar array e juntar
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder # One = tranforma colunas categoricas texto em varias colunas 0/1 - Label=tranforma o rotulo label em 0/1


# X = é o dataframe - categorical=lista com nomes das colunas categoricas - label= nome da coluna alvo - training=se estiver treinando ele treina encode e binazir se nao apenas aplica
def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    # Caso a label seja passada = Label é a coluna alvo
    if label is not None:
        y = X[label] # Vai pegar a coluna alvo
        X = X.drop([label], axis=1) # Vai remover a coluna alvo do dataframe features
    else:
        y = np.array([]) # Se não foi inferencia sem label

    # Separa as colunas do dataframe X em categoricas e numericas
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    # Caso esteja treinando
    if training is True:
        # Vai criar o OnehotEncoder - sparse_output=false(retorna um array normal e nao matriz) - handle_unkon=se aparece uma categoria nova na inferencia nao quebra, vai gerar zeros
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        # Cria o binarizador do label
        lb = LabelBinarizer()
        # Aprende o mapeamento das categorias git e já tranforma em one-hot
        X_categorical = encoder.fit_transform(X_categorical)
        # Aprende como transforma a label em transforma em 0/1
        y = lb.fit_transform(y.values).ravel()

    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb

def apply_label(inference):
    """ Convert the binary label in a single inference sample into string output."""
    if inference[0] == 1:
        return ">50K"
    elif inference[0] == 0:
        return "<=50K"

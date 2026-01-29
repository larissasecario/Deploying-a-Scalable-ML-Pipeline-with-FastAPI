import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    save_model,
    load_model,
    performance_on_categorical_slice,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "census.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
LB_PATH = os.path.join(MODEL_DIR, "lb.pkl")
SLICE_OUTPUT_PATH = os.path.join(BASE_DIR, "slice_output.txt")

LABEL = "salary"
TEST_SIZE = 0.20
RANDOM_STATE = 42

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def load_data(path: str) -> pd.DataFrame:
    print(path)
    return pd.read_csv(path)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL],
    )


def prepare_features(train: pd.DataFrame, test: pd.DataFrame):
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    return X_train, y_train, X_test, y_test, encoder, lb


def train_and_save(X_train, y_train, encoder, lb):
    os.makedirs(MODEL_DIR, exist_ok=True)

    model = train_model(X_train, y_train)

    save_model(model, MODEL_PATH)
    save_model(encoder, ENCODER_PATH)
    save_model(lb, LB_PATH)

    return model


def evaluate(model, X_test, y_test):
    preds = inference(model, X_test)
    p, r, fb = compute_model_metrics(y_test, preds)
    print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")
    return p, r, fb


def write_slice_metrics(test: pd.DataFrame, model, encoder, lb, output_path: str):

    with open(output_path, "w", encoding="utf-8") as f:
        for col in CAT_FEATURES:
            for slicevalue in sorted(test[col].dropna().unique()):
                count = int((test[col] == slicevalue).sum())

                p, r, fb = performance_on_categorical_slice(
                    data=test,
                    column_name=col,
                    slice_value=slicevalue,
                    categorical_features=CAT_FEATURES,
                    label=LABEL,
                    encoder=encoder,
                    lb=lb,
                    model=model,
                )
                f.write(f"{col}: {slicevalue}, Count: {count:,}\n")
                f.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n")


def main():
    df = load_data(DATA_PATH)
    train, test = split_data(df)

    X_train, y_train, X_test, y_test, encoder, lb = prepare_features(train, test)
    train_and_save(X_train, y_train, encoder, lb)
    model = load_model(MODEL_PATH)

    evaluate(model, X_test, y_test)

    write_slice_metrics(test, model, encoder, lb, SLICE_OUTPUT_PATH)
    print(f"Slice performance saved to {SLICE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

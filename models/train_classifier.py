import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

def train_model(filepath, target_column='will_price_go_up_3d', model_name='xgb_model_multicoin.pkl'):
    df = pd.read_csv(filepath)

    # Encode coin name
    if 'coin' in df.columns:
        encoder = LabelEncoder()
        df['coin_encoded'] = encoder.fit_transform(df['coin'])
        # Optional: save encoder if you'll use it during real-time inference

    # Drop unnecessary columns
    drop_cols = ['date', 'price', 'market_cap', 'coin']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns] + [target_column])
    y = df[target_column]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("🧱 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_path = Path(__file__).resolve().parent / "rf_models" / model_name
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"✅ Model saved to: {model_path}")
    return model

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    filepath = base_dir / "features" / "processed" / "multicoin_dataset.csv"
    train_model(filepath)

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def load_data(file_path):
    """Load and preprocess the Titanic dataset."""
    data = pd.read_csv(file_path)
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Embarked'].fillna('S', inplace=True)
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
    data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    return data


def create_model(input_dim):
    """Create a TensorFlow model for binary classification."""
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(model, X_val, y_val):
    """Evaluate the model and print performance metrics."""
    y_pred = (model.predict(X_val) > 0.5).astype("int32")
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")


if __name__ == "__main__":
    # Load and preprocess the dataset
    file_path = "titanic.csv"  # Replace with GCS path when running on AI Platform
    data = load_data(file_path)
    X = data.drop('Survived', axis=1)
    y = data['Survived']

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = create_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Evaluate the model
    evaluate_model(model, X_val, y_val)

    # Save the trained model for deployment
    model.save("/tmp/titanic_model")
    print("Model saved to /tmp/titanic_model")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import pickle
import json
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class AutoencoderMLP:
    def __init__(self, input_dim: int, hidden_layers: list = [64, 32, 16, 32, 64]):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.weights = []
        self.biases = []
        self.scaler = None
        self.threshold = None
        self.trained = False

        self.initialize_network()

    def initialize_network(self):
        layer_sizes = [self.input_dim] + self.hidden_layers + [self.input_dim]

        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            weight = np.random.uniform(
                -limit, limit, (layer_sizes[i], layer_sizes[i + 1])
            )
            bias = np.zeros((1, layer_sizes[i + 1]))

            self.weights.append(weight)
            self.biases.append(bias)

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def forward_pass(self, data: np.ndarray) -> Tuple[list, list]:
        activations = [data]
        z_values = []

        current_activation = data
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(current_activation, W) + b
            z_values.append(z)

            if i < len(self.weights) - 1:
                current_activation = self.relu(z)
            else:
                current_activation = z

            activations.append(current_activation)

        return activations, z_values

    def clip_gradients(self, gradients: list, max_norm: float = 5.0) -> list:
        clipped = []
        for grad in gradients:
            if np.isnan(grad).any() or np.isinf(grad).any():
                clipped.append(np.zeros_like(grad))
                continue

            grad_norm = np.linalg.norm(grad)
            if grad_norm > max_norm:
                grad = grad * (max_norm / grad_norm)
            clipped.append(grad)
        return clipped

    def backward_pass(
        self, data: np.ndarray, activations: list, z_values: list
    ) -> Tuple[list, list]:
        m = data.shape[0]
        weight_gradients = []
        bias_gradients = []

        delta = (activations[-1] - data) / m

        delta = np.clip(delta, -10, 10)

        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)

            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta = delta * self.relu_derivative(z_values[i - 1])

                delta = np.clip(delta, -10, 10)

        weight_gradients = self.clip_gradients(weight_gradients, max_norm=5.0)
        bias_gradients = self.clip_gradients(bias_gradients, max_norm=5.0)

        return weight_gradients, bias_gradients

    def fit(
        self,
        data: np.ndarray,
        epochs: int = 200,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        verbose: bool = True,
    ) -> Dict:
        data_train, data_val = train_test_split(
            data, test_size=validation_split, random_state=42
        )

        self.scaler = StandardScaler()
        train_scaled = self.scaler.fit_transform(data_train)
        val_scaled = self.scaler.transform(data_val)

        history = {"train_loss": [], "val_loss": []}
        n_batches = len(train_scaled) // batch_size

        for epoch in range(epochs):
            indices = np.random.permutation(len(train_scaled))
            data_shuffled = train_scaled[indices]

            epoch_loss = 0

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                data_batch = data_shuffled[start_idx:end_idx]

                activations, z_values = self.forward_pass(data_batch)

                batch_loss = np.mean((activations[-1] - data_batch) ** 2)
                epoch_loss += batch_loss

                weight_gradients, bias_gradients = self.backward_pass(
                    data_batch, activations, z_values
                )

                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * weight_gradients[i]
                    self.biases[i] -= learning_rate * bias_gradients[i]

                    if (
                        np.isnan(self.weights[i]).any()
                        or np.isinf(self.weights[i]).any()
                    ):
                        if verbose:
                            print(
                                f"\nWarning: NaN detected in weights layer {i}, resetting..."
                            )
                        limit = np.sqrt(
                            6.0 / (self.weights[i].shape[0] + self.weights[i].shape[1])
                        )
                        self.weights[i] = np.random.uniform(
                            -limit, limit, self.weights[i].shape
                        )
                        self.biases[i] = np.zeros_like(self.biases[i])

            train_loss = epoch_loss / n_batches

            val_activations, _ = self.forward_pass(val_scaled)
            val_loss = np.mean((val_activations[-1] - val_scaled) ** 2)

            if np.isnan(train_loss) or np.isnan(val_loss):
                if verbose:
                    print(f"\nWarning: NaN loss detected at epoch {epoch + 1}")
                    print("Reinitializing network with lower learning rate...")
                self.initialize_network()
                learning_rate = learning_rate * 0.5
                if verbose:
                    print(f"Reduced learning rate to {learning_rate:.6f}")
                continue

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
                )

        val_errors = np.mean((val_activations[-1] - val_scaled) ** 2, axis=1)
        self.threshold = np.percentile(val_errors, 95)

        self.trained = True

        if verbose:
            print(
                f"\nTraining completed. Anomaly threshold set to: {self.threshold:.6f}"
            )

        return history

    def predict_reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise ValueError("Model must be trained before prediction")

        data_scaled = self.scaler.transform(data)
        activations, _ = self.forward_pass(data_scaled)
        reconstruction = activations[-1]

        errors = np.mean((reconstruction - data_scaled) ** 2, axis=1)

        return errors

    def predict(self, data: np.ndarray) -> np.ndarray:
        errors = self.predict_reconstruction_error(data)
        predictions = (errors > self.threshold).astype(int)
        return predictions

    def save_model(self, filepath: str):
        model_data = {
            "input_dim": self.input_dim,
            "hidden_layers": self.hidden_layers,
            "weights": self.weights,
            "biases": self.biases,
            "scaler": self.scaler,
            "threshold": self.threshold,
            "trained": self.trained,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        model = cls(model_data["input_dim"], model_data["hidden_layers"])
        model.weights = model_data["weights"]
        model.biases = model_data["biases"]
        model.scaler = model_data["scaler"]
        model.threshold = model_data["threshold"]
        model.trained = model_data["trained"]

        print(f"Model loaded from {filepath}")
        return model


class AnomalyDetectionAPI:
    def __init__(self, model_path: str = "autoencoder_model.pkl"):
        self.model_path = model_path
        self.model: Optional[AutoencoderMLP] = None

    def train(
        self,
        data_path: str,
        epochs: int = 200,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        save_model: bool = True,
        verbose: bool = True,
    ) -> Dict:
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)

        label_column = " Label" if " Label" in df.columns else "Label"
        data = df.drop(columns=[label_column])
        labels = df[label_column]

        benign_mask = labels.str.strip().str.upper() == "BENIGN"
        data_benign = data[benign_mask].values

        print(f"Training on {len(data_benign)} benign samples...")
        print(f"Feature dimensions: {data_benign.shape[1]}")

        self.model = AutoencoderMLP(input_dim=data_benign.shape[1])
        history = self.model.fit(
            data_benign,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=validation_split,
            verbose=verbose,
        )

        if save_model:
            self.model.save_model(self.model_path)

        return {
            "history": history,
            "n_benign_samples": len(data_benign),
            "n_features": data_benign.shape[1],
            "threshold": self.model.threshold,
        }

    def evaluate(self, data_path: str, verbose: bool = True) -> Dict:
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        print(f"Loading test data from {data_path}...")
        df = pd.read_csv(data_path)

        label_column = " Label" if " Label" in df.columns else "Label"
        data = df.drop(columns=[label_column])
        labels = df[label_column]

        y_binary = (labels.str.strip().str.upper() != "BENIGN").astype(int)

        print("Making predictions...")
        y_pred = self.model.predict(data.values)
        reconstruction_errors = self.model.predict_reconstruction_error(data.values)

        accuracy = accuracy_score(y_binary, y_pred)
        f1 = f1_score(y_binary, y_pred)
        conf_matrix = confusion_matrix(y_binary, y_pred)

        report = classification_report(
            y_binary, y_pred, target_names=["Benign", "Attack"], output_dict=True
        )

        results = {
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": report,
            "n_samples": len(y_binary),
            "n_benign": int(sum(y_binary == 0)),
            "n_attacks": int(sum(y_binary == 1)),
            "threshold": float(self.model.threshold),
            "mean_reconstruction_error": {
                "benign": float(np.mean(reconstruction_errors[y_binary == 0])),
                "attack": float(np.mean(reconstruction_errors[y_binary == 1])),
            },
        }

        if verbose:
            print("Results Summary:")
            print(f"Total samples: {results['n_samples']}")
            print(f"  Benign: {results['n_benign']}")
            print(f"  Attacks: {results['n_attacks']}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Confusion Matrix:")
            print(f"  TN: {conf_matrix[0, 0]:6d}  |  FP: {conf_matrix[0, 1]:6d}")
            print(f"  FN: {conf_matrix[1, 0]:6d}  |  TP: {conf_matrix[1, 1]:6d}")
            print(f"Mean Reconstruction Error:")
            print(f"  Benign: {results['mean_reconstruction_error']['benign']:.6f}")
            print(f"  Attack: {results['mean_reconstruction_error']['attack']:.6f}")
            print(f"Detection threshold: {results['threshold']:.6f}")

        return results

    def detect(self, data: np.ndarray) -> Dict:
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        predictions = self.model.predict(data)
        errors = self.model.predict_reconstruction_error(data)

        return {
            "predictions": predictions.tolist(),
            "reconstruction_errors": errors.tolist(),
            "threshold": float(self.model.threshold),
            "n_anomalies": int(sum(predictions)),
            "n_normal": int(sum(predictions == 0)),
        }

    def load_model(self):
        self.model = AutoencoderMLP.load_model(self.model_path)

    def save_results(self, results: Dict, output_path: str):
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")


def main():
    train_data = "../preprocessing/output/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX-normalized.csv"
    # train_data = "../preprocessing/output/combined-normalized.csv"
    model_path = "autoencoder_model.pkl"
    results_path = "detection_results.json"

    api = AnomalyDetectionAPI(model_path=model_path)

    training_results = api.train(
        data_path=train_data,
        epochs=200,
        batch_size=256,
        learning_rate=0.001,
        validation_split=0.2,
        save_model=True,
        verbose=True,
    )

    evaluation_results = api.evaluate(data_path=train_data, verbose=True)

    combined_results = {"training": training_results, "evaluation": evaluation_results}

    api.save_results(combined_results, results_path)

    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()

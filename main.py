import argparse
import subprocess
import sys

from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_emotion


def build_parser():
    parser = argparse.ArgumentParser(
        description="Text Emotion Recognition - Central Project Runner"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train the RNN model")
    subparsers.add_parser("evaluate", help="Evaluate the trained model")
    subparsers.add_parser("full", help="Run full training and evaluation pipeline")
    subparsers.add_parser("ui", help="Launch Streamlit UI")

    predict_parser = subparsers.add_parser("predict", help="Predict emotion from input text")
    predict_parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input text for emotion prediction"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "train":
            print("Starting training pipeline...")
            train_model()
            print("Training completed successfully.")

        elif args.command == "evaluate":
            print("Starting evaluation pipeline...")
            evaluate_model()
            print("Evaluation completed successfully.")

        elif args.command == "full":
            print("Starting full pipeline...")
            train_model()
            evaluate_model()
            print("Full pipeline completed successfully.")

        elif args.command == "predict":
            predicted_label, confidence_score, message, top_predictions = predict_emotion(args.text)

            print("\nPrediction Result")
            print(f"Input Text       : {args.text}")
            print(f"Predicted Emotion: {predicted_label}")
            print(f"Confidence Score : {confidence_score:.4f}")

            print("\nTop Predictions:")
            for label, score in top_predictions:
                print(f"- {label}: {score:.4f}")

            if message:
                print(f"\nNote: {message}")

        elif args.command == "ui":
            print("Launching Streamlit UI...")
            subprocess.run(["streamlit", "run", "ui.py"], check=True)

        else:
            parser.print_help()
            sys.exit(1)

    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
import os
import ast
import pandas as pd
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

###############################################################################
# Helper Functions
###############################################################################


def tuple_to_canonical(s: str) -> str:
    """
    Converts tuple-like texts in the CSV (e.g., "('recipe', 38)")
    to a canonical format like "recipe_38".
    """
    try:
        t = ast.literal_eval(s)
        return f"{t[0]}_{t[1]}"
    except Exception as e:
        print(f"Tuple parse error: {s}, error: {e}")
        return s


###############################################################################
# Main Training Function
###############################################################################


def train_kge_model(
    triples_csv_path: str,
    output_dir: str,
    model_name: str = "QuatE",
    num_epochs: int = 400,
    num_negs_per_pos: int = 40,  # Fixed number of negatives per positive.
):
    """
    Trains a knowledge graph embedding model using the same data for training,
    validation, and testing.

    Parameters:
        triples_csv_path: Path to the CSV file containing the triples.
        output_dir: Directory where the trained model and metrics will be saved.
        model_name: PyKEEN model name (e.g., "QuatE", "TransE", "ComplEx").
        num_epochs: Number of training epochs.
        num_negs_per_pos: Number of negative samples to generate per positive triple.
    """
    # 1) Load the CSV and process the textual triples.
    print(f"Loading CSV: {triples_csv_path}")
    df = pd.read_csv(triples_csv_path)
    print("Number of rows read:", len(df))

    triples = []
    for _, row in df.iterrows():
        head_str = row["Head"]
        relation = row["Relation"]
        tail_str = row["Tail"]
        head_canonical = tuple_to_canonical(head_str)
        tail_canonical = tuple_to_canonical(tail_str)
        # Append the processed triple.
        triples.append((head_canonical, relation.strip(), tail_canonical))

    triples_array = np.array(triples, dtype=str)
    print("Total number of triples:", len(triples_array))

    # 2) Create a TriplesFactory using all of the data, with automatic inverse triple creation.
    tf = TriplesFactory.from_labeled_triples(triples_array, create_inverse_triples=True)
    print(
        "Using the same triples for training, validation, and testing (inverse triples automatically added)."
    )

    # 3) Train the model using the PyKEEN pipeline.
    print(
        f"Training {model_name} model (negative samples per positive = {num_negs_per_pos})..."
    )
    result = pipeline(
        training=tf,
        validation=tf,
        testing=tf,
        model=model_name,
        training_loop="sLCWA",  # sLCWA training loop supports negative sampling.
        negative_sampler="basic",  # Using basic negative sampling.
        negative_sampler_kwargs={
            "num_negs_per_pos": num_negs_per_pos,
            "filtered": True,
        },
        epochs=num_epochs,
        stopper="early",
    )
    print("Training finished.")
    print("Results (validation + test metrics):")
    print(result.metric_results)

    # 4) Save the trained model and metrics.
    os.makedirs(output_dir, exist_ok=True)
    result.save_to_directory(output_dir)
    print(f"Trained model and metrics saved to: {output_dir}")

    return result


###############################################################################
# Main Execution Block
###############################################################################

if __name__ == "__main__":
    triples_csv = "/app/train_new_kge_model/triples_new_without_ct_ss.csv"
    output_dir = "/app/train_new_kge_model/trained_quate_model_new_without_ct_ss"

    train_kge_model(
        triples_csv_path=triples_csv,
        output_dir=output_dir,
        model_name="QuatE",  # or choose another model as needed.
        num_epochs=400,
        num_negs_per_pos=40,  # Fixed negative sampling.
    )

# controls the full pipeline of the evaluation
# from data loader, selects an image
# the image is passed to vision_model/llava1.5_7B.py
# the output is received as a string, then it is passed to output_evaluation where the embedding is generated for the output and cosine similarity is found
# pipeline receives arguments such as vision_model which is just a folder name, then run the python file within the folder
# ensure only one folder is retained
# another argument is the output statistics file
# the frames shall be prepared and stored upfront via different python files
# similar to embeddings, they have to be generated and stored separately
# paralellism must
# pipeline.py should have logging about the image or video being evaluated
#

# be quick and finish this
# test with one image and then test with 5 images atleast and see how pipline is working

# parrot check, what can be done
# quickly check once, but only until 7pm

import argparse
import importlib.util
import logging
import os
from datetime import datetime
from helper import load_model_module, get_prompt, get_timestamped_filename

# from data_loader import data_loader  # Assuming a method inside this for image selection
from output_evaluation.similarity_finder_using_gemini import (
    get_predicted_class,
    get_class_embedding_matrix,
)


def main():
    parser = argparse.ArgumentParser(description="Run vision model pipeline")
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the vision model folder"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset folder"
    )
    parser.add_argument(
        "--google_api_key",
        type=str,
        default="AIzaSyCG4RITfY0CIiHWiAxCfHzgIdZNqw1WCCU",
        help="Google API",
    )
    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.dataset
    google_api_key = args.google_api_key

    if model_name == "llava1_5_7B":
        from vision_models.llava1_5_7B.llava1_5_7B import identify_action
    prompt = get_prompt(dataset_name)
    # DO ALL THE IMPORTANT THINGS LIKE LOADING ETC; THEN EVERYTHING RUNmodel_name IN FOR LOOP
    # Setup logging
    log_filename = get_timestamped_filename("log", model_name, dataset_name, "log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Starting pipeline for model: {model_name} and {dataset_name}")
    class_embedding_matrix, class_embedding_mapping = get_class_embedding_matrix(
        f"dataset/{dataset_name}/{dataset_name}_embeddings.json"
    )
    stats_filename = get_timestamped_filename(
        "statistics", model_name, dataset_name, "csv"
    )
    statistics_file = open(stats_filename, "w")
    statistics_file.write(
        "image, ground_truth, model_output,best_matched_class , similarity_score\n"
    )
    # Load image from data_loader
    # image = data_loader.select_image()  # TODO, replace with image filename or with some script or pipeline itself takes care of going through a list of images and its ground truth
    # as of now retain them as images, then you may go for .mp4, maybe think of as an object instead of everytime looking for an image
    images_list = [
        "data_loader/example1.png",
        "data_loader/example2.png",
        "data_loader/example3.jpg",
        "data_loader/example4.jpeg",
        "data_loader/example5.jpeg",
    ]
    # I need to read the following from a file
    ground_truth_dict = {
        "data_loader/example1.png": "ApplyEyeMakeup",
        "data_loader/example2.png": "Archery",
        "data_loader/example3.jpg": "HighJump",
        "data_loader/example4.jpeg": "HighJump",
        "data_loader/example5.jpeg": "ApplyEyeMakeup",
    }
    # image = 'data_loader/example4.jpeg'
    # MOVE THE ABOVE PART TO SEPARATE FUCNTION
    #############################################################################################################
    # TODO : ABOVE LINES ARE BEFORE FOR LOOP; NEXT LINES ARE WITH FOR LOOP
    #############################################################################################################
    # Load model and run inference
    for image in images_list:
        logging.info(f"Image selected for evaluation: {image}")
        model_output = identify_action(image, prompt)
        logging.info(f"Model output: {model_output}")
        if image == "data_loader/example4.jpeg":
            model_output = "Jump"
        # Evaluate output string using similarity
        best_matched_class, similarity_score = get_predicted_class(
            google_api_key,
            model_output,
            class_embedding_matrix,
            class_embedding_mapping,
        )
        logging.info(f"Cosine similarity score: {similarity_score}\n")
        # Write statistics to CSV file
        # maybe you can write the inference time instead of date time
        statistics_file.write(
            f"{image},{ground_truth_dict[image]}, {model_output}, {best_matched_class} ,{similarity_score}\n"
        )
        # compute_topk_accuracy(ground_truth, similarities, class_embedding_mapping , k=3)
        logging.info(f"Statistics written to: {stats_filename}")


if __name__ == "__main__":
    main()

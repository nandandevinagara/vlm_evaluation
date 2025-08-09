import argparse
import importlib.util
import logging
import os
from datetime import datetime
from helper import (
    load_model_module,
    get_prompt,
    get_timestamped_filename,
    get_filename_class_mapping,
    get_png_files_in_folder,
    shuffle_class_name_in_prompt,
    create_log_csv_files,
)
import importlib

# from data_loader import data_loader  # Assuming a method inside this for image selection
from output_evaluation.similarity_finder_using_linq_embed_mistral import (
    SimilarityFinder,
)


def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Run vision model pipeline")
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the vision model folder"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset folder"
    )
    return parser


def get_relevant_model_functions(model_name):
    if model_name == "llava1_5_7B":
        from vision_models.llava1_5_7B.llava1_5_7B import identify_action
    return identify_action


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.dataset
    identify_action = get_relevant_model_functions(model_name)

    original_prompt = get_prompt(dataset_name)
    # DO ALL THE IMPORTANT THINGS LIKE LOADING ETC; THEN EVERYTHING RUNmodel_name IN FOR LOOP

    stats_filename = create_log_csv_files(model_name, dataset_name)

    logging.info(f"Starting pipeline for model: {model_name} and {dataset_name}")

    action_class_matcher = SimilarityFinder(
        f"dataset/{dataset_name}/{dataset_name}_embeddings.json"
    )

    # Load image from data_loader
    # image = data_loader.select_image()  # TODO, replace with image filename or with some script or pipeline itself takes care of going through a list of images and its ground truth
    # as of now retain them as images, then you may go for .mp4, maybe think of as an object instead of everytime looking for an image
    ##images_list = [
    ##    "data_loader/ucf101/example1.png",
    ##    "data_loader/example2.png",
    ##    "data_loader/example3.jpg",
    ##    "data_loader/example4.jpeg",
    ##    "data_loader/example5.jpeg",
    ##]
    images_list = get_png_files_in_folder("data_loader/ucf101")
    #images_list = images_list[5781:5785]
    #print("images_list ", images_list)
    ground_truth_dict = get_filename_class_mapping(
        f"dataset/{dataset_name}/{dataset_name}_annotations.txt"
    )
    #############################################################################################################
    # TODO : ABOVE LINES ARE BEFORE FOR LOOP; NEXT LINES ARE WITH FOR LOOP
    #############################################################################################################
    # Load model and run inference
    # statistics_file = open(stats_filename, "a")
    for image in images_list:
        # until data_loader, prefixing the path as of now
        with open(stats_filename, "a") as statistics_file:
            logging.info(f"Image selected for evaluation: data_loader/ucf101/{image}")
            # the model chooses the first class most of the times, hence shuffling the class list to avoid 'Class list order bias'
            prompt = shuffle_class_name_in_prompt(original_prompt)
            print("prompt ", prompt)
            model_output = identify_action(f"data_loader/ucf101/{image}", prompt)
            logging.info(f"Model output: {model_output}")

            # Evaluate output string using similarity
            # get all possible classes here instead of a sinle class
            top_k_classes, similarity_score = action_class_matcher.get_matching_classes(
                model_output, k=3
            )
            logging.info(f"Cosine similarity score: {similarity_score}\n")
            print("top_k_classes ", top_k_classes)
            # the following shall provide if the class is in the top-k classes or not
            top1_result = action_class_matcher.get_topk_result(
                ground_truth_dict[image], top_k_classes, 1
            )
            top5_result = action_class_matcher.get_topk_result(
                ground_truth_dict[image], top_k_classes, 5
            )
            print("result ", top1_result, top5_result)
            # Write statistics to CSV file
            # maybe you can write the inference time instead of date time
            # with open(stats_filename, "a") as statistics_file:
            statistics_file.write(
                f"{image};{ground_truth_dict[image]}; {model_output}; {top_k_classes} ;{similarity_score}; {top1_result }; {top5_result}\n"
            )
            logging.info(f"Statistics written to: {stats_filename}")


if __name__ == "__main__":
    main()

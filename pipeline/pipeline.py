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

#from data_loader import data_loader  # Assuming a method inside this for image selection
from output_evaluation.similarity_finder_using_gemini import get_predicted_class, get_class_embedding_matrix


def load_model_module(model_name):
    model_path = os.path.join("vision_models", model_name, f"{model_name}.py")
    spec = importlib.util.spec_from_file_location(model_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_timestamped_filename(prefix, model_name, extension):
    now = datetime.now()
    timestamp = now.strftime("%H_%M_%d_%m_%Y")
    return f"{prefix}_{model_name}_{timestamp}.{extension}"

def get_prompt(dataset_name):
    with open(f"dataset/{dataset_name}/{dataset_name}_prompt.txt") as f:
        prompt = f.readlines()
    # NOTE: readlines returns a list, hence take the first index assuming it is having prompt in one line
    return prompt[0]

def main():
    parser = argparse.ArgumentParser(description="Run vision model pipeline")
    parser.add_argument("--model", type=str, required=True, help="Name of the vision model folder")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset folder")
    parser.add_argument("--google_api_key", type=str, default = "AIzaSyCG4RITfY0CIiHWiAxCfHzgIdZNqw1WCCU", help = "Google API")
    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.dataset
    google_api_key= args.google_api_key
    
    if model_name == 'llava1_5_7B':
        from vision_models.llava1_5_7B.llava1_5_7B import identify_action
    prompt = get_prompt(dataset_name)
    #DO ALL THE IMPORTANT THINGS LIKE LOADING ETC; THEN EVERYTHING RUNmodel_name IN FOR LOOP
    # Setup logging
    log_filename = get_timestamped_filename("log", model_name, "log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Starting pipeline for model: {model_name}")
    class_embedding_matrix = get_class_embedding_matrix(f"datatset/{dataset_name}/{dataset_name}_embeddings.json")
    # Load image from data_loader
    #image = data_loader.select_image()  # TODO, replace with image filename or with some script or pipeline itself takes care of going through a list of images and its ground truth
    image = 'data_loader/frame_00060.jpg'
    logging.info(f"Image selected for evaluation: {image}")
#############################################################################################################
                                    # TODO : ABOVE LINES ARE BEFORE FOR LOOP; NEXT LINES ARE WITH FOR LOOP 
#############################################################################################################
    # Load model and run inference
    #model_module = load_model_module(model_name)
    #if not hasattr(model_module, "identify_action"):
    #    logging.error(f"Model module {model_name} missing 'identify_action' function")
    #    return

    model_output = identify_action(image, prompt)
    logging.info(f"Model output: {model_output}")

    # Evaluate output string using similarity
    similarity_score = get_predicted_class(google_api_key, model_output, class_embedding_matrix)
    logging.info(f"Cosine similarity score: {similarity_score}")
    exit()
    # Write statistics to CSV file
    stats_filename = get_timestamped_filename("statistics", model_name, "csv")
    with open(stats_filename, "w") as f:
        f.write("model_name,timestamp,image,similarity\n")
        f.write(f"{model_name},{datetime.now().isoformat()},{image},{similarity_score}\n")
    logging.info(f"Statistics written to: {stats_filename}")


if __name__ == "__main__":
    main()


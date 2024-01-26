import os


def filter_and_save(input_path):
    # Get a list of all text files in the specified directory
    files = [f for f in os.listdir(input_path) if f.endswith(".txt")]

    for file_name in files:
        input_file_path = os.path.join(input_path, file_name)

        # Read the content of the file
        with open(input_file_path, "r") as file:
            lines = file.readlines()

        # Filter out rows that don't start with '0'
        filtered_lines = [line for line in lines if line.startswith("0")]

        # Save the modified content back to the file
        with open(input_file_path, "w") as file:
            file.writelines(filtered_lines)


if __name__ == "__main__":
    # Specify the path where your text files are saved
    input_directory = r"/home/toqa/projects/Algorithms_test/yolo_output_T"

    # Call the function to filter and save each file in the specified directory
    filter_and_save(input_directory)

# Function to generate input data CSV
def generate_input_csv(input_data, output_csv_path):
    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data)
    # Save DataFrame to CSV
    input_df.to_csv(output_csv_path, index=False)
    print(f"Input data CSV saved to: {output_csv_path}")

if __name__ == "__main__":
    # Use the provided functions and data to generate input CSV
    datasets = getDataSets()
    subsets = getSubSets(datasets.copy(), fields, data_class_labels)
    normalized_sets = getNormalizedDataMinMax(subsets, (-1,1))
    reshaped_sets, _ = getReshapedData(normalized_sets, input_shape)
    input_train, _, output_train, _ = getTrainValidationSets(reshaped_sets, sets_train, sets_test, sides_train, sides_test)
    generate_input_csv(input_train, "input_data.csv")
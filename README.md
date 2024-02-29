# Covid19_TimeSeries_Prediction

## Project Description

This project involves creating a deep learning model using Long Short-Term Memory (LSTM) neural networks to predict new COVID-19 cases in Malaysia based on the past 30 days of case numbers. The LSTM model is chosen due to its ability to remember past information, which is crucial in time-series prediction tasks like this.

The model will be trained on historical COVID-19 case data, and its performance will be evaluated based on its accuracy in predicting future cases. The ultimate goal is to provide a tool that can aid decision-makers in implementing timely and effective measures to curb the spread of the virus.

## Steps Taken
### 1. Setup
> Install the necessary libraries

### 2. Data Loading
> The data will be load using pandas.read_csv()

### 3. Data Exploratory
> In this phase, we conducted a comprehensive review of the data details. This involved verifying that certain features were correctly in specific types and inspecting the dataset for any duplicate values.

### 4. Data Preprocessing
> In this step, we will handle columns of object type and convert them into integer type. Additionally, we will address the presence of NaN values in the column by employing an interpolation approach for filling these gaps.

### 5. Data Visualization and Cleaning
> In this stage, we will conduct a visualization of the dataset to identify any discernible patterns when plotted over time. This analysis may yield insights that could enhance the predictive accuracy of our model. Additionally, we will employ boxplot diagrams to identify and address any outliers in the data.

### 6. Data Splitting and Normalization
> This section is dedicated to normalizing the dataset and partitioning it into specific ratios for training, validation, and testing subsets. Additionally, we will generate violin plots to examine the distribution of the features of interest.

### 7. Data Window
> Leveraging a custom Window Generator, we establish a data window for two scenarios: single-time step and multi-time step predictions. For a more detailed understanding of how the Window Generator was implemented in this project, please refer to the time_series_helper.py file.

### 8. Model Development
> In this section, we will concentrate on visualizing the development of the LSTM model across its various layers and also setting the compile_and_fit funtion for model training later

### 9. Model Training
> This section is dedicated to the training phase of the model.

### 10. Model Evaluation
> The Model Evaluation section is primarily dedicated to visualizing the predictive performance of the model in comparison to the actual labels.

### 11. Performance Evaluation
> Last but not least, we will calulcate the MAPE for both model cases






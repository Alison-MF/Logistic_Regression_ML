/*
 * LinearRegression.cpp
 *
 *  Created on: 22 de abr. de 2023
 *      Author:  Alison Michel Fernandes
 */

#include "LinearRegression.h"

Linear_Regression::Linear_Regression() {
    // TODO Auto-generated constructor stub

}

bool Linear_Regression::custom_sort(double a, double b) /* this custom sort function is defined to
 sort on basis of min absolute value or error*/
{
    double a1 = abs(a - 0);
    double b1 = abs(b - 0);
    return a1 < b1;
}

vector<vector<string>> Linear_Regression::load_file() {
    ifstream file("csv_delta.csv");
    string line;

    // Declare a vector to hold the records
    vector<vector<string>> record;

    // Read the header line and discard it
    getline(file, line);

    // Loop through each subsequent line in the file
    while (getline(file, line)) {

        vector<string> columns; // vector to store the separated columns

        // Create a stringstream object from the line
        stringstream ss(line);

        // Use getline to split the line into individual columns
        // separated by commas
        string column;
        while (getline(ss, column, ',')) {
            columns.push_back(column);

        }

        // Add the vector of columns to your record vector
        record.push_back(columns);

    }

    file.close();
    return record;
}

struct Vetores_LRegression Linear_Regression::linear_regression_method(
        vector<vector<string>> record, struct Vetores_LRegression vetores) {

    // Calculate the number of elements to collect for the first dataset (70% of the total)
    int num_elements_70 = record.size() * 0.7;

    //  Add the first num_elements_70 elements to the selected_records_70 vector for training
    for (int i = 0; i < num_elements_70; i++) {

        for (int j = 0; j < record[i].size(); j++) {
            if (j == 1)
                vetores.x1_train.push_back(stod(record[i][j]));
            else if (j == 2)
                vetores.x2_train.push_back(stod(record[i][j]));
            else if (j == 3)
                vetores.x3_train.push_back(stod(record[i][j]));
            else if (j == 4)
                vetores.x4_train.push_back(stod(record[i][j]));
            else if (j == 5)
                vetores.x5_train.push_back(stod(record[i][j]));
            else if (j == 6)
                vetores.x6_train.push_back(stod(record[i][j]));
            else if (j == 7)
                vetores.x7_train.push_back(stod(record[i][j]));
            else if (j == 8)
                vetores.y_train.push_back(stod(record[i][j]));
        }
    }
    double err;
    double b0 = 0;           // initializing b0
    double b1 = 0;           // initializing b1
    double b2 = 0;           // initializing b1
    double b3 = 0;           // initializing b1
    double b4 = 0;           // initializing b1
    double b5 = 0;           // initializing b1
    double b6 = 0;           // initializing b1
    double b7 = 0;           // initializing b1
    double alpha = 0.000001; // intializing error rate

    /*Training Phase*/
    for (int i = 0; i < num_elements_70 * 100; i++) { // since there are 5 values and we want 4 epochs so run for loop for 20 times
        int idx = i % num_elements_70;  // for accessing index after every epoch
        double p = b0 + b1 * vetores.x1_train[idx] + b2 * vetores.x2_train[idx]
                + b3 * vetores.x3_train[idx] + b4 * vetores.x4_train[idx]
                + b5 * vetores.x5_train[idx] + b6 * vetores.x6_train[idx]
                + b7 * vetores.x7_train[idx]; // calculating prediction

        err = p - vetores.y_train[idx]; // calculating error
        b0 = b0 - alpha * err;  // updating b0
        b1 = b1 - alpha * err * vetores.x1_train[idx];
        b2 = b2 - alpha * err * vetores.x2_train[idx];
        b3 = b3 - alpha * err * vetores.x3_train[idx];
        b4 = b4 - alpha * err * vetores.x4_train[idx]; // updating b1
        b5 = b5 - alpha * err * vetores.x5_train[idx];
        b6 = b6 - alpha * err * vetores.x6_train[idx];
        b7 = b7 - alpha * err * vetores.x7_train[idx];

    }

    // Calculate the number of elements to collect for the second dataset (30% of the total)
    int num_elements_30 = record.size() * 0.3;

    // Add the next num_elements_30 elements to the selected_records_30 vector

    for (int i = num_elements_70; i < (num_elements_70 + num_elements_30);
            i++) {
        for (int j = 0; j < record[i].size(); j++) {
            if (j == 1)
                vetores.x1_test.push_back(stod(record[i][j]));
            else if (j == 2)
                vetores.x2_test.push_back(stod(record[i][j]));
            else if (j == 3)
                vetores.x3_test.push_back(stod(record[i][j]));
            else if (j == 4)
                vetores.x4_test.push_back(stod(record[i][j]));
            else if (j == 5)
                vetores.x5_test.push_back(stod(record[i][j]));
            else if (j == 6)
                vetores.x6_test.push_back(stod(record[i][j]));
            else if (j == 7)
                vetores.x7_test.push_back(stod(record[i][j]));
            else if (j == 8)
                vetores.y_test.push_back(stod(record[i][j]));
        }
    }

    cout << "size x1_test" << vetores.x1_test.size() << endl;
    cout << "size y_test" << vetores.y_test.size() << endl;

    vetores.array_coef.push_back(b0);
    vetores.array_coef.push_back(b1);
    vetores.array_coef.push_back(b2);
    vetores.array_coef.push_back(b3);
    vetores.array_coef.push_back(b4);
    vetores.array_coef.push_back(b5);
    vetores.array_coef.push_back(b6);
    vetores.array_coef.push_back(b7);

    return vetores;

}

double Linear_Regression::predict_model(vector<double> coeficientes,
        double x1_sample, double x2_sample, double x3_sample, double x4_sample,
        double x5_sample, double x6_sample, double x7_sample) {

    double pred = coeficientes[0] + coeficientes[1] * x1_sample
            + coeficientes[2] * x2_sample + coeficientes[3] * x3_sample
            + coeficientes[4] * x4_sample + coeficientes[5] * x5_sample
            + coeficientes[6] * x6_sample + coeficientes[7] * x7_sample; // make prediction

    /*cout << "b0 = " << coeficientes[0] << endl;
    cout << "b1 = " << coeficientes[1] << endl;
    cout << "b2 = " << coeficientes[2] << endl;
    cout << "b3 = " << coeficientes[3] << endl;
    cout << "b4 = " << coeficientes[4] << endl;
    cout << "b5 = " << coeficientes[5] << endl;
    cout << "b6 = " << coeficientes[5] << endl;
    cout << "b7 = " << coeficientes[5] << endl;*/

    return pred;
}

double Linear_Regression::mse(struct Vetores_LRegression vetores) {

    double pred;
    vector<double> y_pred;
   for (int i = 0; i<vetores.y_test.size(); i++) {

        pred = predict_model(vetores.array_coef, vetores.x1_test[i], vetores.x2_test[i],vetores.x3_test[i], vetores.x4_test[i], vetores.x5_test[i], vetores.x6_test[i], vetores.x7_test[i]);
        y_pred.push_back(pred);

    }


    double sum_squared_error = 0.0;
    int num_samples = vetores.y_test.size();

    for (int i = 0; i <num_samples; i++) {
        double predicted = y_pred[i];
        double error = vetores.y_test[i] - predicted;
        sum_squared_error += pow(error, 2.0);
    }

    double mean_squared_error = sum_squared_error / num_samples;

    return mean_squared_error;

}

Linear_Regression::~Linear_Regression() {
    // TODO Auto-generated destructor stub
}


/*
 * LinearRegression.h
 *
 *  Created on: 22 de abr. de 2023
 *      Author: Alison Michel Fernandes
 */

#ifndef STACK_PHY_LAYER_MACHINE_LEARNING_LINEARREGRESSION_H_
#define STACK_PHY_LAYER_MACHINE_LEARNING_LINEARREGRESSION_H_

#include <bits/stdc++.h> // header file for all c++ libraries
#include <fstream>
using namespace std;
// stdout library for printing values

struct Vetores_LRegression {
      vector<double> x1_train;
      vector<double> x2_train;
      vector<double> x3_train;
      vector<double> x4_train;
      vector<double> x5_train;
      vector<double> x6_train;
      vector<double> x7_train;
      vector<double> y_train;

      vector<double> x1_test;
      vector<double> x2_test;
      vector<double> x3_test;
      vector<double> x4_test;
      vector<double> x5_test;
      vector<double> x6_test;
      vector<double> x7_test;
      vector<double> y_test;

      vector<double> array_coef;
  };


class Linear_Regression {
public:
    Linear_Regression();
    bool custom_sort(double a, double b);
    vector<vector<string>> load_file();
    struct Vetores_LRegression linear_regression_method(vector<vector<string>> record, struct Vetores_LRegression vetores);
    double predict_model(vector<double> coeficientes, double x1_sample, double x2_sample, double x3_sample, double x4_sample, double x5_sample, double x6_sample, double x7_sample);
    double mse(struct Vetores_LRegression vetores);
    virtual ~Linear_Regression();
};

#endif /* STACK_PHY_LAYER_MACHINE_LEARNING_LINEARREGRESSION_H_ */

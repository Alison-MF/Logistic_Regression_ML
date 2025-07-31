//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

#ifndef STACK_PHY_LAYER_MACHINE_LEARNING_LOGISTICREGRESSION_H_
#define STACK_PHY_LAYER_MACHINE_LEARNING_LOGISTICREGRESSION_H_

#include <bits/stdc++.h> // header file for all c++ libraries
#include <fstream>
using namespace std;
// stdout library for printing values

struct Vetores {
      vector<double> x1_train;
      vector<double> x2_train;
      vector<double> x3_train;
      vector<double> x4_train;
      vector<double> x5_train;
      vector<double> x6_train;
      vector<double> x7_train;
      vector<double> x8_train;
      vector<double> y_train;

      vector<double> x1_test;
      vector<double> x2_test;
      vector<double> x3_test;
      vector<double> x4_test;
      vector<double> x5_test;
      vector<double> x6_test;
      vector<double> x7_test;
      vector<double> x8_test;
      vector<double> y_test;

      vector<double> array_coef;
      vector<int> prediction;
      double sinr_mean;
      double rsrp_mean;
      double distance_mean;



  };

class Logistic_Regression {
public:
    Logistic_Regression();
    bool custom_sort(double a, double b); /* this custom sort function is defined to*/
    vector<vector<string>> load_file();
    struct Vetores logistic_regression_method(vector<vector<string>> record, struct Vetores vetores);
    double acuracia(struct Vetores vetores, double b0, double b1, double b2, double b3, double b4, double b5,double b6,double b7, double b8);
    bool predict_model(vector<double> coeficientes, double x1_sample, double x2_sample, double x3_sample, double x4_sample, double x5_sample,double x6_sample,double x7_sample,double x8_sample);
    vector<int> predict_test(struct Vetores vetores);
    virtual ~Logistic_Regression();

};

#endif /* STACK_PHY_LAYER_MACHINE_LEARNING_LOGISTICREGRESSION_H_ */

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

#include "LogisticRegression.h"
#include <iostream>
#include <fstream>
#include <string>

Logistic_Regression::Logistic_Regression() {
    // TODO Auto-generated constructor stub

}

bool Logistic_Regression::custom_sort(double a, double b) /* this custom sort function is defined to
 sort on basis of min absolute value or error*/
{
    double a1 = abs(a - 0);
    double b1 = abs(b - 0);
    return a1 < b1;
}

vector<vector<string>> Logistic_Regression::load_file() {
    ifstream file("csv_logistic.csv"); // Carrega o arquivo CSV
    string line;
    // Declaração do vetor record
    vector<vector<string>> record;
    // Faz a leitura do cabeçalho e descarta
    getline(file, line);
    // Loop em todas linhas do arquivo
    while (getline(file, line)) {
        vector<string> columns; //Vector para armazenae as colunas
        // Criação do objeto stringstream
        stringstream ss(line);
        // While com a função de separar as colunas ao encontrar as vírgulas
        string column;
        while (getline(ss, column, ',')) {
            columns.push_back(column);
        }
        // Armazena as informações em colunas
        record.push_back(columns);
    }
    file.close();
    return record;
}

struct Vetores Logistic_Regression::logistic_regression_method(
        vector<vector<string>> record, struct Vetores vetores) {
    // Seleção randômica dos elementos
    random_shuffle(record.begin(), record.end());

    double sum_sinr = 0;
    double sum_rsrp = 0;
    double sum_distance = 0;

    // Variável que calcula o tamanho de 70% do dataset
    int num_elements_70 = record.size() * 0.7;
    //  Loop sobre os 70% do dataset separando o dados em novos vetores individuais
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
            {
                vetores.x6_train.push_back(stod(record[i][j]));
                sum_rsrp+=stod(record[i][j]);
            }
            else if (j == 7)
            {
                vetores.x7_train.push_back(stod(record[i][j]));
                sum_sinr+=stod(record[i][j]);
            }
            else if (j == 8)
            {
                vetores.x8_train.push_back(stod(record[i][j]));
                sum_distance+=stod(record[i][j]);
            }

            else if (j == 9)
                vetores.y_train.push_back(stod(record[i][j]));


        }
    }

    vetores.sinr_mean = sum_sinr/vetores.x6_train.size();
    vetores.rsrp_mean = sum_rsrp/vetores.x7_train.size();
    vetores.distance_mean = sum_distance/vetores.x8_train.size();

    // Variável que calcula o tamanho de 30% do dataset
    int num_elements_30 = record.size() * 0.3;
    //  Loop sobre os 30% do dataset separando o dados em novos vetores individuais
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
                vetores.x8_test.push_back(stod(record[i][j]));
            else if (j == 9)
                vetores.y_test.push_back(stod(record[i][j]));
        }
    }

    vector<double> error; // armazena os errosd
    double err;           // variável de erro para cada estágio
    double b0 = 0;        // inicializando b0
    double b1 = 0;        // inicializando b1
    double b2 = 0;        // inicializando b2
    double b3 = 0;        // inicializando b3
    double b4 = 0;        // inicializando b4
    double b5 = 0;        // inicializando b5
    double b6 = 0;        // inicializando b6
    double b7 = 0;        // inicializando b7
    double b8 = 0;        // inicializando b8
    double alpha = 0.000001; // inicializando a learning rate
    double e = 2.71828;    // valor da constante de Euler
    /*Fase de Treinamento*/
    for (int i = 0; i < num_elements_70 * 100; i++) { // serão realizados 10 epochs
        int idx = i % num_elements_70;  // acesso ao índice para cada epoch
        double p = -(b0 + b1 * vetores.x1_train[idx]
                + b2 * vetores.x2_train[idx] + b3 * vetores.x3_train[idx]
                + b4 * vetores.x4_train[idx] + b5 * vetores.x5_train[idx]
                + b6 * vetores.x6_train[idx] + b7 * vetores.x7_train[idx]+ b8 * vetores.x8_train[idx]); // realizando a predição
        double pred = 1 / (1 + pow(e, p)); // calculando a predição e aplicando a sigmoid
        err = vetores.y_train[idx] - pred;              //calculando o erro
        b0 = b0 - alpha * err * pred * (1 - pred) * 1.0;       // atualizando b0
        b1 = b1 + alpha * err * pred * (1 - pred) * vetores.x1_train[idx]; // atualizando b1
        b2 = b2 + alpha * err * pred * (1 - pred) * vetores.x2_train[idx]; // atualizando b2
        b3 = b3 + alpha * err * pred * (1 - pred) * vetores.x3_train[idx]; // atualizando b3
        b4 = b4 + alpha * err * pred * (1 - pred) * vetores.x4_train[idx]; // atualizando b4
        b5 = b5 + alpha * err * pred * (1 - pred) * vetores.x5_train[idx]; // atualizando b5
        b6 = b6 + alpha * err * pred * (1 - pred) * vetores.x6_train[idx]; // atualizando b6
        b7 = b7 + alpha * err * pred * (1 - pred) * vetores.x7_train[idx]; // atualizando b7
        b8 = b8 + alpha * err * pred * (1 - pred) * vetores.x8_train[idx]; // atualizando b8
        error.push_back(err);
    }
    vetores.array_coef.push_back(b0); //armazendo o coeficiente b0
    vetores.array_coef.push_back(b1); //armazendo o coeficiente b1
    vetores.array_coef.push_back(b2); //armazendo o coeficiente b2
    vetores.array_coef.push_back(b3); //armazendo o coeficiente b3
    vetores.array_coef.push_back(b4); //armazendo o coeficiente b4
    vetores.array_coef.push_back(b5); //armazendo o coeficiente b5
    vetores.array_coef.push_back(b6); //armazendo o coeficiente b6
    vetores.array_coef.push_back(b7); //armazendo o coeficiente b7
    vetores.array_coef.push_back(b8); //armazendo o coeficiente b8


    cout << "b0" << b0 << endl;
    cout << "b1" << b1 << endl;
    cout << "b2" << b2 << endl;
    cout << "b3" << b3 << endl;
    cout << "b4" << b4 << endl;
    cout << "b5" << b5 << endl;
    cout << "b6" << b6 << endl;
    cout << "b7" << b7 << endl;
    cout << "b8" << b8 << endl;


    return vetores;
}

double Logistic_Regression::acuracia(struct Vetores vetores, double b0,
        double b1, double b2, double b3, double b4, double b5, double b6,
        double b7,double b8) {
    int count_test = 0;
    for (int i = 0; i < vetores.x1_test.size(); i++) {
        double pred = b0 + b1 * vetores.x1_test[i] + b2 * vetores.x2_test[i]
                + b3 * vetores.x3_test[i] + b4 * vetores.x4_test[i]
                + b5 * vetores.x5_test[i] + b6 * vetores.x6_test[i]
               + b7 * vetores.x7_test[i] + b8 * vetores.x8_test[i]; // make prediction
        if (pred > 0.9) {
            pred = 1;
        } else
            pred = 0;

        if (pred == vetores.y_test[i]) {
            count_test++;
        }
    }
    double accuracy = double(count_test) * 100 / double(vetores.y_test.size());

    return accuracy;
}

bool Logistic_Regression::predict_model(vector<double> coeficientes,
        double x1_sample, double x2_sample, double x3_sample, double x4_sample,
        double x5_sample, double x6_sample, double x7_sample,double x8_sample) {
    bool result;
    //Predição através de amostras e coeficientes
    double pred = coeficientes[0] + coeficientes[1] * x1_sample
            + coeficientes[2] * x2_sample + coeficientes[3] * x3_sample
            + coeficientes[4] * x4_sample + coeficientes[5] * x5_sample
            + coeficientes[6] * x6_sample + coeficientes[7] * x7_sample + coeficientes[8] * x8_sample;
    if (pred >= 0.9) {
        result = true;
    } else
        result = false;

    return result;
}

vector<int> Logistic_Regression::predict_test(struct Vetores vetores) {
    bool result;
    for (int i = 0; i < vetores.x1_test.size(); i++) {
        result = predict_model(vetores.array_coef, vetores.x1_test[i],
                vetores.x2_test[i], vetores.x3_test[i], vetores.x4_test[i],
                vetores.x5_test[i], vetores.x6_test[i], vetores.x7_test[i],vetores.x8_test[i]);
        if (result) {
            vetores.prediction.push_back(1);
        } else {
            vetores.prediction.push_back(0);
        }

    }

    return vetores.prediction;
}

Logistic_Regression::~Logistic_Regression() {
    // TODO Auto-generated destructor stub
}


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

#include "QLearning.h"
using namespace std;

Q_Learning::Q_Learning() {
    // TODO Auto-generated constructor stub
    STATE_DIM = 8;
    ACTION_NUM = 2;
    HANDOVER = 1;
    MAX_EPISODE = 1000;
    alpha = 0.8;
    MAX_ITERATIONS = 100;
    epsilon = 0.2;
    cumulative_reward = 0.0;

}

Q_Learning::~Q_Learning() {
    // TODO Auto-generated destructor stub
}

vector<vector<string>> Q_Learning::load_file() {
    ifstream file("csv_logistic.csv");
    string line;
    vector<vector<string>> record;

    getline(file, line);

    while (getline(file, line)) {
        vector<string> columns;
        stringstream ss(line);

        string column;
        while (getline(ss, column, ',')) {
            columns.push_back(column);
        }

        record.push_back(columns);
    }
    file.close();
    return record;
}

void Q_Learning::print_matrix(vector<vector<double>> m, int rows, int columns) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            cout << m[i][j] << "\t";
        }
        cout << endl;
    }
}

int Q_Learning::update_index_action(vector<vector<double>> Reward_Matrix,
        int state, int new_action[2]) {
    int index_action = 0;
    for (int i = 0; i < 2; i++) {
        if (Reward_Matrix[state][i] >= 0) {
            new_action[index_action] = i;
            index_action++;
        }
    }

    return index_action;
}

double Q_Learning::calc_max_qvalue(vector<vector<double>> Q_Table, vector<vector<double>> Reward_Matrix, int state) {
    double temp_value = 0;
    for (int i = 0; i < 2; ++i) {
        if ((Reward_Matrix[state][i] >= 0)
                && (Q_Table[state][i] > temp_value)) {
            temp_value = Q_Table[state][i];
        }
    }
    return temp_value;
}

int Q_Learning::predict_handover(vector<double> now_state, vector<vector<double>> Q_Table) {
    // Lógica simplificada: retorna a ação com a Q-value mais alta.
    return (Q_Table[now_state[0]][0] > Q_Table[now_state[0]][1]) ? 0 : 1;
}

void Q_Learning::update_QTable(vector<double> init_state, struct Vetores_RL dataset) {
    double Q_before, Q_after;
    int next_action;

    int iterator_;

    for (int k = 0; k < 10; k++) {
        dataset.new_action[0] = -1;
        dataset.new_action[1] = -1;

        iterator_ = update_index_action(dataset.Reward_Matrix, static_cast<int>(init_state[0]), dataset.new_action);

        if (iterator_ == 0) {
            break;
        }

       next_action = (rand() / static_cast<double>(RAND_MAX) < 0.2) ? dataset.new_action[rand() % iterator_] : predict_handover(init_state, dataset.Q_Table);

      // cumulative_reward += dataset.Reward_Matrix[static_cast<int>(init_state[0])][next_action];

        Q_before = dataset.Q_Table[static_cast<int>(init_state[0])][next_action];
        dataset.Q_Table[static_cast<int>(init_state[0])][next_action] = Q_before + 0.8 * (dataset.Reward_Matrix[static_cast<int>(init_state[0])][next_action] + calc_max_qvalue(dataset.Q_Table, dataset.Reward_Matrix, next_action) - Q_before);
        Q_after = dataset.Q_Table[static_cast<int>(init_state[0])][next_action];

       init_state[0] = (next_action == 1 ? static_cast<double>(rand() % 8) : static_cast<double>(next_action));

        if (next_action == 1)
            break;
    }

   // return dataset;
}


struct Vetores_RL Q_Learning::training_phase(const std::vector<std::vector<double>> initial_states, int num_initial_states, struct Vetores_RL dataset) {
    srand((unsigned) time(NULL));

    for (int i = 0; i < 100; ++i) {
        vector<double> initial_state(8);
        for (int j = 0; j < 8; ++j) {
            initial_state[j] = initial_states[rand() % num_initial_states][j];
            //cout<<"i="<<i<<" j="<<j <<endl;
        }

        update_QTable(initial_state,dataset);

        cout<<i<<endl;
    }
    return dataset;
}

struct Vetores_RL Q_Learning::get_reward_from_data(
        vector<vector<string>> record, struct Vetores_RL dataset) {
    const int rows = record.size();

    double rsrp, sinr, distance;
    //const int cols = ACTION_NUM; // Assumindo que ACTION_NUM é definido em outro lugar

   //random_shuffle(record.begin(), record.end());

    for (int i = 0; i <10000 ; i++) {
        int target_value = stoi(record[i].back()); // Supondo que a saída está na última coluna

      /*  for (int j = 0; j < 9; j++) {
            if (j == 6)
                rsrp = stod(record[i][j]);
            else if (j == 7)
                sinr = stod(record[i][j]);
            else if (j == 8)
                distance = stod(record[i][j]);
        }*/

        vector<double> newRow;
        //newRow.push_back((target_value == 1) ? pow(rsrp * sinr * distance, 2) : (rsrp + sinr + distance)); // Exemplo de recompensa binária
        newRow.push_back((target_value == 1) ? 1000.0 : -1.0); // Exemplo de recompensa binária
        dataset.Reward_Matrix.push_back(newRow);

    }

    return dataset;

}
